import { Peer } from 'https://cdn.jsdelivr.net/npm/peerjs@1.5.4/+esm';
import {
  FilesetResolver,
  PoseLandmarker,
  HandLandmarker,
  ObjectDetector
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32';
import {
  $,
  AutoCalibrator,
  ClockSynchronizer,
  FrameRateCounter,
  PoseFusion,
  PosePacketBuffer,
  PoseStabilizer,
  clamp,
  createNeutralPose,
  downloadJson,
  drawTrackingOverlay,
  packPosePacket,
  poseReadiness,
  toast,
  unpackPosePacket
} from './core.js';
import { AvatarScene } from './scene.js';

const PHONE_MODE = location.hash.startsWith('#join=');
const HOST_ID = PHONE_MODE ? decodeURIComponent(location.hash.slice(6)) : null;

const MODEL_URLS = {
  poseLite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
  poseFull: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
  hand: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
  object: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite'
};

const PROFILE_CONFIG = {
  speed: {
    hostModel: 'lite',
    hostFps: 18,
    phoneFps: 13,
    handEvery: 4,
    objectInterval: 0,
    cameraWidth: 960,
    cameraHeight: 540
  },
  balanced: {
    hostModel: 'full',
    hostFps: 22,
    phoneFps: 15,
    handEvery: 3,
    objectInterval: 1450,
    cameraWidth: 1280,
    cameraHeight: 720
  },
  quality: {
    hostModel: 'full',
    hostFps: 26,
    phoneFps: 18,
    handEvery: 2,
    objectInterval: 950,
    cameraWidth: 1280,
    cameraHeight: 720
  }
};

const ALLOWED_OBJECTS = new Map([
  ['bottle', 'bottle'],
  ['cup', 'cup'],
  ['book', 'book'],
  ['cell phone', 'cell phone'],
  ['sports ball', 'sports ball'],
  ['backpack', 'backpack'],
  ['chair', 'chair']
]);

const app = {
  profile: 'speed',
  config: PROFILE_CONFIG.speed,
  hostStream: null,
  sideStream: null,
  phoneStream: null,
  hostPeer: null,
  phonePeer: null,
  connection: null,
  hostRuntime: null,
  phoneRuntime: null,
  frontPacket: null,
  sidePacket: null,
  sideBuffer: new PosePacketBuffer(),
  clock: new ClockSynchronizer(),
  calibrator: new AutoCalibrator(),
  fusion: new PoseFusion(),
  stabilizer: new PoseStabilizer(),
  avatar: null,
  running: false,
  phoneRunning: false,
  phoneFacing: 'environment',
  phoneOrientation: 'portrait',
  phoneObjectsEnabled: false,
  history: [],
  sequence: 0,
  lastUiUpdate: 0,
  lastHistoryAt: 0,
  calibrationRevisionApplied: -1,
  syncTimer: null,
  trackingStartedAt: 0
};

function setText(selector, value) {
  const element = $(selector);
  if (element) element.textContent = value;
}

function setTone(selector, tone) {
  const element = $(selector);
  if (element) element.dataset.tone = tone;
}

function setSignal(selector, active) {
  const element = $(selector);
  if (element) element.classList.toggle('active', Boolean(active));
}

function setGlobalStatus(text, tone = 'idle') {
  setText('#global-status-text', text);
  setTone('#global-status', tone);
}

function setCameraState(selector, text, tone = 'idle') {
  setText(selector, text);
  setTone(selector, tone);
}

function cameraErrorMessage(error) {
  switch (error?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Доступ к камере запрещён. Разрешите камеру в настройках браузера.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'Камера не найдена на этом устройстве.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'Камера занята другим приложением. Закройте его и повторите.';
    case 'OverconstrainedError':
      return 'Выбранный режим камеры не поддерживается.';
    default:
      return 'Не удалось запустить камеру. Обновите страницу и повторите.';
  }
}

function connectionCanSend(connection) {
  if (!connection?.open) return false;
  const bufferedAmount = connection.dataChannel?.bufferedAmount ?? 0;
  return bufferedAmount < 180000;
}

function stopStream(stream) {
  stream?.getTracks().forEach((track) => track.stop());
}

function videoSize(video) {
  return video?.videoWidth && video?.videoHeight
    ? { width: video.videoWidth, height: video.videoHeight }
    : null;
}

class VisionRuntime {
  constructor({ role, profile = 'balanced' }) {
    this.role = role;
    this.profile = profile;
    this.config = PROFILE_CONFIG[profile] || PROFILE_CONFIG.balanced;
    this.targetFps = role === 'phone' ? this.config.phoneFps : this.config.hostFps;
    this.pose = null;
    this.hands = null;
    this.objects = null;
    this.fileset = null;
    this.ready = false;
    this.loadingObjects = false;
    this.objectsEnabled = false;
    this.lastVideoTime = -1;
    this.lastInferenceAt = -Infinity;
    this.lastObjectAt = -Infinity;
    this.poseTimestamp = 0;
    this.handTimestamp = 0;
    this.objectTimestamp = 0;
    this.frameIndex = 0;
    this.lastHands = [];
    this.lastObjects = [];
    this.inferenceDurations = [];
    this.fps = new FrameRateCounter();
  }

  async initialize(onProgress = () => {}) {
    if (this.ready) return;
    onProgress('Загрузка движка…');
    this.fileset = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm'
    );

    const modelType = this.role === 'phone'
      ? 'lite'
      : this.config.hostModel;
    const poseUrl = modelType === 'lite' ? MODEL_URLS.poseLite : MODEL_URLS.poseFull;

    onProgress(this.role === 'phone' ? 'Загрузка быстрой модели тела…' : 'Загрузка точной модели тела…');
    this.pose = await this.createWithDelegate(PoseLandmarker, {
      baseOptions: { modelAssetPath: poseUrl },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: this.role === 'phone' ? 0.48 : 0.55,
      minPosePresenceConfidence: 0.48,
      minTrackingConfidence: 0.5,
      outputSegmentationMasks: false
    });

    this.ready = true;
    onProgress('Модель движения готова');
    if (this.role === 'host' || this.profile !== 'speed') setTimeout(() => this.initializeHands(), 900);
  }

  async initializeHands() {
    if (this.hands || !this.fileset) return;
    try {
      this.hands = await this.createWithDelegate(HandLandmarker, {
        baseOptions: { modelAssetPath: MODEL_URLS.hand },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.48
      });
    } catch (error) {
      console.warn('Hand tracking is unavailable', error);
      this.hands = null;
    }
  }

  async createWithDelegate(TaskClass, options) {
    try {
      return await TaskClass.createFromOptions(this.fileset, {
        ...options,
        baseOptions: { ...options.baseOptions, delegate: 'GPU' }
      });
    } catch (gpuError) {
      console.warn('GPU delegate unavailable; using CPU', gpuError);
      return TaskClass.createFromOptions(this.fileset, {
        ...options,
        baseOptions: { ...options.baseOptions, delegate: 'CPU' }
      });
    }
  }

  async enableObjects() {
    if (!this.config.objectInterval || this.objects || this.loadingObjects || !this.fileset) return;
    this.loadingObjects = true;
    try {
      this.objects = await this.createWithDelegate(ObjectDetector, {
        baseOptions: { modelAssetPath: MODEL_URLS.object },
        runningMode: 'VIDEO',
        maxResults: 7,
        scoreThreshold: 0.42,
        categoryAllowlist: [...ALLOWED_OBJECTS.keys()]
      });
      this.objectsEnabled = true;
    } catch (error) {
      console.warn('Object tracking is unavailable', error);
      this.objects = null;
      this.objectsEnabled = false;
    } finally {
      this.loadingObjects = false;
    }
  }

  shouldEstimate(video, now = performance.now()) {
    if (!this.ready || !video || video.readyState < 2) return false;
    if (video.currentTime === this.lastVideoTime) return false;
    const targetInterval = Math.max(1000 / this.targetFps, this.adaptiveInterval());
    return now - this.lastInferenceAt >= targetInterval;
  }

  adaptiveInterval() {
    if (!this.inferenceDurations.length) return 0;
    const average = this.inferenceDurations.reduce((sum, value) => sum + value, 0)
      / this.inferenceDurations.length;
    return clamp(average * 1.12, 0, 95);
  }

  nextTimestamp(key, now) {
    const property = `${key}Timestamp`;
    const next = Math.max(this[property] + 1, Math.floor(now));
    this[property] = next;
    return next;
  }

  estimate(video, now = performance.now()) {
    if (!this.shouldEstimate(video, now)) return null;

    const startedAt = performance.now();
    this.lastInferenceAt = now;
    this.lastVideoTime = video.currentTime;
    this.frameIndex += 1;

    try {
      const poseResult = this.pose.detectForVideo(video, this.nextTimestamp('pose', now));
      const pose = poseResult.landmarks?.[0] || null;
      const world = poseResult.worldLandmarks?.[0] || null;

      if (this.hands && this.frameIndex % this.config.handEvery === 0) {
        const handResult = this.hands.detectForVideo(video, this.nextTimestamp('hand', now));
        this.lastHands = handResult.landmarks || [];
      }

      if (
        this.objectsEnabled &&
        this.objects &&
        now - this.lastObjectAt >= this.config.objectInterval &&
        this.fps.value >= Math.min(10, this.targetFps - 2)
      ) {
        this.lastObjectAt = now;
        const objectResult = this.objects.detectForVideo(video, this.nextTimestamp('object', now));
        this.lastObjects = this.normalizeDetections(objectResult.detections || [], video);
      }

      const duration = performance.now() - startedAt;
      this.inferenceDurations.push(duration);
      if (this.inferenceDurations.length > 20) this.inferenceDurations.shift();
      const fps = this.fps.tick();

      return {
        sequence: ++app.sequence,
        t: Date.now(),
        pose,
        world,
        hands: this.lastHands,
        objects: this.lastObjects,
        fps,
        size: videoSize(video),
        inferenceMs: Math.round(duration)
      };
    } catch (error) {
      console.warn('Tracking frame skipped', error);
      return null;
    }
  }

  normalizeDetections(detections, video) {
    const width = Math.max(1, video.videoWidth);
    const height = Math.max(1, video.videoHeight);
    return detections
      .map((detection) => {
        const category = detection.categories?.[0];
        const label = ALLOWED_OBJECTS.get(category?.categoryName);
        const box = detection.boundingBox;
        if (!label || !box) return null;
        return {
          label,
          score: Number((category.score || 0).toFixed(3)),
          box: [
            box.originX / width,
            box.originY / height,
            box.width / width,
            box.height / height
          ]
        };
      })
      .filter(Boolean);
  }
}

function hostCameraConstraints() {
  const config = app.config;
  return {
    video: {
      facingMode: { ideal: 'user' },
      width: { ideal: config.cameraWidth },
      height: { ideal: config.cameraHeight },
      frameRate: { ideal: 30, max: 30 }
    },
    audio: false
  };
}

function phoneCameraConstraints() {
  const portrait = app.phoneOrientation === 'portrait';
  return {
    video: {
      facingMode: { ideal: app.phoneFacing },
      width: { ideal: portrait ? 540 : 960 },
      height: { ideal: portrait ? 960 : 540 },
      aspectRatio: { ideal: portrait ? 9 / 16 : 16 / 9 },
      frameRate: { ideal: 24, max: 30 }
    },
    audio: false
  };
}

async function openVideoStream(video, constraints) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera API is unavailable');
  }
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();
  return stream;
}

function makeQrCode(peerId) {
  const target = $('#qr-code');
  if (!target || !window.QRCode) return;
  target.replaceChildren();
  const joinUrl = `${location.origin}${location.pathname}#join=${encodeURIComponent(peerId)}`;
  new window.QRCode(target, {
    text: joinUrl,
    width: 120,
    height: 120,
    colorDark: '#071014',
    colorLight: '#ffffff',
    correctLevel: window.QRCode.CorrectLevel.M
  });
  $('#pair-placeholder')?.classList.add('hidden');
  $('#pair-ready')?.classList.remove('hidden');
}

function showPairConnected() {
  $('#pair-placeholder')?.classList.add('hidden');
  $('#pair-ready')?.classList.add('hidden');
  $('#pair-connected')?.classList.remove('hidden');
}

function showPairReady() {
  $('#pair-connected')?.classList.add('hidden');
  if ($('#qr-code')?.childElementCount) $('#pair-ready')?.classList.remove('hidden');
}

function startClockSync(connection) {
  clearInterval(app.syncTimer);
  const sendPing = () => {
    if (connectionCanSend(connection)) {
      connection.send({ type: 'clock-ping', hostSentAt: Date.now() });
    }
  };
  for (let index = 0; index < 9; index += 1) setTimeout(sendPing, index * 280);
  app.syncTimer = setInterval(sendPing, 5000);
}

function sendPhoneConfiguration() {
  if (!connectionCanSend(app.connection)) return;
  app.connection.send({
    type: 'host-config',
    profile: app.profile,
    objects: Boolean(app.config.objectInterval)
  });
}

function handleHostConnection(connection) {
  app.connection = connection;

  connection.on('open', () => {
    showPairConnected();
    setCameraState('#side-state', 'Соединение', 'warn');
    setGlobalStatus('Телефон подключён. Загружается анализ второй камеры.', 'warn');
    startClockSync(connection);
    sendPhoneConfiguration();
  });

  connection.on('data', (message) => {
    switch (message?.type) {
      case 'phone-ready':
        showPairConnected();
        sendPhoneConfiguration();
        break;
      case 'tracking-ready':
        setGlobalStatus('Две камеры подключены. Идёт автокалибровка.', 'good');
        setCameraState('#side-state', 'AI готов', 'good');
        break;
      case 'camera-changed':
        app.calibrator.reset();
        app.stabilizer.reset();
        app.sideBuffer.clear();
        setCalibrationUi('Камера изменена', 'Автокалибровка начнётся заново', 0);
        break;
      case 'clock-pong':
        app.clock.add(message.hostSentAt, message.phoneAt, Date.now());
        break;
      case 'pose': {
        const packet = unpackPosePacket(message, app.clock.offset);
        app.sidePacket = packet;
        app.sideBuffer.push(packet);
        setText('#side-fps', `${packet.fps || '—'} FPS`);
        const sideVideo = $('#side-video');
        drawTrackingOverlay(sideVideo, $('#side-overlay'), packet, { color: '#52b8ff' });
        const readiness = poseReadiness(packet);
        setCameraState('#side-state', readiness.full ? 'Готово' : readiness.detected ? 'Не целиком' : 'Нет человека', readiness.full ? 'good' : 'warn');
        setText('#side-hint', readiness.message);
        break;
      }
      default:
        break;
    }
  });

  connection.on('close', () => {
    if (app.connection === connection) app.connection = null;
    setCameraState('#side-state', 'Отключено', 'bad');
    setGlobalStatus('Телефон отключён. 3D продолжает работать по камере ноутбука.', 'warn');
    showPairReady();
  });

  connection.on('error', (error) => {
    console.warn('Phone data connection error', error);
    setCameraState('#side-state', 'Ошибка сети', 'bad');
  });
}

function handleRemoteVideoCall(call) {
  call.answer();
  call.on('stream', async (stream) => {
    app.sideStream = stream;
    const video = $('#side-video');
    video.srcObject = stream;
    try {
      await video.play();
      $('#side-empty')?.classList.add('hidden');
      setCameraState('#side-state', 'Видео', 'warn');
      showPairConnected();
    } catch (error) {
      console.warn('Remote video playback failed', error);
    }
  });
}

function ensureHostPeer() {
  if (app.hostPeer && !app.hostPeer.destroyed) return;
  app.hostPeer = new Peer(undefined, { debug: 1 });
  app.hostPeer.on('open', (peerId) => {
    makeQrCode(peerId);
    if (!app.connection) setGlobalStatus('Камера включена. Подключите телефон по QR‑коду.', 'warn');
  });
  app.hostPeer.on('connection', handleHostConnection);
  app.hostPeer.on('call', handleRemoteVideoCall);
  app.hostPeer.on('disconnected', () => {
    setGlobalStatus('Связь временно потеряна. Переподключаюсь…', 'warn');
    if (!app.hostPeer.destroyed) app.hostPeer.reconnect();
  });
  app.hostPeer.on('error', (error) => {
    console.warn('Host peer error', error);
    setGlobalStatus('Ошибка соединения с телефоном. Обновите QR‑код.', 'bad');
  });
}

function setCalibrationUi(step, message, progress) {
  setText('#calibration-step', step);
  setText('#calibration-message', message);
  const bar = $('#calibration-progress');
  if (bar) bar.style.width = `${Math.round(clamp(progress) * 100)}%`;
}

function updateHostUi(front, sideMatch, fusedFrame) {
  const now = performance.now();
  if (now - app.lastUiUpdate < 120) return;
  app.lastUiUpdate = now;

  const frontReady = poseReadiness(front);
  const sideReady = poseReadiness(sideMatch?.packet);
  setCameraState(
    '#front-state',
    frontReady.full ? 'Готово' : frontReady.detected ? 'Не целиком' : 'Нет человека',
    frontReady.full ? 'good' : 'warn'
  );
  setText('#front-hint', frontReady.message);
  setText('#front-fps', `${front?.fps || '—'} FPS`);
  setText('#render-fps', `${app.avatar.renderFps || '—'} FPS`);
  setText('#sync-latency', app.clock.roundTrip ? `${app.clock.roundTrip} мс` : '—');
  setText('#tracking-quality', fusedFrame ? `${Math.round(fusedFrame.quality * 100)}%` : '—');

  setSignal('#signal-front', frontReady.detected);
  setSignal('#signal-side', sideReady.detected);
  setSignal('#signal-floor', Boolean(fusedFrame?.points?.get(27) || fusedFrame?.points?.get(28)));
  setSignal('#signal-body', frontReady.full && sideReady.full);

  if (!fusedFrame) {
    setTone('#fusion-badge', 'idle');
    setText('#fusion-title', 'Ожидание человека');
  } else if (app.calibrator.result && fusedFrame.twoView) {
    setTone('#fusion-badge', 'good');
    setText('#fusion-title', 'Точный 3D · 2 камеры');
  } else if (fusedFrame.twoView) {
    setTone('#fusion-badge', 'warn');
    setText('#fusion-title', '3D · калибровка');
  } else {
    setTone('#fusion-badge', 'warn');
    setText('#fusion-title', 'Предпросмотр · 1 камера');
  }

  if (!frontReady.detected) {
    setCalibrationUi('Камера ноутбука', 'Встаньте перед камерой — 3D появится сразу', 0);
  } else if (!frontReady.full) {
    setCalibrationUi('Камера ноутбука', frontReady.message, 0.08);
  } else if (!sideMatch?.packet) {
    setCalibrationUi('Одна камера работает', 'Подключите телефон для точной глубины', 0.18);
  } else if (!sideReady.full) {
    setCalibrationUi('Камера телефона', sideReady.message, 0.24);
  } else if (!app.calibrator.result) {
    setCalibrationUi('Автокалибровка', 'Стойте ровно 2 секунды — камеры фиксируют пропорции', app.calibrator.progress);
  } else {
    setCalibrationUi('Калибровка готова', 'Двигайтесь свободно — модель удерживает длину костей и пол', 1);
  }
}

function recordFrame(frame) {
  const now = performance.now();
  if (!frame || now - app.lastHistoryAt < 80) return;
  app.lastHistoryAt = now;

  app.history.push({
    t: frame.timestamp,
    twoView: frame.twoView,
    quality: Number(frame.quality.toFixed(3)),
    points: Object.fromEntries(
      [...frame.points].map(([id, point]) => [id, point.toArray().map((value) => Number(value.toFixed(4)))])
    ),
    objects: app.avatar.getObjects()
  });
  if (app.history.length > 6000) app.history.splice(0, app.history.length - 6000);
}

function applyCalibrationIfNeeded(frame) {
  const sideIsFresh = app.sidePacket && app.frontPacket
    ? Math.abs(app.frontPacket.t - app.sidePacket.t) < 260
    : false;
  const calibration = app.calibrator.observe(
    app.frontPacket,
    sideIsFresh ? app.sidePacket : null
  );
  if (!calibration || app.calibrationRevisionApplied === app.calibrator.revision) return;
  app.calibrationRevisionApplied = app.calibrator.revision;
  app.stabilizer.captureRestLengths(frame.points);
  toast('Автокалибровка готова. Теперь не двигайте камеры.');
}

function hostTrackingLoop() {
  if (!app.running) return;

  const now = performance.now();
  const video = $('#front-video');
  const packet = app.hostRuntime?.estimate(video, now);

  if (packet) {
    app.frontPacket = packet;
    $('#steel-robot-frame')?.contentWindow?.postMessage({ type: 'motiontwin-pose', landmarks: packet.pose }, location.origin);
    drawTrackingOverlay(video, $('#front-overlay'), packet, { color: '#43e6a0' });

    const sideMatch = app.sideBuffer.nearest(packet.t, 110);
    const fused = app.fusion.fuse(packet, sideMatch, app.calibrator.result, app.clock.quality || 0.45);
    if (fused) {
      const stabilized = app.stabilizer.update(fused);
      if (stabilized) {
        applyCalibrationIfNeeded(stabilized);
        app.avatar.setFrame(stabilized, app.calibrator.result);
        recordFrame(stabilized);
        updateHostUi(packet, sideMatch, stabilized);
      }
    } else {
      updateHostUi(packet, sideMatch, null);
    }
  }

  requestAnimationFrame(hostTrackingLoop);
}

async function startHost() {
  if (app.running) return;
  const button = $('#start-host');
  button.disabled = true;
  button.textContent = 'Камера…';
  app.profile = $('#quality-profile')?.value || 'speed';
  app.config = PROFILE_CONFIG[app.profile] || PROFILE_CONFIG.speed;
  $('#quality-profile').disabled = true;
  setGlobalStatus('Запрашиваю доступ к камере…', 'warn');

  try {
    app.hostStream = await openVideoStream($('#front-video'), hostCameraConstraints());
    $('#front-empty')?.classList.add('hidden');
    setCameraState('#front-state', 'Видео', 'warn');
    setGlobalStatus('Видео включено. Загружается анализ движения…', 'warn');
    button.textContent = 'AI загружается…';
    ensureHostPeer();

    app.hostRuntime = new VisionRuntime({ role: 'host', profile: app.profile });
    await app.hostRuntime.initialize((message) => setGlobalStatus(message, 'warn'));

    app.running = true;
    app.trackingStartedAt = performance.now();
    button.textContent = 'Камера включена';
    setCameraState('#front-state', 'AI готов', 'good');
    setGlobalStatus(app.connection ? 'Две камеры подключены. Встаньте целиком в кадр.' : 'Трекинг работает. Подключите телефон по QR‑коду.', 'good');
    $('#export-session').disabled = false;
    hostTrackingLoop();

    if (app.config.objectInterval) {
      setTimeout(() => app.hostRuntime?.enableObjects(), 5500);
    }
  } catch (error) {
    console.error(error);
    button.disabled = false;
    button.textContent = 'Повторить';
    $('#quality-profile').disabled = false;
    const message = cameraErrorMessage(error);
    setGlobalStatus(message, 'bad');
    setCameraState('#front-state', 'Ошибка', 'bad');
    toast(message, 5200);
  }
}

function exportSession() {
  downloadJson('motiontwin-session.json', {
    format: 'motiontwin-3d/3',
    createdAt: new Date().toISOString(),
    profile: app.profile,
    calibration: app.calibrator.result,
    frames: app.history
  });
}

async function openPhoneCamera() {
  stopStream(app.phoneStream);
  const video = $('#phone-video');
  app.phoneStream = await openVideoStream(video, phoneCameraConstraints());
  $('#phone-empty')?.classList.add('hidden');
  $('#phone-view')?.classList.toggle('mirror', app.phoneFacing === 'user');
  return app.phoneStream;
}

function updatePhoneButtons() {
  $('#phone-front')?.classList.toggle('active', app.phoneFacing === 'user');
  $('#phone-back')?.classList.toggle('active', app.phoneFacing === 'environment');
  $('#phone-portrait')?.classList.toggle('active', app.phoneOrientation === 'portrait');
  $('#phone-landscape')?.classList.toggle('active', app.phoneOrientation === 'landscape');
  $('#phone-view')?.classList.toggle('portrait', app.phoneOrientation === 'portrait');
  $('#phone-view')?.classList.toggle('landscape', app.phoneOrientation === 'landscape');
}

function connectPhonePeer() {
  if (app.phonePeer && !app.phonePeer.destroyed) return;
  app.phonePeer = new Peer(undefined, { debug: 1 });

  app.phonePeer.on('open', () => {
    app.connection = app.phonePeer.connect(HOST_ID, {
      reliable: false,
      serialization: 'json'
    });

    app.connection.on('open', () => {
      app.connection.send({ type: 'phone-ready' });
      if (app.phoneRuntime?.ready) app.connection.send({ type: 'tracking-ready' });
      setText('#phone-connection', 'Видео подключено');
      setTone('#phone-connection', 'good');
      setText('#phone-message', 'Ноутбук уже видит видео. Загружается AI‑анализ движения…');
    });

    app.connection.on('data', (message) => {
      if (message?.type === 'clock-ping' && connectionCanSend(app.connection)) {
        app.connection.send({
          type: 'clock-pong',
          hostSentAt: message.hostSentAt,
          phoneAt: Date.now()
        });
      }

      if (message?.type === 'host-config') {
        app.phoneObjectsEnabled = Boolean(message.objects);
        if (app.phoneObjectsEnabled && app.phoneRuntime?.ready) {
          setTimeout(() => app.phoneRuntime?.enableObjects(), 7000);
        }
      }
    });

    app.connection.on('close', () => {
      setText('#phone-connection', 'Связь потеряна');
      setTone('#phone-connection', 'bad');
      setText('#phone-message', 'Обновите QR‑код на ноутбуке и подключитесь заново.');
    });

    app.connection.on('error', (error) => {
      console.warn('Phone connection error', error);
      setText('#phone-connection', 'Ошибка связи');
      setTone('#phone-connection', 'bad');
    });

    app.phonePeer.call(HOST_ID, app.phoneStream);
  });

  app.phonePeer.on('error', (error) => {
    console.warn('Phone peer error', error);
    setText('#phone-connection', 'Комната закрыта');
    setTone('#phone-connection', 'bad');
    setText('#phone-message', 'QR‑код устарел. Обновите сайт на ноутбуке и отсканируйте новый код.');
    $('#start-phone').disabled = false;
  });
}

function phoneTrackingLoop() {
  if (!app.phoneRunning) return;
  const video = $('#phone-video');
  const packet = app.phoneRuntime?.estimate(video, performance.now());

  if (packet) {
    drawTrackingOverlay(video, $('#phone-overlay'), packet, { color: '#43e6a0' });
    const readiness = poseReadiness(packet);
    setText('#phone-fps', `${packet.fps || '—'} FPS`);
    setText('#phone-body-state', readiness.message);

    if (connectionCanSend(app.connection)) {
      app.connection.send(packPosePacket(packet));
    }
  }

  requestAnimationFrame(phoneTrackingLoop);
}

async function startPhone() {
  if (app.phoneRunning) return;
  const button = $('#start-phone');
  button.disabled = true;
  button.textContent = 'Включаю видео…';
  setText('#phone-message', 'Разрешите доступ к камере. Видео появится сразу.');

  try {
    await openPhoneCamera();
    button.textContent = 'Подключаю ноутбук…';
    setText('#phone-connection', 'Видео запущено');
    setTone('#phone-connection', 'warn');
    connectPhonePeer();

    app.phoneRuntime = new VisionRuntime({ role: 'phone', profile: 'speed' });
    await app.phoneRuntime.initialize((message) => {
      setText('#phone-message', message);
      button.textContent = 'AI загружается…';
    });

    app.phoneRunning = true;
    button.textContent = 'Камера подключена';
    setText('#phone-message', 'Готово. Поставьте телефон сбоку и встаньте целиком в кадр.');
    setText('#phone-connection', app.connection?.open ? 'AI подключён' : 'Ожидание ноутбука');
    setTone('#phone-connection', app.connection?.open ? 'good' : 'warn');
    if (connectionCanSend(app.connection)) app.connection.send({ type: 'tracking-ready' });
    phoneTrackingLoop();

    if (app.phoneObjectsEnabled) {
      setTimeout(() => app.phoneRuntime?.enableObjects(), 7000);
    }
  } catch (error) {
    console.error(error);
    button.disabled = false;
    button.textContent = 'Повторить';
    const message = cameraErrorMessage(error);
    setText('#phone-message', message);
    setText('#phone-connection', 'Ошибка камеры');
    setTone('#phone-connection', 'bad');
    toast(message, 5200);
  }
}

async function changePhoneCamera({ facing = null, orientation = null }) {
  if (facing) app.phoneFacing = facing;
  if (orientation) app.phoneOrientation = orientation;
  updatePhoneButtons();

  if (!app.phoneStream) return;
  setText('#phone-message', 'Переключаю камеру…');
  try {
    await openPhoneCamera();
    if (app.phonePeer?.open) app.phonePeer.call(HOST_ID, app.phoneStream);
    if (connectionCanSend(app.connection)) app.connection.send({ type: 'camera-changed' });
    setText('#phone-message', 'Камера изменена. Встаньте целиком — автокалибровка повторится.');
  } catch (error) {
    console.error(error);
    setText('#phone-message', cameraErrorMessage(error));
  }
}

function initializeHostPage() {
  $('#phone-app')?.classList.add('hidden');
  $('#host-app')?.classList.remove('hidden');
  app.avatar = new AvatarScene($('#stage'));
  app.avatar.setFrame(createNeutralPose());
  $('#start-host').addEventListener('click', startHost);
  $('#export-session').addEventListener('click', exportSession);
}

function initializePhonePage() {
  $('#host-app')?.classList.add('hidden');
  $('#phone-app')?.classList.remove('hidden');
  updatePhoneButtons();
  $('#start-phone').addEventListener('click', startPhone);
  $('#phone-front').addEventListener('click', () => changePhoneCamera({ facing: 'user' }));
  $('#phone-back').addEventListener('click', () => changePhoneCamera({ facing: 'environment' }));
  $('#phone-portrait').addEventListener('click', () => changePhoneCamera({ orientation: 'portrait' }));
  $('#phone-landscape').addEventListener('click', () => changePhoneCamera({ orientation: 'landscape' }));
}

if (PHONE_MODE) initializePhonePage();
else initializeHostPage();
