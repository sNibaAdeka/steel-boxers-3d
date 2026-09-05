import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js';

export const $ = (selector) => document.querySelector(selector);

export const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12], [11, 13], [13, 15],
  [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20],
  [16, 22], [18, 20], [11, 23], [12, 24],
  [23, 24], [23, 25], [25, 27], [27, 29],
  [29, 31], [27, 31], [24, 26], [26, 28],
  [28, 30], [30, 32], [28, 32]
];

export const BODY_BONES = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [27, 31],
  [24, 26], [26, 28], [28, 32],
  [15, 17], [15, 19], [15, 21],
  [16, 18], [16, 20], [16, 22]
];

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
];

export const CORE_IDS = [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32];
export const RIG_IDS = Array.from({ length: 33 }, (_, index) => index);

export function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

export function lerp(a, b, amount) {
  return a + (b - a) * amount;
}

export function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function mean(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function visibility(landmark) {
  return landmark?.visibility ?? landmark?.presence ?? 0;
}

export function midpoint2D(a, b) {
  if (!a && !b) return null;
  if (!a) return { x: b.x, y: b.y, z: b.z || 0 };
  if (!b) return { x: a.x, y: a.y, z: a.z || 0 };
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: ((a.z || 0) + (b.z || 0)) / 2
  };
}

export function midpoint3D(a, b) {
  return a.clone().add(b).multiplyScalar(0.5);
}

export function compactLandmarks(landmarks) {
  if (!landmarks) return null;
  return landmarks.map((point) => [
    Number(point.x.toFixed(5)),
    Number(point.y.toFixed(5)),
    Number((point.z || 0).toFixed(5)),
    Number(visibility(point).toFixed(3))
  ]);
}

export function expandLandmarks(compact) {
  if (!compact) return null;
  return compact.map((point) => ({
    x: point[0],
    y: point[1],
    z: point[2],
    visibility: point[3],
    presence: point[3]
  }));
}

export function packPosePacket(packet) {
  return {
    type: 'pose',
    sequence: packet.sequence,
    t: packet.t,
    pose: compactLandmarks(packet.pose),
    world: compactLandmarks(packet.world),
    hands: packet.hands?.map(compactLandmarks) || [],
    objects: packet.objects || [],
    fps: packet.fps || 0,
    size: packet.size || null
  };
}

export function unpackPosePacket(packet, clockOffset = 0) {
  return {
    sequence: packet.sequence,
    t: packet.t - clockOffset,
    pose: expandLandmarks(packet.pose),
    world: expandLandmarks(packet.world),
    hands: packet.hands?.map(expandLandmarks) || [],
    objects: packet.objects || [],
    fps: packet.fps || 0,
    size: packet.size || null
  };
}

export function bodyCenter(landmarks) {
  return midpoint2D(landmarks?.[23], landmarks?.[24]);
}

export function shoulderCenter(landmarks) {
  return midpoint2D(landmarks?.[11], landmarks?.[12]);
}

export function pixelBodyHeight(landmarks) {
  if (!landmarks?.length) return 0;
  const headY = Math.min(
    landmarks[0]?.y ?? 1,
    landmarks[7]?.y ?? 1,
    landmarks[8]?.y ?? 1
  );
  const footY = Math.max(
    landmarks[27]?.y ?? 0,
    landmarks[28]?.y ?? 0,
    landmarks[31]?.y ?? 0,
    landmarks[32]?.y ?? 0
  );
  return Math.max(0, footY - headY);
}

export function poseReadiness(packet) {
  const landmarks = packet?.pose;
  if (!landmarks?.length) {
    return {
      detected: false,
      full: false,
      score: 0,
      message: 'Человек не виден'
    };
  }

  const required = [0, 11, 12, 23, 24, 25, 26, 27, 28];
  const visible = required.filter((id) => visibility(landmarks[id]) > 0.45).length;
  const anklesVisible = Math.max(visibility(landmarks[27]), visibility(landmarks[28])) > 0.42;
  const head = landmarks[0];
  const feetY = Math.max(
    landmarks[27]?.y || 0,
    landmarks[28]?.y || 0,
    landmarks[31]?.y || 0,
    landmarks[32]?.y || 0
  );
  const bodyHeight = pixelBodyHeight(landmarks);
  const insideFrame = head && head.y > 0.015 && feetY < 0.995;
  const detected = visible >= 5;
  const full = detected && visible >= 8 && anklesVisible && insideFrame && bodyHeight > 0.42;
  const score = clamp((visible / required.length) * clamp(bodyHeight / 0.68, 0, 1));

  let message = 'Человек виден';
  if (!detected) message = 'Встаньте перед камерой';
  else if (!head || head.y <= 0.015) message = 'В кадре не хватает головы';
  else if (!anklesVisible || feetY >= 0.995) message = 'Отойдите: нужны стопы';
  else if (bodyHeight <= 0.42) message = 'Подойдите немного ближе';
  else if (!full) message = 'Встаньте целиком в кадр';
  else message = 'Тело видно целиком';

  return { detected, full, score, message };
}

export function drawTrackingOverlay(video, canvas, packet, options = {}) {
  if (!video?.videoWidth || !canvas) return;

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const context = canvas.getContext('2d');
  context.clearRect(0, 0, width, height);
  if (!packet?.pose) return;

  const lineWidth = Math.max(2, width / 420);
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.lineWidth = lineWidth;
  context.strokeStyle = options.color || '#43e6a0';
  context.shadowColor = options.color || '#43e6a0';
  context.shadowBlur = Math.max(2, width / 300);

  for (const [a, b] of POSE_CONNECTIONS) {
    const from = packet.pose[a];
    const to = packet.pose[b];
    if (!from || !to || visibility(from) < 0.28 || visibility(to) < 0.28) continue;
    context.beginPath();
    context.moveTo(from.x * width, from.y * height);
    context.lineTo(to.x * width, to.y * height);
    context.stroke();
  }

  context.shadowBlur = 0;
  context.fillStyle = '#ecffff';
  const pointRadius = Math.max(2.2, width / 310);
  for (const point of packet.pose) {
    if (!point || visibility(point) < 0.32) continue;
    context.beginPath();
    context.arc(point.x * width, point.y * height, pointRadius, 0, Math.PI * 2);
    context.fill();
  }

  context.strokeStyle = '#ffcb72';
  context.lineWidth = Math.max(1.4, width / 560);
  for (const hand of packet.hands || []) {
    for (const [a, b] of HAND_CONNECTIONS) {
      if (!hand[a] || !hand[b]) continue;
      context.beginPath();
      context.moveTo(hand[a].x * width, hand[a].y * height);
      context.lineTo(hand[b].x * width, hand[b].y * height);
      context.stroke();
    }
  }
}

export class FrameRateCounter {
  constructor(windowMs = 1000) {
    this.windowMs = windowMs;
    this.startedAt = performance.now();
    this.frames = 0;
    this.value = 0;
  }

  tick(now = performance.now()) {
    this.frames += 1;
    const elapsed = now - this.startedAt;
    if (elapsed >= this.windowMs) {
      this.value = Math.round((this.frames * 1000) / elapsed);
      this.frames = 0;
      this.startedAt = now;
    }
    return this.value;
  }
}

export class PosePacketBuffer {
  constructor(maxPackets = 90) {
    this.maxPackets = maxPackets;
    this.packets = [];
  }

  push(packet) {
    this.packets.push(packet);
    if (this.packets.length > this.maxPackets) {
      this.packets.splice(0, this.packets.length - this.maxPackets);
    }
  }

  nearest(timestamp, toleranceMs = 95) {
    let best = null;
    let bestDelta = Infinity;

    for (let index = this.packets.length - 1; index >= 0; index -= 1) {
      const packet = this.packets[index];
      const delta = Math.abs(packet.t - timestamp);
      if (delta < bestDelta) {
        best = packet;
        bestDelta = delta;
      }
      if (packet.t < timestamp - toleranceMs * 2) break;
    }

    return bestDelta <= toleranceMs ? { packet: best, delta: bestDelta } : null;
  }

  clear() {
    this.packets.length = 0;
  }
}

export class ClockSynchronizer {
  constructor() {
    this.offsetSamples = [];
    this.rttSamples = [];
  }

  add(hostSentAt, phoneAt, hostReceivedAt = Date.now()) {
    const roundTrip = Math.max(0, hostReceivedAt - hostSentAt);
    const estimatedHostAtPhone = hostSentAt + roundTrip / 2;
    const offset = phoneAt - estimatedHostAtPhone;

    this.offsetSamples.push(offset);
    this.rttSamples.push(roundTrip);
    if (this.offsetSamples.length > 12) this.offsetSamples.shift();
    if (this.rttSamples.length > 12) this.rttSamples.shift();
  }

  get offset() {
    return median(this.offsetSamples);
  }

  get roundTrip() {
    return Math.round(median(this.rttSamples));
  }

  get quality() {
    const rtt = this.roundTrip;
    if (!this.rttSamples.length) return 0;
    if (rtt <= 45) return 1;
    if (rtt <= 90) return 0.72;
    if (rtt <= 160) return 0.45;
    return 0.2;
  }
}

class LowPassFilter {
  constructor() {
    this.initialized = false;
    this.value = 0;
  }

  apply(value, alpha) {
    if (!this.initialized) {
      this.value = value;
      this.initialized = true;
      return value;
    }
    this.value = alpha * value + (1 - alpha) * this.value;
    return this.value;
  }

  reset(value = 0) {
    this.initialized = false;
    this.value = value;
  }
}

class OneEuroScalar {
  constructor(minCutoff = 1.7, beta = 0.32, derivativeCutoff = 1) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.derivativeCutoff = derivativeCutoff;
    this.valueFilter = new LowPassFilter();
    this.derivativeFilter = new LowPassFilter();
    this.lastTime = null;
    this.lastRaw = null;
  }

  alpha(cutoff, deltaTime) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / Math.max(deltaTime, 0.001));
  }

  filter(value, timestamp) {
    if (this.lastTime === null || this.lastRaw === null) {
      this.lastTime = timestamp;
      this.lastRaw = value;
      return this.valueFilter.apply(value, 1);
    }

    const deltaTime = clamp((timestamp - this.lastTime) / 1000, 1 / 120, 0.25);
    const derivative = (value - this.lastRaw) / deltaTime;
    const filteredDerivative = this.derivativeFilter.apply(
      derivative,
      this.alpha(this.derivativeCutoff, deltaTime)
    );
    const cutoff = this.minCutoff + this.beta * Math.abs(filteredDerivative);
    const filtered = this.valueFilter.apply(value, this.alpha(cutoff, deltaTime));

    this.lastTime = timestamp;
    this.lastRaw = value;
    return filtered;
  }
}

class OneEuroVector {
  constructor(minCutoff = 1.7, beta = 0.32) {
    this.x = new OneEuroScalar(minCutoff, beta);
    this.y = new OneEuroScalar(minCutoff, beta);
    this.z = new OneEuroScalar(minCutoff, beta);
  }

  filter(vector, timestamp) {
    return new THREE.Vector3(
      this.x.filter(vector.x, timestamp),
      this.y.filter(vector.y, timestamp),
      this.z.filter(vector.z, timestamp)
    );
  }
}

export class AutoCalibrator {
  constructor(targetSamples = 42, assumedHeightMeters = 1.74) {
    this.targetSamples = targetSamples;
    this.assumedHeightMeters = assumedHeightMeters;
    this.samples = [];
    this.result = null;
    this.revision = 0;
  }

  reset() {
    this.samples.length = 0;
    this.result = null;
    this.revision += 1;
  }

  get progress() {
    if (this.result) return 1;
    return clamp(this.samples.length / this.targetSamples);
  }

  observe(front, side) {
    if (this.result) return this.result;

    const frontReady = poseReadiness(front);
    const sideReady = poseReadiness(side);
    if (!frontReady.full || !sideReady.full) {
      if (this.samples.length > 2) this.samples.splice(0, 2);
      return null;
    }

    const frontCenter = bodyCenter(front.pose);
    const sideCenter = bodyCenter(side.pose);
    const frontHeight = pixelBodyHeight(front.pose);
    const sideHeight = pixelBodyHeight(side.pose);
    const worldHeight = this.measureWorldHeight(front.world);
    const noseOffset = (side.pose[0]?.x || sideCenter.x) - sideCenter.x;

    if (!frontCenter || !sideCenter || frontHeight < 0.3 || sideHeight < 0.3) return null;

    this.samples.push({
      frontX: frontCenter.x,
      frontY: frontCenter.y,
      sideX: sideCenter.x,
      sideY: sideCenter.y,
      frontScale: this.assumedHeightMeters / frontHeight,
      sideScale: this.assumedHeightMeters / sideHeight,
      worldScale: worldHeight > 0.7 ? this.assumedHeightMeters / worldHeight : 1,
      sideFlip: noseOffset >= 0 ? -1 : 1
    });

    if (this.samples.length < this.targetSamples) return null;

    const value = (key) => median(this.samples.map((sample) => sample[key]));
    this.result = {
      version: 3,
      createdAt: Date.now(),
      frontCenter: { x: value('frontX'), y: value('frontY') },
      sideCenter: { x: value('sideX'), y: value('sideY') },
      frontScale: value('frontScale'),
      sideScale: value('sideScale'),
      worldScale: clamp(value('worldScale'), 0.72, 1.42),
      sideFlip: value('sideFlip') >= 0 ? 1 : -1,
      assumedHeightMeters: this.assumedHeightMeters
    };
    this.samples.length = 0;
    return this.result;
  }

  measureWorldHeight(world) {
    if (!world?.length) return 0;
    const head = world[0];
    const ankles = midpoint2D(world[27], world[28]);
    const shoulders = midpoint2D(world[11], world[12]);
    if (!head || !ankles || !shoulders) return 0;
    const vertical = Math.abs(ankles.y - head.y);
    const headAllowance = Math.abs(shoulders.y - head.y) * 0.38;
    return vertical + headAllowance;
  }
}

export class PoseFusion {
  constructor() {
    this.defaultHeight = 1.74;
  }

  fuse(front, sideMatch, calibration, clockQuality = 1) {
    if (!front?.pose) return null;
    const frontReadiness = poseReadiness(front);
    if (!frontReadiness.detected) return null;

    const side = sideMatch?.packet || null;
    const sideReadiness = poseReadiness(side);
    const twoView = Boolean(side?.pose && sideReadiness.detected);
    const syncDelta = sideMatch?.delta ?? Infinity;
    const syncWeight = twoView
      ? clamp(1 - Math.max(0, syncDelta - 20) / 100) * clockQuality
      : 0;

    const frontRoot2D = bodyCenter(front.pose) || { x: 0.5, y: 0.58 };
    const sideRoot2D = bodyCenter(side?.pose) || { x: 0.5, y: 0.58 };
    const frontRootWorld = midpoint2D(front.world?.[23], front.world?.[24]);

    const fallbackFrontScale = this.defaultHeight / Math.max(pixelBodyHeight(front.pose), 0.45);
    const fallbackSideScale = side?.pose
      ? this.defaultHeight / Math.max(pixelBodyHeight(side.pose), 0.45)
      : fallbackFrontScale;

    const frontScale = calibration?.frontScale || fallbackFrontScale;
    const sideScale = calibration?.sideScale || fallbackSideScale;
    const worldScale = calibration?.worldScale || 1;
    const sideFlip = calibration?.sideFlip || this.inferSideFlip(side);
    const baseFrontCenter = calibration?.frontCenter || frontRoot2D;
    const baseSideCenter = calibration?.sideCenter || sideRoot2D;

    const rootX = -(frontRoot2D.x - baseFrontCenter.x) * frontScale;
    const rootZ = twoView
      ? (sideRoot2D.x - baseSideCenter.x) * sideScale * sideFlip
      : 0;

    const points = new Map();
    const confidence = new Map();

    for (const id of RIG_IDS) {
      const frontPoint = front.pose[id];
      const frontWorld = front.world?.[id];
      if (!frontPoint && !frontWorld) continue;

      const frontVisibility = visibility(frontPoint);
      if (frontVisibility < 0.14 && !frontWorld) continue;

      let x;
      let y;
      let z;

      if (frontWorld && frontRootWorld) {
        x = -(frontWorld.x - frontRootWorld.x) * worldScale;
        y = -(frontWorld.y - frontRootWorld.y) * worldScale;
        z = -(frontWorld.z - frontRootWorld.z) * worldScale;
      } else {
        x = -(frontPoint.x - frontRoot2D.x) * frontScale;
        y = (frontRoot2D.y - frontPoint.y) * frontScale;
        z = -(frontPoint.z || 0) * frontScale * 0.55;
      }

      x += rootX;
      z += rootZ;

      const sidePoint = side?.pose?.[id];
      const sideVisibility = visibility(sidePoint);
      if (twoView && sidePoint && sideVisibility > 0.18) {
        const sideDepth = (sidePoint.x - sideRoot2D.x) * sideScale * sideFlip + rootZ;
        const sideY = (sideRoot2D.y - sidePoint.y) * sideScale;
        const jointWeight = syncWeight * clamp(sideVisibility, 0, 1) * 0.72;
        z = lerp(z, sideDepth, jointWeight);
        y = lerp(y, sideY, jointWeight * 0.28);
      }

      const jointConfidence = clamp(
        frontVisibility * 0.68 + (twoView ? sideVisibility * 0.32 : 0.16)
      );
      points.set(id, new THREE.Vector3(x, y, z));
      confidence.set(id, jointConfidence);
    }

    this.anchorToFloor(points);
    const hands3D = this.fuseHands(front.hands, front.pose, points, frontScale);
    const bounds = this.getBounds(points);
    const averageConfidence = mean([...confidence.values()]);
    const quality = clamp(
      averageConfidence * 0.62 +
      frontReadiness.score * 0.2 +
      (twoView ? syncWeight * sideReadiness.score * 0.18 : 0)
    );

    return {
      timestamp: front.t,
      points,
      hands3D,
      confidence,
      hands: front.hands || [],
      objectsFront: front.objects || [],
      objectsSide: side?.objects || [],
      twoView,
      syncDelta,
      quality,
      bounds,
      frontReadiness,
      sideReadiness
    };
  }

  inferSideFlip(side) {
    const center = bodyCenter(side?.pose);
    const nose = side?.pose?.[0];
    if (!center || !nose) return 1;
    return nose.x >= center.x ? -1 : 1;
  }

  fuseHands(hands, pose, bodyPoints, frontScale) {
    if (!hands?.length || !pose?.length) return [];
    const wristCandidates = [15, 16]
      .map((id) => ({ id, image: pose[id], world: bodyPoints.get(id) }))
      .filter((candidate) => candidate.image && candidate.world);

    const usedWrists = new Set();
    const result = [];
    for (const hand of hands) {
      const handWrist = hand?.[0];
      if (!handWrist) continue;
      const nearest = wristCandidates
        .filter((candidate) => !usedWrists.has(candidate.id))
        .map((candidate) => ({
          ...candidate,
          distance: Math.hypot(
            handWrist.x - candidate.image.x,
            handWrist.y - candidate.image.y
          )
        }))
        .sort((a, b) => a.distance - b.distance)[0];
      if (!nearest || nearest.distance > 0.24) continue;

      usedWrists.add(nearest.id);
      const scale = frontScale * 0.68;
      const points = hand.map((landmark, index) => {
        if (index === 0) return nearest.world.clone();
        return new THREE.Vector3(
          nearest.world.x - (landmark.x - handWrist.x) * scale,
          nearest.world.y + (handWrist.y - landmark.y) * scale,
          nearest.world.z - ((landmark.z || 0) - (handWrist.z || 0)) * scale * 0.72
        );
      });
      result.push({ wristId: nearest.id, points });
    }
    return result;
  }

  anchorToFloor(points) {
    const floorCandidates = [27, 28, 29, 30, 31, 32]
      .map((id) => points.get(id)?.y)
      .filter(Number.isFinite);
    if (!floorCandidates.length) return;
    const floor = Math.min(...floorCandidates);
    for (const point of points.values()) point.y -= floor;
  }

  getBounds(points) {
    const box = new THREE.Box3();
    for (const point of points.values()) box.expandByPoint(point);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    return { box, size, center };
  }
}

export class PoseStabilizer {
  constructor() {
    this.filters = new Map();
    this.lastPoints = new Map();
    this.lastSeenAt = new Map();
    this.restLengths = new Map();
    this.footLocks = new Map();
    this.lastTimestamp = 0;
  }

  reset() {
    this.filters.clear();
    this.lastPoints.clear();
    this.lastSeenAt.clear();
    this.restLengths.clear();
    this.footLocks.clear();
    this.lastTimestamp = 0;
  }

  captureRestLengths(points) {
    for (const [a, b] of BODY_BONES) {
      const from = points.get(a);
      const to = points.get(b);
      if (!from || !to) continue;
      const length = from.distanceTo(to);
      if (length > 0.04 && length < 0.85) this.restLengths.set(`${a}:${b}`, length);
    }
  }

  update(frame) {
    if (!frame?.points) return null;
    const timestamp = frame.timestamp || Date.now();
    const stabilized = new Map();

    for (const id of RIG_IDS) {
      const current = frame.points.get(id);
      const jointConfidence = frame.confidence.get(id) || 0;

      if (current && jointConfidence > 0.18) {
        const previous = this.lastPoints.get(id);
        const distance = previous ? previous.distanceTo(current) : 0;
        const beta = distance > 0.22 ? 0.58 : jointConfidence > 0.72 ? 0.32 : 0.2;
        let filter = this.filters.get(id);
        if (!filter) {
          filter = new OneEuroVector(1.55, beta);
          this.filters.set(id, filter);
        }
        const filtered = filter.filter(current, timestamp);
        stabilized.set(id, filtered);
        this.lastPoints.set(id, filtered.clone());
        this.lastSeenAt.set(id, timestamp);
      } else {
        const previous = this.lastPoints.get(id);
        const age = timestamp - (this.lastSeenAt.get(id) || 0);
        if (previous && age < 260) stabilized.set(id, previous.clone());
      }
    }

    if (!this.restLengths.size && frame.quality > 0.58) this.captureRestLengths(stabilized);
    this.applyBoneConstraints(stabilized);
    this.applyFootLock(stabilized, timestamp);
    this.anchorToFloor(stabilized);
    this.lastTimestamp = timestamp;

    return { ...frame, points: stabilized };
  }

  applyBoneConstraints(points) {
    if (!this.restLengths.size) return;

    for (let iteration = 0; iteration < 2; iteration += 1) {
      for (const [a, b] of BODY_BONES) {
        const from = points.get(a);
        const to = points.get(b);
        const restLength = this.restLengths.get(`${a}:${b}`);
        if (!from || !to || !restLength) continue;

        const delta = to.clone().sub(from);
        const distance = delta.length();
        if (distance < 0.0001) continue;

        const errorRatio = Math.abs(distance - restLength) / restLength;
        if (errorRatio < 0.035) continue;

        const correction = delta.multiplyScalar((distance - restLength) / distance);
        const rootBone = a === 23 || a === 24;
        if (rootBone) {
          to.sub(correction.multiplyScalar(0.88));
        } else {
          from.add(correction.clone().multiplyScalar(0.38));
          to.sub(correction.multiplyScalar(0.62));
        }
      }
    }
  }

  applyFootLock(points, timestamp) {
    const pairs = [[27, 31], [28, 32]];
    const deltaTime = this.lastTimestamp
      ? clamp((timestamp - this.lastTimestamp) / 1000, 1 / 120, 0.2)
      : 1 / 30;

    for (const [ankleId, toeId] of pairs) {
      const ankle = points.get(ankleId);
      const toe = points.get(toeId);
      if (!ankle || !toe) continue;

      const previous = this.lastPoints.get(toeId);
      const speed = previous ? previous.distanceTo(toe) / deltaTime : Infinity;
      const grounded = Math.min(ankle.y, toe.y) < 0.07;
      const key = `${ankleId}:${toeId}`;
      let lock = this.footLocks.get(key);

      if (grounded && speed < 0.32) {
        if (!lock) {
          lock = { x: toe.x, z: toe.z, strength: 0 };
          this.footLocks.set(key, lock);
        }
        lock.strength = clamp(lock.strength + 0.16);
        toe.x = lerp(toe.x, lock.x, lock.strength * 0.72);
        toe.z = lerp(toe.z, lock.z, lock.strength * 0.72);
      } else if (lock) {
        lock.strength -= 0.25;
        if (lock.strength <= 0) this.footLocks.delete(key);
      }
    }
  }

  anchorToFloor(points) {
    const values = [27, 28, 29, 30, 31, 32]
      .map((id) => points.get(id)?.y)
      .filter(Number.isFinite);
    if (!values.length) return;
    const floor = Math.min(...values);
    for (const point of points.values()) point.y -= floor;
  }
}

export function createNeutralPose() {
  const points = new Map();
  const set = (id, x, y, z = 0) => points.set(id, new THREE.Vector3(x, y, z));

  set(0, 0, 1.64, -0.02);
  set(7, -0.09, 1.65, 0);
  set(8, 0.09, 1.65, 0);
  set(9, -0.035, 1.58, -0.08);
  set(10, 0.035, 1.58, -0.08);
  set(11, -0.22, 1.43, 0);
  set(12, 0.22, 1.43, 0);
  set(13, -0.34, 1.16, 0.02);
  set(14, 0.34, 1.16, 0.02);
  set(15, -0.29, 0.9, -0.02);
  set(16, 0.29, 0.9, -0.02);
  set(17, -0.32, 0.86, -0.02);
  set(18, 0.32, 0.86, -0.02);
  set(19, -0.29, 0.84, -0.06);
  set(20, 0.29, 0.84, -0.06);
  set(21, -0.27, 0.86, -0.08);
  set(22, 0.27, 0.86, -0.08);
  set(23, -0.145, 0.92, 0);
  set(24, 0.145, 0.92, 0);
  set(25, -0.15, 0.5, 0.015);
  set(26, 0.15, 0.5, 0.015);
  set(27, -0.15, 0.08, 0.01);
  set(28, 0.15, 0.08, 0.01);
  set(29, -0.15, 0.045, -0.035);
  set(30, 0.15, 0.045, -0.035);
  set(31, -0.15, 0.02, -0.2);
  set(32, 0.15, 0.02, -0.2);

  return {
    timestamp: Date.now(),
    points,
    confidence: new Map(RIG_IDS.map((id) => [id, 1])),
    twoView: false,
    quality: 0,
    bounds: null,
    frontReadiness: { detected: false, full: false, score: 0 },
    sideReadiness: { detected: false, full: false, score: 0 },
    objectsFront: [],
    objectsSide: []
  };
}

export function toast(message, duration = 3600) {
  const element = $('#toast');
  if (!element) return;
  element.textContent = message;
  element.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove('show'), duration);
}

export function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}
