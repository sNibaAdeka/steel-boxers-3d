import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js';
import { clamp, lerp, midpoint3D, FrameRateCounter, HAND_CONNECTIONS } from './core.js';

const UP = new THREE.Vector3(0, 1, 0);

const SEGMENTS = [
  { name: 'upper-arm-left', a: 11, b: 13, top: 0.026, bottom: 0.026 },
  { name: 'lower-arm-left', a: 13, b: 15, top: 0.023, bottom: 0.023 },
  { name: 'upper-arm-right', a: 12, b: 14, top: 0.026, bottom: 0.026 },
  { name: 'lower-arm-right', a: 14, b: 16, top: 0.023, bottom: 0.023 },
  { name: 'upper-leg-left', a: 23, b: 25, top: 0.034, bottom: 0.034 },
  { name: 'lower-leg-left', a: 25, b: 27, top: 0.029, bottom: 0.029 },
  { name: 'upper-leg-right', a: 24, b: 26, top: 0.034, bottom: 0.034 },
  { name: 'lower-leg-right', a: 26, b: 28, top: 0.029, bottom: 0.029 }
];

const JOINT_IDS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32];

function physicalMaterial(color, options = {}) {
  return new THREE.MeshStandardMaterial({
    color,
    emissive: options.emissive ?? 0x000000,
    emissiveIntensity: options.emissiveIntensity ?? 0,
    roughness: options.roughness ?? 0.36,
    metalness: options.metalness ?? 0.02,
    transparent: Boolean(options.transparent),
    opacity: options.opacity ?? 1
  });
}

function setShadow(mesh) {
  return mesh;
}

function midpointFromIds(points, a, b) {
  const first = points.get(a);
  const second = points.get(b);
  if (!first || !second) return null;
  return midpoint3D(first, second);
}

function makeBasis(points) {
  const leftShoulder = points.get(11);
  const rightShoulder = points.get(12);
  const leftHip = points.get(23);
  const rightHip = points.get(24);
  if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return null;

  const shoulderCenter = midpoint3D(leftShoulder, rightShoulder);
  const hipCenter = midpoint3D(leftHip, rightHip);
  const right = rightShoulder.clone().sub(leftShoulder).normalize();
  let up = shoulderCenter.clone().sub(hipCenter).normalize();
  let forward = right.clone().cross(up).normalize();
  if (forward.lengthSq() < 0.001) forward.set(0, 0, 1);
  up = forward.clone().cross(right).normalize();

  const matrix = new THREE.Matrix4().makeBasis(right, up, forward);
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(matrix);
  return { right, up, forward, quaternion, shoulderCenter, hipCenter };
}

class ObjectWorld {
  constructor(scene, materials) {
    this.scene = scene;
    this.materials = materials;
    this.tracks = new Map();
    this.lastTime = performance.now();
  }

  geometryFor(label) {
    switch (label) {
      case 'sports ball':
      case 'ball':
        return new THREE.SphereGeometry(0.12, 20, 14);
      case 'bottle':
        return new THREE.CylinderGeometry(0.055, 0.072, 0.28, 16);
      case 'cup':
        return new THREE.CylinderGeometry(0.07, 0.06, 0.16, 16);
      case 'book':
        return new THREE.BoxGeometry(0.26, 0.045, 0.19);
      case 'cell phone':
      case 'phone':
        return new THREE.BoxGeometry(0.08, 0.015, 0.16);
      case 'backpack':
        return new THREE.BoxGeometry(0.3, 0.4, 0.16);
      case 'chair':
        return new THREE.BoxGeometry(0.48, 0.48, 0.48);
      default:
        return new THREE.BoxGeometry(0.16, 0.16, 0.16);
    }
  }

  createTrack(id, label, position) {
    const mesh = setShadow(new THREE.Mesh(this.geometryFor(label), this.materials.object));
    mesh.position.copy(position);
    this.scene.add(mesh);

    const track = {
      id,
      label,
      mesh,
      position: position.clone(),
      velocity: new THREE.Vector3(),
      state: 'moving',
      seenAt: performance.now(),
      heldBy: null
    };
    this.tracks.set(id, track);
    return track;
  }

  detectionCenter(detection) {
    const box = detection?.box;
    if (!box) return null;
    return { x: box[0] + box[2] / 2, y: box[1] + box[3] / 2, bottom: box[1] + box[3] };
  }

  update(frame, calibration) {
    const now = performance.now();
    const deltaTime = clamp((now - this.lastTime) / 1000, 1 / 120, 0.1);
    this.lastTime = now;

    if (calibration && frame?.objectsFront?.length) {
      for (const frontObject of frame.objectsFront) {
        const frontCenter = this.detectionCenter(frontObject);
        if (!frontCenter) continue;
        const sideObject = frame.objectsSide?.find(
          (candidate) => candidate.label === frontObject.label && candidate.score > 0.38
        );
        const sideCenter = this.detectionCenter(sideObject);

        const x = -(frontCenter.x - calibration.frontCenter.x) * calibration.frontScale;
        const y = Math.max(0.1, (calibration.frontCenter.y - frontCenter.bottom) * calibration.frontScale + 0.92);
        const z = sideCenter
          ? (sideCenter.x - calibration.sideCenter.x) * calibration.sideScale * calibration.sideFlip
          : 0;
        const measured = new THREE.Vector3(x, y, z);
        const id = `${frontObject.label}:0`;
        let track = this.tracks.get(id);
        if (!track) track = this.createTrack(id, frontObject.label, measured);

        const previous = track.position.clone();
        track.position.lerp(measured, 0.34);
        track.velocity.lerp(track.position.clone().sub(previous).divideScalar(deltaTime), 0.28);
        track.seenAt = now;
        if (track.state !== 'held') track.state = 'moving';
      }
    }

    const wrists = [
      { id: 15, position: frame?.points?.get(15) },
      { id: 16, position: frame?.points?.get(16) }
    ].filter((item) => item.position);

    for (const track of this.tracks.values()) {
      const nearestWrist = wrists
        .map((wrist) => ({ ...wrist, distance: wrist.position.distanceTo(track.position) }))
        .sort((a, b) => a.distance - b.distance)[0];

      if (nearestWrist && nearestWrist.distance < 0.28 && track.velocity.length() > 0.06) {
        track.state = 'held';
        track.heldBy = nearestWrist.id;
      }

      if (track.state === 'held') {
        const wrist = frame?.points?.get(track.heldBy);
        if (wrist && wrist.distanceTo(track.position) < 0.48) {
          track.position.lerp(wrist, 0.66);
          track.velocity.multiplyScalar(0.7);
        } else {
          track.state = 'released';
          track.heldBy = null;
        }
      } else if (track.state === 'released' || now - track.seenAt > 900) {
        track.state = 'released';
        track.velocity.y -= 9.8 * deltaTime;
        track.position.addScaledVector(track.velocity, deltaTime);
        if (track.position.y <= 0.12) {
          track.position.y = 0.12;
          track.velocity.set(0, 0, 0);
          track.state = 'resting';
        }
      }

      track.mesh.position.lerp(track.position, 0.5);
      if (track.state === 'held') track.mesh.rotation.y += deltaTime * 1.5;
      track.mesh.visible = now - track.seenAt < 7000 || track.state === 'held' || track.state === 'resting';
    }
  }

  serialize() {
    return [...this.tracks.values()].map((track) => ({
      id: track.id,
      label: track.label,
      state: track.state,
      position: track.position.toArray()
    }));
  }
}

export class AvatarScene {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(33, 1, 0.05, 60);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    this.renderer.shadowMap.enabled = false;
    this.container.prepend(this.renderer.domElement);

    this.materials = {
      stick: physicalMaterial(0x65747a, { emissive: 0x101b20, emissiveIntensity: 0.25, roughness: 0.42, metalness: 0.88 }),
      joint: physicalMaterial(0x18252c, { emissive: 0x102f36, emissiveIntensity: 0.42, roughness: 0.3, metalness: 0.9 }),
      head: physicalMaterial(0x8c989b, { emissive: 0x132830, emissiveIntensity: 0.3, roughness: 0.4, metalness: 0.9 }),
      armor: physicalMaterial(0x4f5c62, { roughness: 0.34, metalness: 0.9 }),
      visor: physicalMaterial(0x45e6dd, { emissive: 0x21b8b2, emissiveIntensity: 1.4, roughness: 0.18, metalness: 0.35 }),
      object: physicalMaterial(0xffbd66, { roughness: 0.38 })
    };

    this.group = new THREE.Group();
    this.group.name = 'human-stick-skeleton';
    this.scene.add(this.group);

    this.bones = new Map();
    this.joints = new Map();
    this.handRigs = new Map();
    this.liveFrame = null;
    this.lastFrameAt = 0;
    this.targetCamera = new THREE.Vector3(0, 0.92, 3.65);
    this.targetLookAt = new THREE.Vector3(0, 0.9, 0);
    this.currentLookAt = this.targetLookAt.clone();
    this.camera.position.copy(this.targetCamera);
    this.camera.lookAt(this.currentLookAt);
    this.renderCounter = new FrameRateCounter();
    this.renderFps = 0;
    this.idleTime = 0;

    this.buildEnvironment();
    this.buildAvatar();
    this.objectWorld = new ObjectWorld(this.scene, this.materials);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
    this.renderer.setAnimationLoop((time) => this.render(time));
  }

  buildEnvironment() {
    const hemisphere = new THREE.HemisphereLight(0xd6f8ff, 0x071014, 2.15);
    this.scene.add(hemisphere);

    const key = new THREE.DirectionalLight(0xc7f5ff, 3.3);
    key.position.set(3.2, 6.5, 4.3);
    this.scene.add(key);

    const rim = new THREE.DirectionalLight(0x43e6a0, 2.2);
    rim.position.set(-4, 3.2, -3.5);
    this.scene.add(rim);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(3.2, 72),
      new THREE.MeshStandardMaterial({ color: 0x0b2028, roughness: 0.9, metalness: 0.02 })
    );
    floor.rotation.x = -Math.PI / 2;
    this.scene.add(floor);

    const ring = new THREE.Mesh(
      new THREE.RingGeometry(1.45, 1.47, 96),
      new THREE.MeshBasicMaterial({ color: 0x2d7181, transparent: true, opacity: 0.55, side: THREE.DoubleSide })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.006;
    this.scene.add(ring);

    const grid = new THREE.GridHelper(6.2, 24, 0x235768, 0x16323d);
    grid.position.y = 0.004;
    grid.material.transparent = true;
    grid.material.opacity = 0.38;
    this.scene.add(grid);

  }

  buildAvatar() {
    const cylinder = (top, bottom, material, radialSegments = 8) => {
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(top, bottom, 1, radialSegments, 1, false),
        material
      );
      return setShadow(mesh);
    };

    for (const segment of SEGMENTS) {
      const mesh = cylinder(segment.top * 1.85, segment.bottom * 1.85, this.materials.stick, 10);
      mesh.name = segment.name;
      this.group.add(mesh);
      this.bones.set(segment.name, mesh);
    }

    for (const id of JOINT_IDS) {
      const radius = [23, 24].includes(id) ? 0.078 : 0.062;
      const mesh = setShadow(new THREE.Mesh(new THREE.SphereGeometry(radius, 10, 8), this.materials.joint));
      mesh.name = `joint-${id}`;
      this.group.add(mesh);
      this.joints.set(id, mesh);
    }

    this.shoulderBar = cylinder(0.06, 0.06, this.materials.armor, 10);
    this.hipBar = cylinder(0.07, 0.07, this.materials.armor, 10);
    this.spine = cylinder(0.075, 0.075, this.materials.joint, 10);
    this.neck = cylinder(0.055, 0.055, this.materials.joint, 10);
    this.neck.name = 'neck';
    this.head = setShadow(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 0.78), this.materials.head));
    this.head.name = 'head';
    this.visor = setShadow(new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.16, 0.06), this.materials.visor));
    this.leftFoot = cylinder(0.06, 0.06, this.materials.armor, 10);
    this.rightFoot = cylinder(0.06, 0.06, this.materials.armor, 10);
    this.torsoArmor = setShadow(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 0.56), this.materials.armor));
    this.pelvisArmor = setShadow(new THREE.Mesh(new THREE.BoxGeometry(1, 0.42, 0.52), this.materials.armor));

    this.group.add(
      this.shoulderBar,
      this.hipBar,
      this.spine,
      this.neck,
      this.head,
      this.visor,
      this.torsoArmor,
      this.pelvisArmor,
      this.leftFoot,
      this.rightFoot
    );

    this.createHandRig(15);
    this.createHandRig(16);
  }

  createHandRig(wristId) {
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(HAND_CONNECTIONS.length * 2 * 3), 3)
    );
    const lines = new THREE.LineSegments(
      lineGeometry,
      new THREE.LineBasicMaterial({ color: 0x8cf7df, transparent: true, opacity: 0.92 })
    );

    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(21 * 3), 3)
    );
    const points = new THREE.Points(
      pointGeometry,
      new THREE.PointsMaterial({ color: 0xeaffff, size: 0.025, sizeAttenuation: true })
    );
    lines.visible = false;
    points.visible = false;
    this.group.add(lines, points);
    this.handRigs.set(wristId, { lines, points });
  }

  setFrame(frame, calibration = null) {
    if (!frame?.points?.size) return;
    this.liveFrame = frame;
    this.lastFrameAt = performance.now();
    this.updateAvatar(frame);
    this.objectWorld.update(frame, calibration);
    this.updateCameraTarget(frame.points);
  }

  updateAvatar(frame) {
    const points = frame.points;
    const basis = makeBasis(points);
    if (!basis) return;

    for (const segment of SEGMENTS) {
      this.alignMesh(this.bones.get(segment.name), points.get(segment.a), points.get(segment.b));
    }

    for (const [id, mesh] of this.joints) {
      const point = points.get(id);
      mesh.visible = Boolean(point);
      if (point) mesh.position.copy(point);
    }

    this.alignMesh(this.shoulderBar, points.get(11), points.get(12));
    this.alignMesh(this.hipBar, points.get(23), points.get(24));
    this.alignMesh(this.spine, basis.hipCenter, basis.shoulderCenter);

    const shoulderWidth = points.get(11).distanceTo(points.get(12));
    const torsoHeight = basis.shoulderCenter.distanceTo(basis.hipCenter);
    this.torsoArmor.visible = true;
    this.torsoArmor.position.copy(basis.hipCenter).lerp(basis.shoulderCenter, 0.56);
    this.torsoArmor.quaternion.copy(basis.quaternion);
    this.torsoArmor.scale.set(shoulderWidth * 1.12, torsoHeight * 0.88, shoulderWidth * 0.56);
    this.pelvisArmor.visible = true;
    this.pelvisArmor.position.copy(basis.hipCenter);
    this.pelvisArmor.quaternion.copy(basis.quaternion);
    this.pelvisArmor.scale.set(shoulderWidth * 0.78, torsoHeight * 0.25, shoulderWidth * 0.5);
    const nose = points.get(0);
    if (nose) {
      const headRadius = clamp(shoulderWidth * 0.25, 0.095, 0.145);
      this.head.visible = true;
      this.head.position.copy(nose);
      this.head.quaternion.copy(basis.quaternion);
      this.head.scale.set(headRadius * 1.25, headRadius * 1.42, headRadius * 1.05);
      this.visor.visible = true;
      this.visor.position.copy(nose).addScaledVector(basis.forward, headRadius * 0.46);
      this.visor.quaternion.copy(basis.quaternion);
      this.visor.scale.set(headRadius * 1.06, headRadius * 0.46, headRadius * 0.55);

      const neckTop = nose.clone().addScaledVector(basis.up, -headRadius * 0.82);
      const neckBottom = basis.shoulderCenter.clone().addScaledVector(basis.up, 0.035);
      this.alignMesh(this.neck, neckBottom, neckTop);
    } else {
      this.head.visible = false;
      this.visor.visible = false;
      this.neck.visible = false;
    }

    this.alignMesh(this.leftFoot, points.get(27), points.get(31));
    this.alignMesh(this.rightFoot, points.get(28), points.get(32));
    this.updateFingerRigs(frame.hands3D || []);
  }

  updateFingerRigs(hands) {
    for (const rig of this.handRigs.values()) {
      rig.lines.visible = false;
      rig.points.visible = false;
    }

    for (const hand of hands) {
      const rig = this.handRigs.get(hand.wristId);
      if (!rig || hand.points?.length !== 21) continue;

      const linePositions = rig.lines.geometry.attributes.position.array;
      for (let index = 0; index < HAND_CONNECTIONS.length; index += 1) {
        const [startId, endId] = HAND_CONNECTIONS[index];
        const start = hand.points[startId];
        const end = hand.points[endId];
        const offset = index * 6;
        linePositions[offset] = start.x;
        linePositions[offset + 1] = start.y;
        linePositions[offset + 2] = start.z;
        linePositions[offset + 3] = end.x;
        linePositions[offset + 4] = end.y;
        linePositions[offset + 5] = end.z;
      }
      rig.lines.geometry.attributes.position.needsUpdate = true;

      const pointPositions = rig.points.geometry.attributes.position.array;
      for (let index = 0; index < hand.points.length; index += 1) {
        const point = hand.points[index];
        const offset = index * 3;
        pointPositions[offset] = point.x;
        pointPositions[offset + 1] = point.y;
        pointPositions[offset + 2] = point.z;
      }
      rig.points.geometry.attributes.position.needsUpdate = true;
      rig.lines.visible = true;
      rig.points.visible = true;
    }
  }

  alignMesh(mesh, start, end) {
    if (!mesh || !start || !end) {
      if (mesh) mesh.visible = false;
      return;
    }
    const direction = end.clone().sub(start);
    const length = direction.length();
    if (length < 0.008) {
      mesh.visible = false;
      return;
    }
    mesh.visible = true;
    mesh.position.copy(start).add(end).multiplyScalar(0.5);
    mesh.scale.set(1, length, 1);
    mesh.quaternion.setFromUnitVectors(UP, direction.normalize());
  }

  updateCameraTarget(points) {
    const box = new THREE.Box3();
    for (const point of points.values()) box.expandByPoint(point);
    if (box.isEmpty()) return;

    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    const fullHeight = Math.max(size.y + 0.24, 1.84);
    const fullWidth = Math.max(size.x + 0.34, 0.8);
    const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
    const distanceForHeight = (fullHeight * 0.5) / Math.tan(verticalFov * 0.5);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov * 0.5) * this.camera.aspect);
    const distanceForWidth = (fullWidth * 0.5) / Math.tan(horizontalFov * 0.5);
    const distance = Math.max(distanceForHeight, distanceForWidth) * 1.22;

    this.targetLookAt.set(center.x, Math.max(0.86, center.y), center.z);
    this.targetCamera.set(center.x, this.targetLookAt.y + 0.03, center.z + clamp(distance, 3.25, 5.2));
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  render(time) {
    if (this.previousRenderTime && time - this.previousRenderTime < 32) return;
    const deltaSeconds = this.previousRenderTime
      ? clamp((time - this.previousRenderTime) / 1000, 0, 0.1)
      : 1 / 60;
    this.previousRenderTime = time;
    this.idleTime += deltaSeconds;

    const dataAge = performance.now() - this.lastFrameAt;
    if (dataAge > 800) {
      const breath = Math.sin(this.idleTime * 1.35) * 0.004;
      this.group.position.y = breath;
      this.group.rotation.y = Math.sin(this.idleTime * 0.23) * 0.025;
    } else {
      this.group.position.y = lerp(this.group.position.y, 0, 0.12);
      this.group.rotation.y = lerp(this.group.rotation.y, 0, 0.12);
    }

    this.camera.position.lerp(this.targetCamera, 0.075);
    this.currentLookAt.lerp(this.targetLookAt, 0.085);
    this.camera.lookAt(this.currentLookAt);
    this.renderer.render(this.scene, this.camera);
    this.renderFps = this.renderCounter.tick(time);
  }

  getObjects() {
    return this.objectWorld.serialize();
  }
}
