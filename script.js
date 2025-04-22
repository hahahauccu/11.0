const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const restartBtn = document.getElementById('restartBtn');
const poseImage = document.getElementById('poseImage');
const finishMessage = document.getElementById('finishMessage');

let detector, rafId;
let currentPoseIndex = 0;
const totalPoses = 7;
let standardKeypointsList = [];
let poseOrder = [];
let successFrames = 0;
let failFrames = 0;
const REQUIRED_FRAMES = 50;
const MAX_FAIL_FRAMES = 10;
let isPlaying = false;

// 避免 pose5 和 pose7 連在一起
function shufflePoseOrder() {
  const poses = [1, 2, 3, 4, 5, 6, 7];
  let valid = false;

  while (!valid) {
    for (let i = poses.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [poses[i], poses[j]] = [poses[j], poses[i]];
    }
    valid = true;
    for (let i = 0; i < poses.length - 1; i++) {
      if ((poses[i] === 5 && poses[i + 1] === 7) || (poses[i] === 7 && poses[i + 1] === 5)) {
        valid = false;
        break;
      }
    }
  }

  poseOrder = poses;
}

function resolvePoseImageName(base) {
  const png = `poses/${base}.png`;
  const PNG = `poses/${base}.PNG`;
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => resolve(png);
    img.onerror = () => resolve(PNG);
    img.src = png;
  });
}

async function loadStandardKeypoints() {
  standardKeypointsList = [];
  for (const i of poseOrder) {
    const res = await fetch(`poses/pose${i}.json`);
    const json = await res.json();
    const keypoints = json.keypoints || json;
    standardKeypointsList.push({
      id: i,
      keypoints,
      imagePath: await resolvePoseImageName(`pose${i}`)
    });
  }
}

function computeAngle(a, b, c) {
  const ab = { x: b.x - a.x, y: b.y - a.y };
  const cb = { x: b.x - c.x, y: b.y - c.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const abLen = Math.hypot(ab.x, ab.y);
  const cbLen = Math.hypot(cb.x, cb.y);
  const angleRad = Math.acos(dot / (abLen * cbLen));
  return angleRad * (180 / Math.PI);
}

function compareKeypointsAngleBased(user, standard) {
  const angles = [
    ["left_shoulder", "left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow", "right_wrist"],
    ["left_hip", "left_knee", "left_ankle"],
    ["right_hip", "right_knee", "right_ankle"],
    ["left_elbow", "left_shoulder", "left_hip"],
    ["right_elbow", "right_shoulder", "right_hip"]
  ];

  let totalDiff = 0, count = 0;
  for (const [a, b, c] of angles) {
    const aU = user.find(kp => kp.name === a);
    const bU = user.find(kp => kp.name === b);
    const cU = user.find(kp => kp.name === c);
    const aS = standard.find(kp => kp.name === a);
    const bS = standard.find(kp => kp.name === b);
    const cS = standard.find(kp => kp.name === c);
    if ([aU, bU, cU, aS, bS, cS].every(kp => kp?.score > 0.5)) {
      const angleUser = computeAngle(aU, bU, cU);
      const angleStd = computeAngle(aS, bS, cS);
      totalDiff += Math.abs(angleUser - angleStd);
      count++;
    }
  }
  if (!count) return 1000;
  return totalDiff / count;
}

function drawKeypoints(kps, color, radius, alpha) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  kps.forEach(kp => {
    if (kp.score > 0.4) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, radius, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
  ctx.globalAlpha = 1.0;
}

async function detect() {
  const result = await detector.estimatePoses(video);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const currentPose = standardKeypointsList[currentPoseIndex];
  if (currentPose) drawKeypoints(currentPose.keypoints, 'blue', 6, 0.5);

  if (result.length > 0) {
    const user = result[0].keypoints;
    drawKeypoints(user, 'red', 6, 1.0);

    const avgDiff = compareKeypointsAngleBased(user, currentPose.keypoints);
    if (avgDiff < 20) {
      successFrames++;
    } else {
      failFrames++;
    }

    if (successFrames >= REQUIRED_FRAMES) {
      currentPoseIndex++;
      successFrames = 0;
      failFrames = 0;
      if (currentPoseIndex < totalPoses) {
        poseImage.src = standardKeypointsList[currentPoseIndex].imagePath;
      } else {
        cancelAnimationFrame(rafId);
        poseImage.style.display = 'none';
        finishMessage.style.display = "block";
        restartBtn.style.display = "block";
      }
    } else if (failFrames > MAX_FAIL_FRAMES) {
      successFrames = 0;
      failFrames = 0;
    }
  }

  rafId = requestAnimationFrame(detect);
}

async function startGame() {
  cancelAnimationFrame(rafId);
  poseImage.style.display = 'block';
  poseImage.src = "";
  finishMessage.style.display = "none";
  restartBtn.style.display = "none";
  standardKeypointsList = [];
  currentPoseIndex = 0;
  successFrames = 0;
  failFrames = 0;
  isPlaying = true;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { exact: 'environment' },
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    alert("⚠️ 無法開啟攝影機：" + err.message);
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.setTransform(-1, 0, 0, 1, canvas.width, 0);

  try {
    await tf.setBackend('webgl'); await tf.ready();
  } catch {
    await tf.setBackend('wasm'); await tf.ready();
  }

  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
  );

  shufflePoseOrder();
  await loadStandardKeypoints();

  poseImage.src = standardKeypointsList[0].imagePath;
  detect();
}

startBtn.addEventListener("click", startGame);
restartBtn.addEventListener("click", startGame);
document.body.addEventListener('click', () => {
  if (!standardKeypointsList.length || !isPlaying) return;
  currentPoseIndex++;
  if (currentPoseIndex < totalPoses) {
    poseImage.src = standardKeypointsList[currentPoseIndex].imagePath;
  } else {
    cancelAnimationFrame(rafId);
    poseImage.style.display = 'none';
    finishMessage.style.display = "block";
    restartBtn.style.display = "block";
    isPlaying = false;
  }
});
