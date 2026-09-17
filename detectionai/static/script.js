(() => {
  const video = document.querySelector("#camera");
  const cameraFrame = document.querySelector(".camera-frame");
  const startButton = document.querySelector("#startCamera");
  const switchCameraButton = document.querySelector("#switchCamera");
  const scanButton = document.querySelector("#scanFace");
  const uploadButton = document.querySelector("#uploadImage");
  const uploadImageInput = document.querySelector("#uploadImageInput");
  const recordVideoButton = document.querySelector("#recordVideo");
  const recordingStatus = document.querySelector("#recordingStatus");
  const placeholder = document.querySelector("#cameraPlaceholder");
  const overlay = document.querySelector("#scanOverlay");
  const statusMessage = document.querySelector("#statusMessage");
  const resultCards = document.querySelector("#resultCards");
  const lastScanCard = document.querySelector("#lastScanCard");
  const lastScannedImage = document.querySelector("#lastScannedImage");
  const lastVideoCard = document.querySelector("#lastVideoCard");
  const lastCapturedVideo = document.querySelector("#lastCapturedVideo");
  const poseEmojiRail = document.querySelector("#poseEmojiRail");
  const poseDisplayLarge = document.querySelector("#poseDisplayLarge");
  const poseStatusText = document.querySelector("#poseStatusText");

  const MODEL_URL = "/static/models";
  const EXPRESSION_MODEL_URL = "https://justadudewhohacks.github.io/face-api.js/models";
  const POSE_EMOJI_MAP = {
    Happy: "😀",
    Smile: "😃",
    Zoom_Out: "😎",
    Laugh: "😂",
    Love: "😍",
    Thinking: "🤔",
    Angry: "😡",
    Cry: "😭",
    Video_Record: "📹",
    zoom_In: "😍",
    default: "🤔"
  };
  const FACE_EXPRESSION_MAP = {
    happy: "Smile",
    neutral: "Happy",
    sad: "Cry",
    angry: "Angry",
    fearful: "Thinking",
    disgusted: "Love",
    surprised: "Laugh"
  };
  const profiles = Array.isArray(window.ENROLLED_PROFILES) ? window.ENROLLED_PROFILES : [];
  const objectManifest = Array.isArray(window.OBJECT_MANIFEST) ? window.OBJECT_MANIFEST : [];
  const threshold = Number(window.FACE_MATCH_THRESHOLD || 0.58);
  // Object dictionary matching is a secondary fallback, kept strict so background
  // furniture, laptops, or walls never override live face recognition.
  const OBJECT_MATCH_THRESHOLD = 0.68;
  const objectReferenceCache = new Map();
  let stream = null;
  let cameraFacingMode = "user";
  let matcher = null;
  let isScanning = false;
  let scanningPaused = false;
  let scanTimer = null;
  let mediaRecorder = null;
  let recordingChunks = [];
  let recordingUrl = null;
  let recordingStartedAt = 0;
  const recordingMatchesByPerson = new Map();
  // 416 inputSize is fast and lightweight on mobile and back camera, with 0.15 score threshold
  const DETECTION_OPTIONS = new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.15 });
  const CAMERA_SCAN_INTERVAL = 1400;
  const RECORDING_SCAN_INTERVAL = 700;
  const SCAN_SETTLE_TIME = 1500;
  const MODEL_LOAD_TIMEOUT = 25000;
  let settleTimer = null;
  let settleStartedAt = 0;
  let poseModel = null;
  let posePredictionTimer = null;
  let lastExpressionLabel = null;
  let expressionStability = 0;
  // Real finger-count hand gesture control (MediaPipe Hands): 1 finger = zoom in,
  // 2 fingers = start video recording, 3 fingers = zoom out.
  let handsModel = null;
  let handGesturePredictionTimer = null;
  let handGestureBusy = false;
  let lastFingerCount = null;
  let fingerCountStability = 0;
  const FINGER_TIP_IDS = [8, 12, 16, 20];
  const FINGER_PIP_IDS = [6, 10, 14, 18];
  const FINGER_GESTURE_DISPLAY = {
    none: { emoji: "🖐️", text: "Show 1, 2, or 3 fingers to control zoom & recording." },
    0: { emoji: "✊", text: "Fist detected · no action" },
    1: { emoji: "☝️", text: "1 finger detected · Zoom In" },
    2: { emoji: "✌️", text: "2 fingers detected · Video Record" },
    3: { emoji: "🤟", text: "3 fingers detected · Zoom Out" }
  };
  const FINGER_ACTION_LABEL = { 1: "zoom_In", 2: "Video_Record", 3: "Zoom_Out" };

  function setStatus(message, kind = "") {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${kind}`;
  }

  async function scheduleStableScan() {
    if (!stream || isScanning || isRecording()) return;
    if (settleTimer) window.clearTimeout(settleTimer);
    settleStartedAt = Date.now();
    scanButton.disabled = true;
    scanButton.textContent = "Hold steady… 4s";

    const updateCountdown = () => {
      const remaining = Math.max(0, SCAN_SETTLE_TIME - (Date.now() - settleStartedAt));
      const seconds = Math.ceil(remaining / 1000);
      if (remaining > 0) {
        setStatus(`Hold the camera steady… capturing in ${seconds} second${seconds === 1 ? "" : "s"}.`);
        scanButton.textContent = `Hold steady… ${seconds}s`;
        settleTimer = window.setTimeout(updateCountdown, 250);
        return;
      }
      settleTimer = null;
      scanButton.textContent = "Scanning…";
      scanFrame();
    };

    updateCountdown();

    if (cameraFrame) {
      cameraFrame.dataset.videoRatio = `${video.videoWidth}:${video.videoHeight}`;
    }
  }

  function buildImageSignature(image, sourceX = 0, sourceY = 0, sourceWidth = null, sourceHeight = null) {
    const canvas = document.createElement("canvas");
    canvas.width = 48;
    canvas.height = 48;
    const context = canvas.getContext("2d");
    const width = sourceWidth || image.videoWidth || image.naturalWidth || image.width;
    const height = sourceHeight || image.videoHeight || image.naturalHeight || image.height;
    context.drawImage(image, sourceX, sourceY, width, height, 0, 0, canvas.width, canvas.height);
    const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
    const grayscale = [];
    const edges = [];
    const brightnessBins = new Array(8).fill(0);
    const colourBins = new Array(12).fill(0);
    for (let index = 0; index < data.length; index += 4) {
      const brightness = (data[index] + data[index + 1] + data[index + 2]) / 3;
      grayscale.push(brightness / 255);
      brightnessBins[Math.min(7, Math.floor(brightness / 32))] += 1;
      const max = Math.max(data[index], data[index + 1], data[index + 2]);
      const min = Math.min(data[index], data[index + 1], data[index + 2]);
      const colourBin = Math.min(11, Math.floor(((data[index] + data[index + 1] + data[index + 2]) / 3) / 64));
      colourBins[colourBin] += max - min + 1;
    }
    for (let index = 0; index < grayscale.length; index += 1) {
      const right = index % 48 === 47 ? grayscale[index] : grayscale[index + 1];
      const below = index >= grayscale.length - 48 ? grayscale[index] : grayscale[index + 48];
      edges.push(Math.min(1, Math.abs(grayscale[index] - right) + Math.abs(grayscale[index] - below)));
    }
    const normalize = (values) => {
      const magnitude = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)) || 1;
      return values.map((value) => value / magnitude);
    };
    return {
      pixels: normalize(grayscale),
      edges: normalize(edges),
      brightness: normalize(brightnessBins),
      colour: normalize(colourBins)
    };
  }

  function compareImageSignatures(liveSignature, targetSignature) {
    const cosine = (left, right) => {
      let result = 0;
      for (let index = 0; index < left.length; index += 1) result += left[index] * right[index];
      return Math.max(0, Math.min(1, result));
    };
    const pixelSimilarity = 1 - liveSignature.pixels.reduce(
      (sum, value, index) => sum + Math.abs(value - targetSignature.pixels[index]),
      0
    ) / liveSignature.pixels.length;
    const edgeSimilarity = 1 - liveSignature.edges.reduce(
      (sum, value, index) => sum + Math.abs(value - targetSignature.edges[index]),
      0
    ) / liveSignature.edges.length;
    return (
      Math.max(0, pixelSimilarity) * 0.35
      + Math.max(0, edgeSimilarity) * 0.4
      + cosine(liveSignature.brightness, targetSignature.brightness) * 0.1
      + cosine(liveSignature.colour, targetSignature.colour) * 0.15
    );
  }

  function cameraObjectCrops(width, height) {
    const crops = [];
    const squareSize = Math.min(width, height);
    for (const scale of [1, 0.84, 0.68]) {
      const cropSize = squareSize * scale;
      crops.push({
        x: Math.max(0, (width - cropSize) / 2),
        y: Math.max(0, (height - cropSize) / 2),
        width: cropSize,
        height: cropSize
      });
    }
    return crops;
  }

  async function matchObjectFromCameraFrame() {
    if (!objectManifest.length || !video || !video.videoWidth || !video.videoHeight) {
      return null;
    }
    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const captureContext = captureCanvas.getContext("2d");

    const rankedMatches = [];
    for (const item of objectManifest) {
      const referenceUrls = Array.isArray(item.reference_images) ? item.reference_images : [];
      for (const referenceUrl of referenceUrls) {
        try {
          let referenceSignature = objectReferenceCache.get(referenceUrl);
          if (!referenceSignature) {
            const image = await new Promise((resolve, reject) => {
              const objectImage = new Image();
              objectImage.crossOrigin = "anonymous";
              objectImage.onload = () => resolve(objectImage);
              objectImage.onerror = () => reject(new Error(`Could not load ${referenceUrl}`));
              objectImage.src = referenceUrl;
            });
            const referenceWidth = image.naturalWidth || image.width;
            const referenceHeight = image.naturalHeight || image.height;
            const referenceSize = Math.min(referenceWidth, referenceHeight) * 0.9;
            referenceSignature = buildImageSignature(
              image,
              (referenceWidth - referenceSize) / 2,
              (referenceHeight - referenceSize) / 2,
              referenceSize,
              referenceSize
            );
            objectReferenceCache.set(referenceUrl, referenceSignature);
          }
          let score = 0;
          for (const crop of cameraObjectCrops(captureCanvas.width, captureCanvas.height)) {
            captureContext.clearRect(0, 0, captureCanvas.width, captureCanvas.height);
            const liveSignature = buildImageSignature(
              video,
              crop.x,
              crop.y,
              crop.width,
              crop.height
            );
            score = Math.max(score, compareImageSignatures(liveSignature, referenceSignature));
          }
          rankedMatches.push({ score, item, referenceUrl });
        } catch (error) {
          console.warn("Object image match failed:", error);
        }
      }
    }

    rankedMatches.sort((left, right) => right.score - left.score);
    const bestMatch = rankedMatches[0];
    if (!bestMatch || bestMatch.score < OBJECT_MATCH_THRESHOLD) {
      return null;
    }

    return bestMatch;
  }

  function createObjectResultCard(match, objectEntry, sourceNote = "Object recognized from camera scan") {
    const card = document.createElement("article");
    card.className = "result-card";

    const icon = document.createElement("div");
    icon.className = "result-icon object-result-icon";
    icon.textContent = "✓";

    const content = document.createElement("div");
    const entryInfo = objectEntry.entry || {};
    const displayName = entryInfo.word || objectEntry.name || "Recognized object";
    const label = document.createElement("p");
    label.className = "result-label";
    label.textContent = "Recognized object";

    const name = document.createElement("h2");
    name.textContent = displayName;

    const confidence = document.createElement("p");
    confidence.className = "confidence";
    confidence.textContent = `Match confidence: ${(match.score * 100).toFixed(1)}%`;

    const source = document.createElement("p");
    source.className = "source-note";
    source.textContent = sourceNote;

    const description = document.createElement("p");
    description.className = "confidence";
    description.textContent = objectEntry.summary || "Object scanned from the live camera feed.";

    const details = document.createElement("dl");
    details.className = "details-list";
    if (entryInfo.word || objectEntry.name) {
      addDetailLine(details, "Keyword", displayName);
    }
    if (entryInfo.description || objectEntry.summary) {
      addDetailLine(details, "Description", entryInfo.description || objectEntry.summary || "No description available.");
    }
    if (entryInfo.primary_use) {
      addDetailLine(details, "Why use it", entryInfo.primary_use || "Not specified.");
    }
    if (entryInfo.category) addDetailLine(details, "Category", entryInfo.category);
    if (entryInfo.common_locations) addDetailLine(details, "Common locations", entryInfo.common_locations);

    const captureFrame = document.createElement("img");
    captureFrame.className = "profile-snapshot";
    captureFrame.src = match.frameUrl || "";
    captureFrame.alt = `${displayName} captured from live camera`;

    const referenceImage = document.createElement("img");
    referenceImage.className = "profile-snapshot";
    referenceImage.src = match.referenceUrl || (Array.isArray(entryInfo.reference_images) ? entryInfo.reference_images[0] : "");
    referenceImage.alt = `${displayName} reference image`;

    content.append(label, name, confidence, source, description, details, captureFrame, referenceImage);
    card.append(icon, content);
    return card;
  }

  function showObjectMatch(objectMatch, sourceNote = "Object recognized from the live camera scan") {
    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCanvas.getContext("2d").drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const frameUrl = captureCanvas.toDataURL("image/jpeg", 0.9);
    const objectEntry = objectMatch.item;
    resultCards.hidden = false;
    resultCards.replaceChildren(
      createObjectResultCard({ ...objectMatch, frameUrl }, objectEntry, sourceNote)
    );
    stopAutomaticScanning();
    setStatus(`${objectEntry.entry?.word || objectEntry.name || "Object"} matched from the camera feed.`, "success");
  }

  function modelError(error) {
    console.error("Face recognition setup failed", error);
    recordVideoButton.disabled = true;
    // Object matching uses the uploaded catalogue independently of the face
    // model, so keep the camera scanner available when only face models fail.
    scanButton.disabled = !stream;
    setStatus("Face model could not load. Object scanning is still available; check internet access to enable face matching.", "warning");
  }

  function clearResult() {
    resultCards.hidden = true;
    resultCards.replaceChildren();
    lastScanCard.hidden = true;
    lastScannedImage.removeAttribute("src");
  }

  function activatePoseEmoji(label) {
    const resolvedLabel = resolvePoseAction(label);
    const normalizedLabel = String(resolvedLabel || "").trim();
    const displayLabel = normalizedLabel === "happy" || normalizedLabel === "Smile"
      ? "Smile"
      : normalizedLabel === "cry" || normalizedLabel === "sad"
        ? "Cry"
        : normalizedLabel === "angry"
          ? "Angry"
          : normalizedLabel === "thinking" || normalizedLabel === "fearful"
            ? "Thinking"
            : normalizedLabel === "laugh" || normalizedLabel === "surprised"
              ? "Laugh"
              : normalizedLabel;
    const emoji = POSE_EMOJI_MAP[displayLabel] || POSE_EMOJI_MAP[normalizedLabel] || POSE_EMOJI_MAP.default;
    if (poseDisplayLarge) poseDisplayLarge.textContent = emoji;
    if (poseStatusText) poseStatusText.textContent = displayLabel ? `${displayLabel} detected` : "Waiting for a hand posture…";
    if (!poseEmojiRail) return;
    poseEmojiRail.querySelectorAll(".pose-emoji-btn").forEach((button) => {
      const isActive = button.dataset.label === displayLabel || button.dataset.label === normalizedLabel || button.dataset.emoji === emoji;
      button.classList.toggle("active", Boolean(isActive));
    });
  }

  function normalizePoseLabel(label) {
    return String(label || "")
      .trim()
      .replace(/[_\s-]+/g, "")
      .toLowerCase();
  }

  function resolvePoseAction(label) {
    const key = normalizePoseLabel(label);
    if (["zoomin", "zoom_in", "1", "one"].includes(key)) return "zoom_In";
    if (["videorecord", "video_record", "record", "2", "two"].includes(key)) return "Video_Record";
    if (["zoomout", "zoom_out", "3", "three"].includes(key)) return "Zoom_Out";
    return String(label || "").trim();
  }

  function applyCameraZoom(level) {
    const safeLevel = Number.isFinite(level) ? Math.min(Math.max(level, 0.8), 1.35) : 1;
    if (video) {
      const mirror = cameraFacingMode === "user" ? -1 : 1;
      video.style.transition = "transform 0.22s ease";
      video.style.transformOrigin = "center center";
      video.style.transform = `scale(${safeLevel}) scaleX(${mirror})`;
    }
    if (cameraFrame) {
      cameraFrame.style.transition = "transform 0.22s ease";
      cameraFrame.style.transformOrigin = "center center";
      cameraFrame.style.transform = `scale(${safeLevel})`;
    }
  }

  function distanceBetween(pointA, pointB) {
    return Math.hypot(pointB.x - pointA.x, pointB.y - pointA.y);
  }

  function classifyFaceExpressionFromLandmarks(landmarks) {
    const mouth = landmarks.getMouth();
    if (!mouth || mouth.length < 8) {
      return null;
    }
    const leftCorner = mouth.reduce((best, point) => (point.x < best.x ? point : best), mouth[0]);
    const rightCorner = mouth.reduce((best, point) => (point.x > best.x ? point : best), mouth[0]);
    const upperLip = mouth.reduce((best, point) => (point.y < best.y ? point : best), mouth[0]);
    const lowerLip = mouth.reduce((best, point) => (point.y > best.y ? point : best), mouth[0]);
    const mouthWidth = distanceBetween(leftCorner, rightCorner);
    const mouthHeight = distanceBetween(upperLip, lowerLip);
    const smileRatio = mouthWidth / Math.max(10, mouthHeight);
    const mouthCornerLift = Math.min(leftCorner.y, rightCorner.y) - Math.max(upperLip.y, lowerLip.y);

    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();
    const leftEyeHeight = distanceBetween(leftEye[1], leftEye[5]);
    const rightEyeHeight = distanceBetween(rightEye[1], rightEye[5]);
    const eyeOpen = (leftEyeHeight + rightEyeHeight) / 2;

    const leftBrow = landmarks.getLeftEyebrow();
    const rightBrow = landmarks.getRightEyebrow();
    const browSpread = Math.abs(leftBrow[2].x - rightBrow[2].x);

    if (mouthWidth > 28 && smileRatio >= 1.6 && mouthCornerLift > 0) {
      const score = Math.min(100, ((smileRatio - 1.3) / 1.2) * 100);
      return { label: "Smile", score };
    }
    if (mouthWidth > 35 && smileRatio > 1.7 && mouthCornerLift < -2) {
      const score = Math.min(100, ((smileRatio - 1.3) / 1.4) * 100);
      return { label: "Laugh", score };
    }
    if (mouthHeight > 12 && mouthWidth < 32 && eyeOpen < 8 && browSpread < 35) {
      return { label: "Cry", score: 90 };
    }
    if (mouthWidth < 28 && mouthHeight < 10 && eyeOpen < 6 && browSpread < 45) {
      return { label: "Angry", score: 82 };
    }
    return null;
  }

  async function detectFaceExpression() {
    if (!window.faceapi || !video || !video.videoWidth || !video.videoHeight || video.readyState < 2) {
      return null;
    }
    try {
      const detection = await faceapi
        .detectSingleFace(video, DETECTION_OPTIONS)
        .withFaceLandmarks()
        .withFaceExpressions();
      if (detection && detection.expressions) {
        const bestExpression = Object.entries(detection.expressions).sort((left, right) => right[1] - left[1])[0];
        if (bestExpression && bestExpression[1] >= 0.38) {
          const label = bestExpression[0].toLowerCase();
          if (label === "happy") return { label: "Smile", score: bestExpression[1] * 100 };
          if (label === "sad" || label === "neutral") return { label: "Cry", score: bestExpression[1] * 100 };
          if (label === "angry") return { label: "Angry", score: bestExpression[1] * 100 };
          if (label === "surprised") return { label: "Laugh", score: bestExpression[1] * 100 };
        }
      }

      const fallback = await faceapi
        .detectSingleFace(video, DETECTION_OPTIONS)
        .withFaceLandmarks();
      if (!fallback || !fallback.landmarks) {
        return null;
      }
      return classifyFaceExpressionFromLandmarks(fallback.landmarks);
    } catch (error) {
      try {
        const fallback = await faceapi
          .detectSingleFace(video, DETECTION_OPTIONS)
          .withFaceLandmarks();
        if (!fallback || !fallback.landmarks) return null;
        return classifyFaceExpressionFromLandmarks(fallback.landmarks);
      } catch (fallbackError) {
        return null;
      }
    }
  }

  async function loadPoseModel() {
    const tmPoseLib = window.tmPose || window.teachablemachine?.pose;
    if (!window.tf || !tmPoseLib) {
      if (poseStatusText) poseStatusText.textContent = "Pose model script not ready yet.";
      return false;
    }
    try {
      poseModel = await tmPoseLib.load("/my-pose-model/model.json", "/my-pose-model/metadata.json");
      if (poseStatusText) poseStatusText.textContent = "Show 1, 2, or 3 fingers to control zoom & recording.";
      if (poseDisplayLarge) poseDisplayLarge.textContent = "🤔";
      if (posePredictionTimer) window.clearInterval(posePredictionTimer);
      posePredictionTimer = window.setInterval(async () => {
        if (!video || !video.videoWidth || !video.videoHeight || video.readyState < 2) {
          return;
        }
        try {
          const expression = await detectFaceExpression();
          if (expression && expression.score >= 38) {
            if (lastExpressionLabel !== expression.label) {
              expressionStability = 1;
              lastExpressionLabel = expression.label;
            } else {
              expressionStability += 1;
            }
            if (expressionStability >= 2) {
              activatePoseEmoji(expression.label);
              if (poseStatusText) {
                poseStatusText.textContent = `${expression.label} detected · ${expression.score.toFixed(1)}%`;
              }
            }
          } else {
            lastExpressionLabel = null;
            expressionStability = 0;
          }

          // Zoom In / Zoom Out / Video Record are now driven by real finger-count
          // hand gestures (see loadHandGestureModel below), not this pose net.
        } catch (error) {
          if (poseStatusText) poseStatusText.textContent = "Waiting for a stable smile or hand posture…";
        }
      }, 900);
      return true;
    } catch (error) {
      console.error("Pose model failed to load", error);
      if (poseStatusText) poseStatusText.textContent = "Pose model failed to load. Try again after the camera starts.";
      return false;
    }
  }

  // Counts how many of the four fingers (index, middle, ring, pinky) are
  // extended, using MediaPipe Hands' 21-point landmark list. The thumb is
  // intentionally ignored so a natural "1 / 2 / 3 fingers" gesture is easy
  // to hold and reliable regardless of which hand is shown.
  function countExtendedFingers(landmarks) {
    if (!Array.isArray(landmarks) || landmarks.length < 21) return 0;
    let count = 0;
    for (let i = 0; i < FINGER_TIP_IDS.length; i += 1) {
      const tip = landmarks[FINGER_TIP_IDS[i]];
      const pip = landmarks[FINGER_PIP_IDS[i]];
      if (tip && pip && tip.y < pip.y - 0.02) {
        count += 1;
      }
    }
    return count;
  }

  function setFingerGestureDisplay(key) {
    const display = FINGER_GESTURE_DISPLAY[key] || FINGER_GESTURE_DISPLAY.none;
    if (poseDisplayLarge) poseDisplayLarge.textContent = display.emoji;
    if (poseStatusText) poseStatusText.textContent = display.text;
    const actionLabel = FINGER_ACTION_LABEL[key];
    poseEmojiRail?.querySelectorAll(".pose-emoji-btn").forEach((button) => {
      button.classList.toggle("active", Boolean(actionLabel) && button.dataset.label === actionLabel);
    });
  }

  function handleHandGestureResults(results) {
    const landmarks = results?.multiHandLandmarks && results.multiHandLandmarks[0];
    if (!landmarks) {
      lastFingerCount = null;
      fingerCountStability = 0;
      applyCameraZoom(1);
      setFingerGestureDisplay("none");
      return;
    }

    const count = countExtendedFingers(landmarks);
    if (lastFingerCount === count) {
      fingerCountStability += 1;
    } else {
      fingerCountStability = 1;
      lastFingerCount = count;
    }

    setFingerGestureDisplay(count <= 3 ? count : 0);

    // Require a few consecutive matching frames before acting, so a hand
    // passing through mid-gesture doesn't flicker the zoom or fire the
    // recorder by accident.
    if (fingerCountStability < 3) return;

    if (count === 1) {
      applyCameraZoom(1.2);
    } else if (count === 3) {
      applyCameraZoom(0.9);
    } else {
      applyCameraZoom(1);
      if (count === 2 && !isRecording()) {
        startRecording();
      }
    }
  }

  async function loadHandGestureModel() {
    const HandsLib = window.Hands;
    if (!HandsLib) {
      if (poseStatusText) poseStatusText.textContent = "Hand gesture script not ready yet.";
      return false;
    }
    try {
      handsModel = new HandsLib({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
      });
      handsModel.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5
      });
      handsModel.onResults(handleHandGestureResults);

      setFingerGestureDisplay("none");

      if (handGesturePredictionTimer) window.clearInterval(handGesturePredictionTimer);
      handGesturePredictionTimer = window.setInterval(async () => {
        if (!video || !video.videoWidth || !video.videoHeight || video.readyState < 2) return;
        if (handGestureBusy) return;
        handGestureBusy = true;
        try {
          await handsModel.send({ image: video });
        } catch (error) {
          // Ignore transient frame errors (e.g. camera switching mid-send).
        } finally {
          handGestureBusy = false;
        }
      }, 220);
      return true;
    } catch (error) {
      console.error("Hand gesture model failed to load", error);
      if (poseStatusText) poseStatusText.textContent = "Hand gesture model failed to load. Try again after the camera starts.";
      return false;
    }
  }

  function setRecordingStatus(message = "") {
    recordingStatus.textContent = message;
    recordingStatus.hidden = !message;
  }

  function startAutomaticScanning(interval = CAMERA_SCAN_INTERVAL) {
    if (scanTimer) window.clearInterval(scanTimer);
    scanningPaused = false;
    scanTimer = window.setInterval(scanFrame, interval);
  }

  function isRecording() {
    return mediaRecorder?.state === "recording";
  }

  function stopCameraStream() {
    if (isRecording()) mediaRecorder.stop();
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    if (scanTimer) window.clearInterval(scanTimer);
    scanTimer = null;
    if (settleTimer) window.clearTimeout(settleTimer);
    settleTimer = null;
  }

  function clearRecordedVideo() {
    if (recordingUrl) URL.revokeObjectURL(recordingUrl);
    recordingUrl = null;
    lastCapturedVideo.removeAttribute("src");
    lastCapturedVideo.load();
    lastVideoCard.hidden = true;
  }

  function formatVideoTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.floor(seconds % 60).toString().padStart(2, "0");
    return `${minutes}:${remainder}`;
  }

  async function descriptorFromImage(url) {
    try {
      const image = await faceapi.fetchImage(url);
      let result = await faceapi
        .detectSingleFace(image, DETECTION_OPTIONS)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!result) {
        // Fallback with slightly relaxed options for diverse lighting/angles
        const relaxedOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.1 });
        result = await faceapi
          .detectSingleFace(image, relaxedOptions)
          .withFaceLandmarks()
          .withFaceDescriptor();
      }

      if (!result) {
        console.warn(`No face detected in reference image: ${url}`);
        return null;
      }
      return result.descriptor;
    } catch (error) {
      console.warn(`Could not load descriptor for ${url}:`, error);
      return null;
    }
  }

  async function loadRecognition() {
    if (!window.faceapi) throw new Error("The face recognition library did not load.");
    if (!profiles.length) throw new Error("No enrolled profiles are available.");
    setStatus("Loading face recognition model…");
    const modelLoad = Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(EXPRESSION_MODEL_URL)
    ]);
    await Promise.race([
      modelLoad,
      new Promise((_, reject) => window.setTimeout(
        () => reject(new Error("Face model loading timed out.")),
        MODEL_LOAD_TIMEOUT
      ))
    ]);
    setStatus("Encoding enrolled reference photos…");

    const labeledDescriptors = [];
    for (const profile of profiles) {
      const descriptors = [];
      const referenceImages = Array.isArray(profile.reference_images) ? profile.reference_images : [];
      for (const url of referenceImages) {
        const desc = await descriptorFromImage(url);
        if (desc) {
          descriptors.push(desc);
        }
      }
      if (descriptors.length > 0) {
        labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(profile.person_id, descriptors));
        console.log(`Enrolled ${profile.person_id} (${profile.name}) with ${descriptors.length} face descriptors.`);
      } else {
        console.warn(`No clear face descriptors found for profile ${profile.person_id}`);
      }
    }

    if (!labeledDescriptors.length) {
      throw new Error("Could not extract any valid face descriptors from reference photos.");
    }

    matcher = new faceapi.FaceMatcher(labeledDescriptors, threshold);
    recordVideoButton.disabled = !stream || !window.MediaRecorder;
    scanButton.disabled = !stream;
    setStatus(`${labeledDescriptors.length} enrolled profile(s) ready. Start the camera to scan.`, "success");
  }

  function addDetailLine(list, label, value) {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    list.append(term, description);
  }

  function appendProfileDetails(list, details, clothesColour) {
    for (const [key, value] of Object.entries(details || {})) {
      if (key.toLowerCase() === "name") continue;
      const displayValue = typeof value === "string" ? value : JSON.stringify(value);
      addDetailLine(list, key.replace(/[_-]/g, " "), displayValue);
    }
    addDetailLine(list, "clothes colour", clothesColour === "not visible" ? "Show upper body in camera" : clothesColour);
  }

  async function sendProfileResult(profile, distance, clothesColour, button) {
    button.disabled = true;
    button.textContent = "Sending result…";
    try {
      const response = await fetch(`/api/send-result/${encodeURIComponent(profile.person_id)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          match_confidence: Math.max(0, (1 - distance) * 100),
          clothes_colour: clothesColour
        })
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !payload.sent) throw new Error(payload.error || "Email could not be sent.");
      button.classList.add("sent");
      button.textContent = "Result sent ✓";
      setStatus(`${profile.name}'s scan details were sent to ${payload.recipient}.`);
    } catch (error) {
      button.disabled = false;
      button.textContent = "Send result by email";
      setStatus(error.message || "Email could not be sent. Please try again.", "error");
    }
  }

  function rgbToHsv(red, green, blue) {
    const r = red / 255;
    const g = green / 255;
    const b = blue / 255;
    const maximum = Math.max(r, g, b);
    const minimum = Math.min(r, g, b);
    const delta = maximum - minimum;
    let hue = 0;
    if (delta) {
      if (maximum === r) hue = ((g - b) / delta) % 6;
      else if (maximum === g) hue = (b - r) / delta + 2;
      else hue = (r - g) / delta + 4;
      hue = (hue * 60 + 360) % 360;
    }
    return { hue, saturation: maximum ? delta / maximum : 0, value: maximum };
  }

  function colourName(hue, saturation, value) {
    if (value < 0.18) return "black";
    if (saturation < 0.16) return value > 0.84 ? "white" : "grey";
    // Dark warm fabric is perceived as brown before it is perceived as orange.
    if (hue >= 12 && hue < 52 && value < 0.62) return "brown";
    if (hue < 12 || hue >= 348) return "red";
    if (hue < 38) return "orange";
    if (hue < 65) return "yellow";
    if (hue < 165) return "green";
    if (hue < 200) return "teal";
    if (hue < 255) return "blue";
    if (hue < 292) return "purple";
    return "pink";
  }

  function detectClothesColour(context, faceBox, frameWidth, frameHeight) {
    // Sample the upper torso below the detected face, avoiding skin pixels in
    // the face itself. A full upper-body preview gives the most reliable label.
    const left = Math.max(0, Math.floor(faceBox.x - faceBox.width * 0.38));
    const top = Math.max(0, Math.floor(faceBox.y + faceBox.height * 0.92));
    const right = Math.min(frameWidth, Math.ceil(faceBox.x + faceBox.width * 1.38));
    const bottom = Math.min(frameHeight, Math.ceil(faceBox.y + faceBox.height * 2.7));
    if (right - left < 20 || bottom - top < 20) return "not visible";

    const pixels = context.getImageData(left, top, right - left, bottom - top).data;
    const hueBins = Array.from({ length: 36 }, () => ({ weight: 0, hue: 0, saturation: 0, value: 0 }));
    const neutral = { black: 0, white: 0, grey: 0 };
    for (let index = 0; index < pixels.length; index += 16) {
      const { hue, saturation, value } = rgbToHsv(pixels[index], pixels[index + 1], pixels[index + 2]);
      if (value < 0.18) { neutral.black += 1; continue; }
      if (saturation < 0.16) {
        neutral[value > 0.84 ? "white" : "grey"] += 1;
        continue;
      }
      const bin = hueBins[Math.min(35, Math.floor(hue / 10))];
      const weight = saturation * (0.55 + value * 0.45);
      bin.weight += weight;
      bin.hue += hue * weight;
      bin.saturation += saturation * weight;
      bin.value += value * weight;
    }
    const colourful = hueBins.reduce((best, bin) => bin.weight > best.weight ? bin : best, hueBins[0]);
    const strongestNeutral = Object.entries(neutral).reduce((best, entry) => entry[1] > best[1] ? entry : best, ["grey", 0]);
    if (!colourful.weight || strongestNeutral[1] > colourful.weight * 2.2) return strongestNeutral[0];
    return colourName(
      colourful.hue / colourful.weight,
      colourful.saturation / colourful.weight,
      colourful.value / colourful.weight
    );
  }

  async function fetchProfile(personId) {
    const response = await fetch(`/api/details/${encodeURIComponent(personId)}`);
    if (!response.ok) throw new Error("The recognised profile could not be loaded.");
    const payload = await response.json();
    const enrolledProfile = profiles.find((profile) => profile.person_id === personId) || {};
    return {
      person_id: personId,
      name: payload.details?.name || personId,
      details: payload.details || {},
      reference_images: Array.isArray(enrolledProfile.reference_images) ? enrolledProfile.reference_images : []
    };
  }

  async function findClosestProfileForDescriptor(descriptor) {
    let bestMatch = { person_id: null, distance: Number.POSITIVE_INFINITY };
    for (const profile of profiles) {
      const referenceImages = Array.isArray(profile.reference_images) ? profile.reference_images : [];
      for (const referenceUrl of referenceImages) {
        try {
          const referenceDescriptor = await descriptorFromImage(referenceUrl);
          const distance = faceapi.euclideanDistance(referenceDescriptor, descriptor);
          if (distance < bestMatch.distance) {
            bestMatch = {
              person_id: profile.person_id,
              distance
            };
          }
        } catch (error) {
          console.warn(`Could not compare descriptor against ${profile.person_id}`, error);
        }
      }
    }
    return bestMatch;
  }

  async function resolveFaceMatch(descriptor) {
    if (!matcher) {
      return { label: "unknown", distance: Number.POSITIVE_INFINITY, fallbackUsed: false };
    }

    const bestMatch = matcher.findBestMatch(descriptor);
    if (bestMatch.label !== "unknown") {
      return { ...bestMatch, fallbackUsed: false };
    }

    const closestProfile = await findClosestProfileForDescriptor(descriptor);
    if (closestProfile.person_id && closestProfile.distance <= 0.60) {
      return {
        label: closestProfile.person_id,
        distance: closestProfile.distance,
        fallbackUsed: true
      };
    }

    return { ...bestMatch, fallbackUsed: false };
  }

  function captureAnnotatedImage(detections, matches, source = video, showCapture = true) {
    const canvas = document.createElement("canvas");
    canvas.width = source.videoWidth;
    canvas.height = source.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    context.lineWidth = Math.max(3, Math.round(canvas.width / 220));
    context.font = `bold ${Math.max(16, Math.round(canvas.width / 32))}px system-ui, sans-serif`;
    const clothesColours = detections.map((detection) =>
      detectClothesColour(context, detection.detection.box, canvas.width, canvas.height)
    );
    detections.forEach((detection, index) => {
      const box = detection.detection.box;
      const matchedProfile = matches[index];
      const label = matchedProfile ? `${matchedProfile.name} · ${clothesColours[index]}` : "Face detected";
      const colour = matchedProfile ? "#38e7a3" : "#ff7885";
      context.strokeStyle = colour;
      context.strokeRect(box.x, box.y, box.width, box.height);
      const width = Math.min(canvas.width - 8, context.measureText(label).width + 20);
      const x = Math.max(4, Math.min(canvas.width - width - 4, box.x));
      const y = Math.max(4, box.y - 34);
      context.fillStyle = "rgba(5, 9, 20, .88)";
      context.fillRect(x, y, width, 29);
      context.fillStyle = colour;
      context.fillText(label, x + 10, y + 21);
    });
    const imageUrl = canvas.toDataURL("image/jpeg", 0.9);
    if (showCapture) {
      lastScannedImage.src = imageUrl;
      lastScanCard.hidden = false;
    }
    return { clothesColours, imageUrl };
  }

  function stopAutomaticScanning() {
    if (scanTimer) window.clearInterval(scanTimer);
    scanTimer = null;
    scanningPaused = true;
    scanButton.textContent = "Restart Scanning";
  }

  async function matchUploadedImage(file) {
    if (!file || !matcher) {
      setStatus("Recognition is still loading. Please wait a moment and try again.", "warning");
      return;
    }
    if (!file.type.startsWith("image/")) {
      setStatus("Please choose a valid image file such as JPG, PNG, or WEBP.", "error");
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    try {
      const image = await new Promise((resolve, reject) => {
        const uploadedImage = new Image();
        uploadedImage.onload = () => resolve(uploadedImage);
        uploadedImage.onerror = () => reject(new Error("This image could not be loaded."));
        uploadedImage.src = objectUrl;
      });
      const detection = await faceapi
        .detectSingleFace(image, DETECTION_OPTIONS)
        .withFaceLandmarks()
        .withFaceDescriptor();
      if (!detection) {
        setStatus("No clear face was found in the selected image. Try a front-facing portrait photo.", "warning");
        return;
      }
      const bestMatch = await resolveFaceMatch(detection.descriptor);
      if (bestMatch.label === "unknown") {
        showNoMatch("The selected image does not match any enrolled face. Try a different photo or add a reference image for that person.");
        setStatus("Selected image did not match an enrolled face.", "warning");
        return;
      }
      const profile = await fetchProfile(bestMatch.label);
      resultCards.hidden = false;
      resultCards.replaceChildren(
        createProfileCard(
          profile,
          bestMatch.distance,
          "uploaded image",
          0,
          "Matched using the uploaded image from your device",
          objectUrl
        )
      );
      stopAutomaticScanning();
      setStatus(`${profile.name} matched from the uploaded image.`, "success");
    } catch (error) {
      console.error("Uploaded image matching failed", error);
      setStatus("The uploaded image could not be matched. Try a clearer face photo.", "error");
    } finally {
      if (uploadImageInput) uploadImageInput.value = "";
      URL.revokeObjectURL(objectUrl);
    }
  }

  function createProfileCard(profile, distance, clothesColour, index, sourceNote = "", snapshotUrl = "") {
    const card = document.createElement("article");
    card.className = "result-card";
    const icon = document.createElement("div");
    icon.className = "result-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "✓";
    const content = document.createElement("div");
    const label = document.createElement("p");
    label.className = "result-label";
    label.textContent = `Recognized face ${index + 1}`;
    const name = document.createElement("h2");
    name.textContent = profile.name;
    const confidence = document.createElement("p");
    confidence.className = "confidence";
    confidence.textContent = `Match confidence: ${Math.max(0, (1 - distance) * 100).toFixed(1)}%`;
    const source = document.createElement("p");
    source.className = "source-note";
    source.textContent = sourceNote;
    const details = document.createElement("dl");
    details.className = "details-list";
    appendProfileDetails(details, profile.details, clothesColour);
    const summary = document.createElement("section");
    summary.className = "scan-summary";
    summary.setAttribute("aria-label", `${profile.name} quick scan summary`);
    const summaryLabel = document.createElement("p");
    summaryLabel.className = "summary-label";
    summaryLabel.textContent = "Quick summary";
    const summaryList = document.createElement("dl");
    summaryList.className = "details-list summary-list";
    addDetailLine(
      summaryList,
      "Overall",
      `${profile.details?.age || "Age not provided"} · ${profile.details?.profession || profile.details?.face_shape || "Enrolled profile verified"}`
    );
    addDetailLine(
      summaryList,
      "Appearance",
      `${profile.details?.head_hair || "Hair details unavailable"}; ${profile.details?.facial_hair || profile.details?.eyes || "Facial details verified"}`
    );
    addDetailLine(
      summaryList,
      "Live scan",
      `Clothes colour: ${clothesColour === "not visible" ? "show upper body in camera" : clothesColour}`
    );
    summary.append(summaryLabel, summaryList);
    const sendButton = document.createElement("button");
    sendButton.className = "button send-result-button";
    sendButton.type = "button";
    sendButton.textContent = "Send result by email";
    sendButton.setAttribute("aria-label", `Send ${profile.name}'s scan result by email`);
    sendButton.addEventListener("click", () => sendProfileResult(profile, distance, clothesColour, sendButton));
    const referenceImageUrl = Array.isArray(profile.reference_images) ? profile.reference_images[0] : "";
    content.append(label, name, confidence);
    if (sourceNote) content.append(source);
    content.append(details, summary, sendButton);
    if (referenceImageUrl) {
      const referenceImage = document.createElement("img");
      referenceImage.className = "profile-snapshot";
      referenceImage.src = referenceImageUrl;
      referenceImage.alt = `${profile.name} enrolled reference photo`;
      content.append(referenceImage);
    }
    if (snapshotUrl) {
      const snapshot = document.createElement("img");
      snapshot.className = "profile-snapshot";
      snapshot.src = snapshotUrl;
      snapshot.alt = `${profile.name} identified in a captured camera frame`;
      content.append(snapshot);
    }
    card.append(icon, content);
    return card;
  }

  function showNoMatch(message) {
    resultCards.hidden = false;
    const card = document.createElement("article");
    card.className = "result-card not-recognized";
    const icon = document.createElement("div");
    icon.className = "result-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "!";
    const content = document.createElement("div");
    const label = document.createElement("p");
    label.className = "result-label";
    label.textContent = "Scan result";
    const name = document.createElement("h2");
    name.textContent = "No confident match";
    const description = document.createElement("p");
    description.className = "confidence";
    description.textContent = message;
    content.append(label, name, description);
    card.append(icon, content);
    resultCards.replaceChildren(card);
  }

  function renderRecordingMatches() {
    const recognised = [...recordingMatchesByPerson.values()].sort((first, second) => first.time - second.time);
    if (!recognised.length) return;
    resultCards.hidden = false;
    resultCards.replaceChildren(...recognised.map((face, index) =>
      createProfileCard(
        face.profile,
        face.distance,
        face.clothesColour,
        index,
        `Identified while recording at ${formatVideoTime(face.time)}`,
        face.snapshotUrl
      )
    ));
  }

  function retainRecordingMatches(recognizedFaces, snapshotUrl) {
    const elapsed = (Date.now() - recordingStartedAt) / 1000;
    for (const face of recognizedFaces) {
      const existing = recordingMatchesByPerson.get(face.profile.person_id);
      if (!existing || face.distance < existing.distance) {
        recordingMatchesByPerson.set(face.profile.person_id, { ...face, snapshotUrl, time: elapsed });
      }
    }
    renderRecordingMatches();
  }

  async function scanFrame() {
    const recording = isRecording();
    if (scanningPaused && !recording) {
      scanningPaused = false;
      clearResult();
      scanButton.textContent = "Scan Face / Object";
    }
    if (isScanning || !stream) return;
    isScanning = true;
    scanButton.disabled = true;
    overlay.hidden = false;
    setStatus(
      recording
        ? "Recording… scanning camera frames for Alia Bhatt and enrolled profiles…"
        : matcher
          ? "Scanning face in this browser…"
          : "Scanning camera feed…"
    );
    try {
      const detections = matcher
        ? await faceapi
          .detectAllFaces(video, DETECTION_OPTIONS)
          .withFaceLandmarks()
          .withFaceDescriptors()
        : [];
      if (!detections.length) {
        if (!recording) {
          // Only attempt object dictionary match during manual scan if no face was found
          const objectMatch = await matchObjectFromCameraFrame();
          if (objectMatch) {
            showObjectMatch(objectMatch);
            return;
          }
          showNoMatch("No enrolled face was detected in this camera frame. Face the camera directly in good lighting.");
          setStatus("No face detected in current camera frame.", "warning");
        } else {
          setStatus("Recording video… point camera at Alia Bhatt or enrolled person.", "warning");
        }
        return;
      }
      setStatus(`${detections.length} face${detections.length === 1 ? "" : "s"} detected. Matching profile…`);
      const bestMatches = await Promise.all(detections.map(async (detection) => resolveFaceMatch(detection.descriptor)));
      const matchedProfiles = await Promise.all(bestMatches.map((match) =>
        match.label === "unknown" ? null : fetchProfile(match.label)
      ));
      const capturedFrame = captureAnnotatedImage(detections, matchedProfiles);
      const recognizedFaces = matchedProfiles
        .map((profile, index) => profile ? { profile, distance: bestMatches[index].distance, clothesColour: capturedFrame.clothesColours[index] } : null)
        .filter(Boolean);
      if (recognizedFaces.length && recording) {
        retainRecordingMatches(recognizedFaces, capturedFrame.imageUrl);
        const personNames = recognizedFaces.map((f) => f.profile.name).join(", ");
        const bestConf = (Math.max(0, (1 - recognizedFaces[0].distance) * 100)).toFixed(1);
        setRecordingStatus(`● Recording — 🎯 Identified: ${personNames} (${bestConf}%)`);
        setStatus(`🎯 Recording: Identified ${personNames} (${bestConf}% match)! Details displayed below.`, "success");
      } else if (recognizedFaces.length) {
        resultCards.hidden = false;
        const cards = recognizedFaces.map((face, index) =>
          createProfileCard(face.profile, face.distance, face.clothesColour, index)
        );
        resultCards.replaceChildren(...cards);
        stopAutomaticScanning();
        const names = recognizedFaces.map((f) => f.profile.name).join(", ");
        setStatus(`🎯 Identified: ${names}! Profile details loaded below.`, "success");
      } else {
        if (recording) {
          setStatus("Recording video… face visible, matching enrolled profiles…", "warning");
        } else {
          showNoMatch("Face detected, but it does not match enrolled profiles (Alia Bhatt, Sanskar Jain, Deepak Sharma).");
          setStatus("Visible face did not match enrolled profiles.", "warning");
        }
      }
    } catch (error) {
      console.error("Scan failed", error);
      setStatus("The scan could not finish. Improve lighting and try again.", "error");
    } finally {
      isScanning = false;
      overlay.hidden = true;
      scanButton.disabled = !stream;
      scanButton.textContent = scanningPaused ? "Restart Scanning" : "Scan Face / Object";
    }
  }

  function recordingMimeType() {
    if (!window.MediaRecorder?.isTypeSupported) return "";
    return ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"]
      .find((type) => MediaRecorder.isTypeSupported(type)) || "";
  }

  function finishRecording() {
    const blob = new Blob(recordingChunks, { type: mediaRecorder?.mimeType || "video/webm" });
    if (blob.size) {
      clearRecordedVideo();
      recordingUrl = URL.createObjectURL(blob);
      lastCapturedVideo.src = recordingUrl;
      lastVideoCard.hidden = false;
    }
    const count = recordingMatchesByPerson.size;
    if (count) {
      renderRecordingMatches();
      setStatus(`Video captured. ${count} enrolled face${count === 1 ? "" : "s"} identified with separate details below.`);
    } else {
      showNoMatch("No enrolled face was identified while this video was being recorded.");
      setStatus("Video captured, but no enrolled face was identified.", "warning");
    }
    setRecordingStatus("");
    recordVideoButton.classList.remove("is-recording");
    recordVideoButton.textContent = "Capture Video";
    recordVideoButton.disabled = !stream || !matcher;
    if (stream) startAutomaticScanning(CAMERA_SCAN_INTERVAL);
  }

  function startRecording() {
    if (!stream) {
      setStatus("Start the camera before capturing a video.", "warning");
      return;
    }
    if (!matcher) {
      setStatus("Recognition is still loading. Please wait a moment.", "warning");
      return;
    }
    if (!window.MediaRecorder) {
      setStatus("This browser cannot record video. Open the site in a current Chrome, Edge, or Firefox browser.", "error");
      return;
    }
    try {
      clearResult();
      clearRecordedVideo();
      recordingMatchesByPerson.clear();
      recordingChunks = [];
      recordingStartedAt = Date.now();
      scanningPaused = false;
      const mimeType = recordingMimeType();
      mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      mediaRecorder.addEventListener("dataavailable", (event) => {
        if (event.data?.size) recordingChunks.push(event.data);
      });
      mediaRecorder.addEventListener("stop", finishRecording, { once: true });
      mediaRecorder.start(1000);
      recordVideoButton.classList.add("is-recording");
      recordVideoButton.textContent = "Stop Recording";
      setRecordingStatus("● Recording — looking for enrolled faces in every frame.");
      setStatus("Recording video and scanning for enrolled faces…");
      startAutomaticScanning(RECORDING_SCAN_INTERVAL);
      window.setTimeout(scanFrame, 120);
    } catch (error) {
      console.error("Video recording failed", error);
      setStatus("Could not start video recording. Try again after restarting the camera.", "error");
    }
  }

  function syncCameraFrameRatio() {
    if (!video || !video.videoWidth || !video.videoHeight) return;
    const ratio = video.videoWidth / video.videoHeight;
    if (cameraFrame) {
      cameraFrame.dataset.videoRatio = `${video.videoWidth}:${video.videoHeight}`;
      cameraFrame.style.aspectRatio = ratio ? `${ratio}` : "16 / 9";
    }
    video.style.objectFit = ratio >= 1 ? "contain" : "cover";
  }

  function toggleRecording() {
    if (isRecording()) {
      recordVideoButton.disabled = true;
      recordVideoButton.textContent = "Saving Video…";
      mediaRecorder.stop();
      return;
    }
    startRecording();
  }

  async function startCamera() {
    clearResult();
    if (!window.isSecureContext) {
      setStatus("Camera access needs HTTPS (or localhost). Open this site through Vercel or use localhost.", "error");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("Use a current Chrome, Edge, or Firefox browser and allow camera access.", "error");
      return;
    }

    if (stream) {
      stopCameraStream();
    }

    try {
      const compactScreen = window.matchMedia("(max-width: 600px)").matches;
      const attempts = [
        {
          video: compactScreen
            ? { facingMode: { exact: cameraFacingMode }, width: { ideal: 1080 }, height: { ideal: 1920 }, aspectRatio: { ideal: 0.5625 } }
            : { facingMode: { exact: cameraFacingMode }, width: { ideal: 1920 }, height: { ideal: 1080 }, aspectRatio: { ideal: 1.7778 } },
          audio: false
        },
        { video: { facingMode: { ideal: cameraFacingMode } }, audio: false },
        { video: { facingMode: { ideal: cameraFacingMode === "user" ? "environment" : "user" } }, audio: false },
        { video: true, audio: false }
      ];

      let lastCameraError = null;
      for (const constraints of attempts) {
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
          break;
        } catch (error) {
          lastCameraError = error;
          if (["NotAllowedError", "SecurityError"].includes(error.name)) throw error;
        }
      }

      if (!stream) {
        const error = new Error(lastCameraError?.message || "No camera device could be opened.");
        error.name = lastCameraError?.name || "NotReadableError";
        throw error;
      }

      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      video.setAttribute("playsinline", "true");
      video.addEventListener("loadedmetadata", syncCameraFrameRatio, { once: true });
      await video.play();
      syncCameraFrameRatio();
      video.classList.toggle("rear-camera", cameraFacingMode === "environment");
      placeholder.hidden = true;
      startButton.textContent = "Camera On";
      startButton.disabled = true;
      switchCameraButton.disabled = false;
      switchCameraButton.textContent = cameraFacingMode === "user" ? "Use Back Camera" : "Use Front Camera";
      scanButton.disabled = false;
      recordVideoButton.disabled = !matcher || !window.MediaRecorder;
      recordVideoButton.textContent = "Capture Video";
      recordVideoButton.classList.remove("is-recording");
      setRecordingStatus("");
      setStatus(
        cameraFacingMode === "user"
          ? "Front camera ready. Click Capture Video or Scan Face."
          : "Back camera active! Point camera at Alia Bhatt or click Capture Video to record & identify."
      );
      return true;
    } catch (error) {
      const denied = error.name === "NotAllowedError" || error.name === "SecurityError";
      const unavailable = error.name === "NotReadableError" || error.name === "AbortError";
      setStatus(
        denied
          ? "Camera permission was denied. Click the lock icon in the address bar, allow Camera, then reload."
          : unavailable
            ? "Camera is busy or blocked by Windows. Close Teams, Zoom, WhatsApp, or another camera app, then click Start Camera again."
            : `Could not start camera: ${error.message}`,
        "error"
      );
      switchCameraButton.disabled = true;
      return false;
    }
  }

  async function switchCamera() {
    if (!stream) return;
    if (isRecording()) {
      setStatus("Stop video recording before switching cameras.", "warning");
      return;
    }
    const previousMode = cameraFacingMode;
    cameraFacingMode = cameraFacingMode === "user" ? "environment" : "user";
    switchCameraButton.disabled = true;
    stopCameraStream();
    const started = await startCamera();
    if (!started) {
      cameraFacingMode = previousMode;
      setStatus("The selected camera is unavailable. The previous camera was kept.", "warning");
      await startCamera();
    }
  }

  poseEmojiRail?.querySelectorAll(".pose-emoji-btn").forEach((button) => {
    button.addEventListener("click", () => {
      const label = button.dataset.label || "Thinking";
      const emoji = button.dataset.emoji || "🤔";
      activatePoseEmoji(label);
      if (label === "Video_Record") {
        if (!isRecording()) {
          startRecording();
        }
      } else if (label === "zoom_In") {
        applyCameraZoom(1.2);
      } else if (label === "Zoom_Out") {
        applyCameraZoom(0.9);
      } else {
        applyCameraZoom(1);
      }
      poseEmojiRail.querySelectorAll(".pose-emoji-btn").forEach((item) => item.classList.toggle("active", item === button));
    });
  });

  startButton.addEventListener("click", startCamera);
  switchCameraButton.addEventListener("click", switchCamera);
  scanButton.addEventListener("click", scheduleStableScan);
  uploadButton.addEventListener("click", () => uploadImageInput.click());
  uploadImageInput.addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    if (file) await matchUploadedImage(file);
  });
  recordVideoButton.addEventListener("click", toggleRecording);
  window.addEventListener("beforeunload", () => {
    stopCameraStream();
    if (recordingUrl) URL.revokeObjectURL(recordingUrl);
  });
  // Load the face model before the user starts the camera.
  loadRecognition().catch(modelError);
  loadPoseModel();
  loadHandGestureModel();

  // ---- Tab Switching & Dictionary Module ----
  const tabScanner = document.querySelector("#tabScanner");
  const tabDictionary = document.querySelector("#tabDictionary");
  const scannerSection = document.querySelector("#scannerSection");
  const dictionarySection = document.querySelector("#dictionarySection");

  const dictSearchForm = document.querySelector("#dictSearchForm");
  const dictSearchInput = document.querySelector("#dictSearchInput");
  const dictClearButton = document.querySelector("#dictClearButton");
  const dictCategoryPills = document.querySelector("#dictCategoryPills");
  const dictDetailCard = document.querySelector("#dictDetailCard");
  const dictDetailCategory = document.querySelector("#dictDetailCategory");
  const dictDetailTitle = document.querySelector("#dictDetailTitle");
  const dictDetailSlug = document.querySelector("#dictDetailSlug");
  const dictDetailPurpose = document.querySelector("#dictDetailPurpose");
  const dictDetailDescription = document.querySelector("#dictDetailDescription");
  const dictDetailPhotosSection = document.querySelector("#dictDetailPhotosSection");
  const dictDetailPhotosGrid = document.querySelector("#dictDetailPhotosGrid");
  const dictDetailPhotosCount = document.querySelector("#dictDetailPhotosCount");
  const dictPhotoChipsContainer = document.querySelector("#dictPhotoChipsContainer");
  const dictPhotoChips = document.querySelector("#dictPhotoChips");
  const dictDetailVisual = document.querySelector("#dictDetailVisual");
  const dictDetailLocations = document.querySelector("#dictDetailLocations");
  const dictDetailMaterials = document.querySelector("#dictDetailMaterials");
  const dictDetailAliases = document.querySelector("#dictDetailAliases");
  const dictSpecFolder = document.querySelector("#dictSpecFolder");
  const dictSpecLabel = document.querySelector("#dictSpecLabel");
  const dictSpecPhotos = document.querySelector("#dictSpecPhotos");
  const dictSpecNotes = document.querySelector("#dictSpecNotes");
  const dictSpeakBtn = document.querySelector("#dictSpeakBtn");
  const dictCopyBtn = document.querySelector("#dictCopyBtn");
  const dictResultsCount = document.querySelector("#dictResultsCount");
  const dictResultsSub = document.querySelector("#dictResultsSub");
  const dictResultsGrid = document.querySelector("#dictResultsGrid");
  const dictToggleAddBtn = document.querySelector("#dictToggleAddBtn");
  const dictAddFormContainer = document.querySelector("#dictAddFormContainer");
  const dictAddForm = document.querySelector("#dictAddForm");
  const dictAddStatus = document.querySelector("#dictAddStatus");
  const gotoDictBtn = document.querySelector("#gotoDictBtn");

  let activeCategory = "";
  let activeEntry = null;
  let dictionaryLoadedOnce = false;

  function switchTab(target) {
    if (target === "dictionary") {
      tabScanner?.classList.remove("active");
      tabScanner?.setAttribute("aria-selected", "false");
      tabDictionary?.classList.add("active");
      tabDictionary?.setAttribute("aria-selected", "true");
      scannerSection?.classList.remove("active");
      if (scannerSection) scannerSection.hidden = true;
      dictionarySection?.classList.add("active");
      if (dictionarySection) dictionarySection.hidden = false;
      if (!dictionaryLoadedOnce) {
        dictionaryLoadedOnce = true;
        lookupDictionaryEntry("Apple", true);
        loadPhotoChips();
      }
    } else {
      tabDictionary?.classList.remove("active");
      tabDictionary?.setAttribute("aria-selected", "false");
      tabScanner?.classList.add("active");
      tabScanner?.setAttribute("aria-selected", "true");
      dictionarySection?.classList.remove("active");
      if (dictionarySection) dictionarySection.hidden = true;
      scannerSection?.classList.add("active");
      if (scannerSection) scannerSection.hidden = false;
    }
  }

  tabScanner?.addEventListener("click", () => switchTab("scanner"));
  tabDictionary?.addEventListener("click", () => switchTab("dictionary"));
  gotoDictBtn?.addEventListener("click", () => switchTab("dictionary"));

  if (window.location.hash === "#dictionary") {
    switchTab("dictionary");
  }

  function displayDetail(entry) {
    if (!entry || !dictDetailCard) return;
    activeEntry = entry;
    dictDetailCard.hidden = false;
    if (dictDetailCategory) dictDetailCategory.textContent = entry.category || "General";
    if (dictDetailTitle) dictDetailTitle.textContent = entry.word || "Unknown";
    if (dictDetailSlug) dictDetailSlug.textContent = entry.slug || entry.scan_label || "";
    if (dictDetailPurpose) dictDetailPurpose.textContent = entry.primary_use || "Commonly used for daily activities, utility, or specialized tasks.";
    if (dictDetailDescription) dictDetailDescription.textContent = entry.description || "No description provided.";
    if (dictDetailVisual) dictDetailVisual.textContent = entry.visual_identification || "Not specified.";
    if (dictDetailLocations) dictDetailLocations.textContent = entry.common_locations || "Various environments.";

    // Materials
    if (dictDetailMaterials) {
      dictDetailMaterials.replaceChildren();
      const materials = Array.isArray(entry.materials) ? entry.materials : [entry.materials].filter(Boolean);
      if (materials.length) {
        materials.forEach(mat => {
          const span = document.createElement("span");
          span.className = "dict-material-pill";
          span.textContent = mat;
          dictDetailMaterials.appendChild(span);
        });
      } else {
        dictDetailMaterials.textContent = "Standard material composition";
      }
    }

    // Aliases
    if (dictDetailAliases) {
      dictDetailAliases.replaceChildren();
      const aliases = Array.isArray(entry.aliases) ? entry.aliases : [entry.aliases].filter(Boolean);
      if (aliases.length) {
        aliases.forEach(al => {
          const span = document.createElement("span");
          span.className = "dict-alias-pill";
          span.textContent = al;
          dictDetailAliases.appendChild(span);
        });
      } else {
        dictDetailAliases.textContent = "None listed";
      }
    }

    // Supplied reference photos (real images uploaded for this keyword, if any)
    if (dictDetailPhotosSection && dictDetailPhotosGrid) {
      const photos = Array.isArray(entry.reference_images) ? entry.reference_images : [];
      dictDetailPhotosGrid.replaceChildren();
      if (photos.length) {
        dictDetailPhotosSection.hidden = false;
        if (dictDetailPhotosCount) dictDetailPhotosCount.textContent = `(${photos.length})`;
        photos.forEach((url, i) => {
          const fig = document.createElement("figure");
          fig.className = "dict-photo-item";
          const img = document.createElement("img");
          img.src = url;
          img.alt = `${entry.word || "Object"} reference photo ${i + 1}`;
          img.loading = "lazy";
          fig.appendChild(img);
          dictDetailPhotosGrid.appendChild(fig);
        });
      } else {
        dictDetailPhotosSection.hidden = true;
        if (dictDetailPhotosCount) dictDetailPhotosCount.textContent = "";
      }
    }

    // Specifications
    if (dictSpecFolder) dictSpecFolder.textContent = entry.image_folder || entry.slug || "folder";
    if (dictSpecLabel) dictSpecLabel.textContent = entry.scan_label || entry.slug || "label";
    const rec = entry.recommended_images || {};
    if (dictSpecPhotos) dictSpecPhotos.textContent = `${rec.minimum || 25} min / ${rec.better || 50} optimal`;
    if (dictSpecNotes) dictSpecNotes.textContent = rec.capture_notes || "Ensure clear visibility and varied lighting.";

    // Highlight card in results grid if present
    document.querySelectorAll(".dict-card").forEach(c => {
      c.classList.toggle("selected", c.dataset.id === String(entry.id));
    });
  }

  async function loadPhotoChips() {
    if (!dictPhotoChips || !dictPhotoChipsContainer) return;
    try {
      const res = await fetch("/api/dictionary/photographed");
      if (!res.ok) return;
      const data = await res.json();
      const objects = data.objects || [];
      dictPhotoChips.replaceChildren();
      if (!objects.length) {
        dictPhotoChipsContainer.hidden = true;
        return;
      }
      objects.forEach(obj => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "photo-chip";
        chip.dataset.word = obj.word;
        chip.innerHTML = `📷 ${obj.word} <span class="photo-chip-count">${obj.image_count}</span>`;
        chip.addEventListener("click", () => {
          if (dictSearchInput) {
            dictSearchInput.value = obj.word;
            if (dictClearButton) dictClearButton.hidden = false;
          }
          lookupDictionaryEntry(obj.word, true);
        });
        dictPhotoChips.appendChild(chip);
      });
      dictPhotoChipsContainer.hidden = false;
    } catch (err) {
      console.error("Could not load photographed objects", err);
    }
  }

  async function lookupDictionaryEntry(keyword, populateGrid = false) {
    try {
      if (dictResultsSub) dictResultsSub.textContent = "Searching dictionary...";
      const res = await fetch(`/api/dictionary/entry/${encodeURIComponent(keyword)}`);
      if (res.ok) {
        const data = await res.json();
        if (data.found && data.entry) {
          displayDetail(data.entry);
        }
      }
      if (populateGrid) {
        await executeDictionarySearch(keyword, "");
      }
    } catch (err) {
      console.error("Dictionary lookup error", err);
    }
  }

  async function executeDictionarySearch(query = "", category = "") {
    if (!dictResultsGrid) return;
    try {
      dictResultsGrid.replaceChildren();
      const params = new URLSearchParams();
      if (query) params.set("q", query);
      if (category) params.set("category", category);
      params.set("limit", "40");

      const res = await fetch(`/api/dictionary/search?${params.toString()}`);
      if (!res.ok) throw new Error("Failed to search dictionary.");
      const data = await res.json();
      const results = data.results || [];

      if (dictResultsCount) {
        dictResultsCount.textContent = query || category
          ? `Found ${results.length} result(s)`
          : `Physical Object Directory (${results.length} featured)`;
      }
      if (dictResultsSub) {
        dictResultsSub.textContent = query
          ? `Showing matches for "${query}"${category ? ` in ${category}` : ""}`
          : category
            ? `Filtered by category: ${category}`
            : "Click any item to view full definition & purpose";
      }

      if (results.length === 0) {
        const empty = document.createElement("div");
        empty.className = "dict-empty-state";
        empty.innerHTML = `
          <div class="dict-empty-icon">🔍</div>
          <h4>No matching objects found</h4>
          <p>Try searching for a different keyword like "Apple", "Chair", "Laptop", or add a custom word below.</p>
        `;
        dictResultsGrid.appendChild(empty);
        return;
      }

      // If active entry is not set, display the first result
      if (!activeEntry && results.length > 0) {
        displayDetail(results[0]);
      }

      results.forEach(item => {
        const card = document.createElement("article");
        card.className = "dict-card";
        card.dataset.id = String(item.id);
        if (activeEntry && String(activeEntry.id) === String(item.id)) {
          card.classList.add("selected");
        }

        const top = document.createElement("div");
        top.className = "dict-card-top";

        const title = document.createElement("h4");
        title.className = "dict-card-title";
        title.textContent = item.word || item.slug;

        const cat = document.createElement("span");
        cat.className = "dict-card-cat";
        cat.textContent = item.category || "Item";

        top.appendChild(title);
        top.appendChild(cat);
        if (item.has_reference_images) {
          const photoBadge = document.createElement("span");
          photoBadge.className = "dict-card-photo-badge";
          photoBadge.title = "Real reference photos supplied for this object";
          photoBadge.textContent = `📷 ${(item.reference_images || []).length}`;
          top.appendChild(photoBadge);
        }

        const desc = document.createElement("p");
        desc.className = "dict-card-desc";
        desc.textContent = item.description || "";

        const purpose = document.createElement("p");
        purpose.className = "dict-card-purpose-peek";
        purpose.innerHTML = `<strong>Why use it:</strong> ${item.primary_use || "Daily utility & usage"}`;

        const footer = document.createElement("div");
        footer.className = "dict-card-footer";
        footer.innerHTML = `<span>ID: #${item.id}</span><span class="dict-view-link">View Breakdown &rarr;</span>`;

        card.appendChild(top);
        card.appendChild(desc);
        card.appendChild(purpose);
        card.appendChild(footer);

        card.addEventListener("click", () => {
          displayDetail(item);
          dictDetailCard?.scrollIntoView({ behavior: "smooth", block: "nearest" });
        });

        dictResultsGrid.appendChild(card);
      });
    } catch (err) {
      console.error("Dictionary search error", err);
      if (dictResultsSub) dictResultsSub.textContent = "Error loading dictionary items.";
    }
  }

  // Search input and clear button handlers
  let searchDebounceTimer = null;
  dictSearchInput?.addEventListener("input", (e) => {
    const val = e.target.value.trim();
    if (dictClearButton) dictClearButton.hidden = !val;
    if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
      executeDictionarySearch(val, activeCategory);
    }, 280);
  });

  dictClearButton?.addEventListener("click", () => {
    if (dictSearchInput) dictSearchInput.value = "";
    dictClearButton.hidden = true;
    dictSearchInput?.focus();
    executeDictionarySearch("", activeCategory);
  });

  dictSearchForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const val = dictSearchInput ? dictSearchInput.value.trim() : "";
    executeDictionarySearch(val, activeCategory);
    if (val) lookupDictionaryEntry(val, false);
  });

  // Quick Tags
  document.querySelectorAll(".quick-tag").forEach(tag => {
    tag.addEventListener("click", () => {
      const word = tag.dataset.word;
      if (dictSearchInput) {
        dictSearchInput.value = word;
        if (dictClearButton) dictClearButton.hidden = false;
      }
      lookupDictionaryEntry(word, true);
    });
  });

  // Category Pills
  dictCategoryPills?.addEventListener("click", (e) => {
    const pill = e.target.closest(".category-pill");
    if (!pill) return;
    document.querySelectorAll(".category-pill").forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    activeCategory = pill.dataset.category || "";
    executeDictionarySearch(dictSearchInput ? dictSearchInput.value.trim() : "", activeCategory);
  });

  // Pronounce / Speak
  dictSpeakBtn?.addEventListener("click", () => {
    if (!activeEntry) return;
    if (!("speechSynthesis" in window)) {
      alert("Speech synthesis is not supported in this browser.");
      return;
    }
    window.speechSynthesis.cancel();
    const textToSpeak = `${activeEntry.word}. ${activeEntry.description}. Why use it: ${activeEntry.primary_use}`;
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  });

  // Copy Details
  dictCopyBtn?.addEventListener("click", async () => {
    if (!activeEntry) return;
    const formatted = `=== ${activeEntry.word} (Category: ${activeEntry.category}) ===\n` +
      `Description: ${activeEntry.description}\n` +
      `Why Use It / Purpose: ${activeEntry.primary_use}\n` +
      `Visual Identification: ${activeEntry.visual_identification}\n` +
      `Common Locations: ${activeEntry.common_locations}\n` +
      `Materials: ${(activeEntry.materials || []).join(", ")}\n` +
      `Aliases: ${(activeEntry.aliases || []).join(", ")}`;
    try {
      await navigator.clipboard.writeText(formatted);
      const span = dictCopyBtn.querySelector("span");
      if (span) {
        const originalText = span.textContent;
        span.textContent = "Copied! ✔";
        setTimeout(() => { span.textContent = originalText; }, 2000);
      }
    } catch (err) {
      console.warn("Clipboard copy failed", err);
    }
  });

  // Add Item Toggle
  dictToggleAddBtn?.addEventListener("click", () => {
    if (!dictAddFormContainer) return;
    const isHidden = dictAddFormContainer.hidden;
    dictAddFormContainer.hidden = !isHidden;
    dictToggleAddBtn.classList.toggle("expanded", isHidden);
  });

  // Add Item Form Submit
  dictAddForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const word = document.querySelector("#newWord")?.value.trim();
    const category = document.querySelector("#newCategory")?.value.trim() || "General";
    const description = document.querySelector("#newDescription")?.value.trim();
    const primary_use = document.querySelector("#newPrimaryUse")?.value.trim();
    const visual = document.querySelector("#newVisual")?.value.trim();
    const locations = document.querySelector("#newLocations")?.value.trim();
    const materials = document.querySelector("#newMaterials")?.value.trim();
    const aliases = document.querySelector("#newAliases")?.value.trim();

    if (!word || !description || !primary_use) {
      alert("Please fill in required fields: Word, Description, and Primary Purpose.");
      return;
    }

    try {
      if (dictAddStatus) {
        dictAddStatus.hidden = false;
        dictAddStatus.className = "dict-add-status";
        dictAddStatus.textContent = "Adding to dataset...";
      }
      const res = await fetch("/api/dictionary/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          word,
          category,
          description,
          primary_use,
          visual_identification: visual,
          common_locations: locations,
          materials,
          aliases
        })
      });
      const data = await res.json();
      if (res.ok && data.success) {
        if (dictAddStatus) {
          dictAddStatus.className = "dict-add-status success";
          dictAddStatus.textContent = `"${word}" successfully added to dataset!`;
        }
        dictAddForm.reset();
        displayDetail(data.entry);
        executeDictionarySearch(word, "");
        setTimeout(() => {
          if (dictAddFormContainer) dictAddFormContainer.hidden = true;
          dictToggleAddBtn?.classList.remove("expanded");
          if (dictAddStatus) dictAddStatus.hidden = true;
        }, 3000);
      } else {
        if (dictAddStatus) {
          dictAddStatus.className = "dict-add-status error";
          dictAddStatus.textContent = data.error || "Could not add entry.";
        }
      }
    } catch (err) {
      if (dictAddStatus) {
        dictAddStatus.className = "dict-add-status error";
        dictAddStatus.textContent = "Network error adding entry.";
      }
    }
  });
})();
