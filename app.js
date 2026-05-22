/**
 * Antigravity Sports Ball Tracker & Smart Panning Camera
 * Core Application Logic (PWA & Client-Side Computer Vision)
 */

// Global App State
const state = {
  // Model & Inference
  model: null,
  modelLoaded: false,
  isProcessing: false,
  inferenceThrottle: 1, // Run detection every N frames
  frameCount: 0,
  fps: 0,
  lastFpsUpdate: 0,
  fpsFrames: 0,

  // Stream & Video
  videoEl: null,
  canvasEl: null,
  ctx: null,
  stream: null,
  activeCamera: 'environment', // 'environment' (back) or 'user' (front)
  activeResolution: '720p',    // '720p' or '1080p'
  videoWidth: 1280,
  videoHeight: 720,

  // Virtual Camera (Center Stage) Viewport Coordinates
  viewport: {
    x: 640,
    y: 360,
    w: 1280,
    h: 720,
    targetX: 640,
    targetY: 360,
    targetW: 1280,
    targetH: 720,
    panLerp: 0.05,
    zoomLerp: 0.03,
    maxZoom: 2.2,
    mode: 'player_with_ball', // Default to action-oriented Player with Ball tracking!
  },

  // Ball Memory / Tracking Prediction System
  ballTracker: {
    lastX: null,
    lastY: null,
    vx: 0,
    vy: 0,
    lostFrames: 0,
    maxMemoryFrames: 15, // Track occluded ball for 0.5s at 30FPS
  },

  // Possession & Action Tracking
  possession: {
    holder: null,        // Bounding box of the player currently holding the ball
    lastHolder: null,    // Coords/box of the last player who had the ball
    isBallInAir: false,  // Whether the ball is currently in the air (pass/shot)
    airTime: 0,          // Duration ball has been in the air
  },

  // Recording State
  mediaRecorder: null,
  recordedChunks: [],
  isRecording: false,
  recordStartTime: 0,
  recordTimerInterval: null,

  // UI state
  showDebug: true,
  ecoMode: false,
};

// UI Selectors
const DOM = {
  video: () => document.getElementById('rawVideo'),
  canvas: () => document.getElementById('virtualCanvas'),
  aiStatus: () => document.getElementById('aiStatus'),
  fpsCounter: () => document.getElementById('fpsCounter'),
  targetCounter: () => document.getElementById('targetCounter'),
  flipCameraBtn: () => document.getElementById('flipCameraBtn'),
  toggleDebugBtn: () => document.getElementById('toggleDebugBtn'),
  streamInfoBtn: () => document.getElementById('streamInfoBtn'),
  openSettingsBtn: () => document.getElementById('openSettingsBtn'),
  closeSettingsBtn: () => document.getElementById('closeSettingsBtn'),
  recordBtn: () => document.getElementById('recordBtn'),
  recordTimer: () => document.getElementById('recordTimer'),
  toggleEcoBtn: () => document.getElementById('toggleEcoBtn'),
  settingsDrawer: () => document.getElementById('settingsDrawer'),
  streamModal: () => document.getElementById('streamModal'),
  closeStreamBtn: () => document.getElementById('closeStreamBtn'),
  cameraSelector: () => document.getElementById('cameraSelector'),
  smoothnessSlider: () => document.getElementById('smoothnessSlider'),
  smoothnessVal: () => document.getElementById('smoothnessVal'),
  zoomSlider: () => document.getElementById('zoomSlider'),
  zoomVal: () => document.getElementById('zoomVal'),
  throttleSlider: () => document.getElementById('throttleSlider'),
  throttleVal: () => document.getElementById('throttleVal'),
  toastContainer: () => document.getElementById('toastContainer'),
  quickResolution: () => document.getElementById('quickResolution'),
  quickProfile: () => document.getElementById('quickProfile'),
  quickZoom: () => document.getElementById('quickZoom'),
};

// Initialize Application on Window Load
window.addEventListener('DOMContentLoaded', () => {
  registerServiceWorker();
  initApp();
});

// 1. PWA Service Worker Registration
function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js')
      .then(() => console.log('[PWA] Service Worker Registered'))
      .catch((err) => console.error('[PWA] Service Worker Failed', err));
  }
}

// 2. Main Application Initialization
async function initApp() {
  state.videoEl = DOM.video();
  state.canvasEl = DOM.canvas();
  state.ctx = state.canvasEl.getContext('2d');

  // Handle Resize Events (supports phone orientation changes!)
  window.addEventListener('resize', handleResize);
  handleResize();

  // Load Model
  await loadAiModel();

  // Setup Camera Stream
  await setupCamera();

  // Bind UI Events
  setupEventListeners();

  // Toast AI is Ready
  showToast('Welcome to Antigravity Tracker! Point at a court to begin.', 'info');

  // Start Core Panning and Processing Loop
  requestAnimationFrame(processFrame);
}

// 3. Responsive Canvas Fitting
function handleResize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = state.canvasEl.getBoundingClientRect();
  
  // Set drawing buffer resolution matching device physical pixels
  state.canvasEl.width = rect.width * dpr;
  state.canvasEl.height = rect.height * dpr;
  
  console.log(`[Canvas] Resized to ${state.canvasEl.width}x${state.canvasEl.height} (DPR: ${dpr})`);
}

// 4. Load COCO-SSD Object Detector
async function loadAiModel() {
  try {
    console.log('[AI] Loading TensorFlow.js model...');
    const statusVal = DOM.aiStatus().querySelector('.value');
    
    // cocoSsd is loaded via global script tag CDN
    if (typeof cocoSsd === 'undefined') {
      throw new Error('COCO-SSD library failed to load. Check connection.');
    }

    // Set WebGL backend (uses mobile GPU acceleration!)
    if (tf) {
      await tf.setBackend('webgl');
      await tf.ready();
      console.log('[TensorFlow] Backend configured: WebGL GPU');
    }

    state.model = await cocoSsd.load({
      base: 'lite_mobilenet_v2' // Fast mobile-optimized model architecture
    });
    
    state.modelLoaded = true;
    statusVal.textContent = 'READY';
    statusVal.className = 'value state-active';
    console.log('[AI] Model loaded successfully.');
    showToast('AI Detection Engine loaded successfully!', 'success');
  } catch (error) {
    console.error('[AI] Model load error:', error);
    const statusVal = DOM.aiStatus().querySelector('.value');
    statusVal.textContent = 'ERROR';
    statusVal.style.color = '#ff3b30';
    showToast('Failed to load AI model. Using manual/center camera fallback.', 'error');
  }
}

// 5. Camera Stream Setup
async function setupCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
  }

  const is1080 = state.activeResolution === '1080p';
  console.log(`[Camera] Requesting camera constraints for resolution: ${state.activeResolution} (${is1080 ? '1920x1080' : '1280x720'})`);

  // Mobile camera constraints
  const constraints = {
    audio: true, // Captures court sounds/screams for high quality videos!
    video: {
      facingMode: state.activeCamera === 'environment' ? 'environment' : 'user',
      width: { ideal: is1080 ? 1920 : 1280 },
      height: { ideal: is1080 ? 1080 : 720 },
      frameRate: { ideal: 30 }
    }
  };

  try {
    state.stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.videoEl.srcObject = state.stream;
    
    // Wait for video metadata to load so we know dimensions
    await new Promise((resolve) => {
      state.videoEl.onloadedmetadata = () => {
        state.videoWidth = state.videoEl.videoWidth;
        state.videoHeight = state.videoEl.videoHeight;
        
        // Reset virtual camera coordinates to physical frame center
        state.viewport.x = state.videoWidth / 2;
        state.viewport.y = state.videoHeight / 2;
        state.viewport.w = state.videoWidth;
        state.viewport.h = state.videoHeight;
        
        state.viewport.targetX = state.viewport.x;
        state.viewport.targetY = state.viewport.y;
        state.viewport.targetW = state.viewport.w;
        state.viewport.targetH = state.viewport.h;
        
        resolve();
      };
    });

    console.log(`[Camera] Started raw feed: ${state.videoWidth}x${state.videoHeight}`);
    showToast(`Resolution switched to ${state.videoWidth}x${state.videoHeight}`, 'success');
  } catch (error) {
    console.error('[Camera] Access denied or unavailable:', error);
    showToast('Failed to switch resolution. Using fallback camera stream.', 'error');
  }
}

// 6. UI Event Triggers
function setupEventListeners() {
  // Quick resolution selector in the header
  DOM.quickResolution().addEventListener('change', async (e) => {
    state.activeResolution = e.target.value;
    await setupCamera();
  });

  // Quick tracking profile selector in the header
  DOM.quickProfile().addEventListener('change', (e) => {
    state.viewport.mode = e.target.value;
    
    // Synchronize settings drawer radio buttons
    const radio = document.querySelector(`input[name="trackingMode"][value="${e.target.value}"]`);
    if (radio) radio.checked = true;
    
    showToast(`Tracking Profile: ${e.target.value.toUpperCase().replace(/_/g, ' ')}`, 'info');
  });

  // Quick zoom selector in the header
  DOM.quickZoom().addEventListener('change', (e) => {
    state.viewport.maxZoom = parseFloat(e.target.value);
    
    // Synchronize slider and text value in the settings drawer
    DOM.zoomSlider().value = state.viewport.maxZoom;
    DOM.zoomVal().textContent = state.viewport.maxZoom.toFixed(1) + 'x';
    
    showToast(`Max Zoom: ${state.viewport.maxZoom.toFixed(1)}x`, 'info');
  });

  // Flip Camera
  DOM.flipCameraBtn().addEventListener('click', async () => {
    state.activeCamera = state.activeCamera === 'environment' ? 'user' : 'environment';
    DOM.cameraSelector().value = state.activeCamera;
    await setupCamera();
    showToast(`Switched to ${state.activeCamera === 'environment' ? 'Rear' : 'Front'} camera`, 'info');
  });

  DOM.cameraSelector().addEventListener('change', async (e) => {
    state.activeCamera = e.target.value;
    await setupCamera();
    showToast(`Switched to ${state.activeCamera === 'environment' ? 'Rear' : 'Front'} camera`, 'info');
  });

  // Debug HUD overlay toggling
  DOM.toggleDebugBtn().addEventListener('click', () => {
    state.showDebug = !state.showDebug;
    DOM.toggleDebugBtn().classList.toggle('active', state.showDebug);
    showToast(`AI HUD overlay ${state.showDebug ? 'Enabled' : 'Disabled'}`, 'info');
  });

  // Battery Eco Mode toggle
  DOM.toggleEcoBtn().addEventListener('click', () => {
    state.ecoMode = !state.ecoMode;
    DOM.toggleEcoBtn().classList.toggle('active', state.ecoMode);
    
    // Eco Mode throttles detection to every 3 frames (10 FPS detection, 60 FPS smooth interpolation rendering!)
    state.inferenceThrottle = state.ecoMode ? 3 : 1;
    DOM.throttleSlider().value = state.inferenceThrottle;
    DOM.throttleVal().textContent = state.ecoMode ? '3 (Battery Saver)' : '1 (Max Performance)';
    
    showToast(`Battery Saver ${state.ecoMode ? 'Activated (FPS throttling)' : 'Deactivated'}`, 'info');
  });

  // Settings Slide-up Drawer
  DOM.openSettingsBtn().addEventListener('click', () => {
    DOM.settingsDrawer().classList.add('open');
  });
  DOM.closeSettingsBtn().addEventListener('click', () => {
    DOM.settingsDrawer().classList.remove('open');
  });

  // Stream instructions modal
  DOM.streamInfoBtn().addEventListener('click', () => {
    DOM.streamModal().classList.add('open');
  });
  DOM.closeStreamBtn().addEventListener('click', () => {
    DOM.streamModal().classList.remove('open');
  });
  DOM.streamModal().addEventListener('click', (e) => {
    if (e.target === DOM.streamModal()) {
      DOM.streamModal().classList.remove('open');
    }
  });

  // Tuning Sliders Handlers
  DOM.smoothnessSlider().addEventListener('input', (e) => {
    state.viewport.panLerp = parseFloat(e.target.value);
    DOM.smoothnessVal().textContent = state.viewport.panLerp.toFixed(2);
  });

  DOM.zoomSlider().addEventListener('input', (e) => {
    state.viewport.maxZoom = parseFloat(e.target.value);
    DOM.zoomVal().textContent = state.viewport.maxZoom.toFixed(1) + 'x';
    
    // Synchronize quick zoom selector on top-right HUD
    const val = state.viewport.maxZoom;
    const options = Array.from(DOM.quickZoom().options).map(opt => parseFloat(opt.value));
    const closest = options.reduce((prev, curr) => Math.abs(curr - val) < Math.abs(prev - val) ? curr : prev);
    DOM.quickZoom().value = closest.toFixed(1);
  });

  DOM.throttleSlider().addEventListener('input', (e) => {
    state.inferenceThrottle = parseInt(e.target.value);
    let cap = state.inferenceThrottle.toString();
    if (state.inferenceThrottle === 1) cap += ' (Max Performance)';
    else if (state.inferenceThrottle === 3) cap += ' (Battery Saver)';
    DOM.throttleVal().textContent = cap;
  });

  // Tracking profile modes
  const modeRadios = document.querySelectorAll('input[name="trackingMode"]');
  modeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
      state.viewport.mode = e.target.value;
      
      // Synchronize quick header dropdown select
      DOM.quickProfile().value = e.target.value;
      
      showToast(`Tracking Profile: ${e.target.value.toUpperCase().replace(/_/g, ' ')}`, 'info');
    });
  });

  // Canvas Media Recorder Trigger
  DOM.recordBtn().addEventListener('click', () => {
    if (state.isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });
}

// 7. Core Canvas Loop (AI Object Detection + Center Stage Math)
async function processFrame(timestamp) {
  state.frameCount++;

  // Update real FPS telemetry
  if (!state.lastFpsUpdate) state.lastFpsUpdate = timestamp;
  state.fpsFrames++;
  if (timestamp - state.lastFpsUpdate >= 1000) {
    state.fps = Math.round((state.fpsFrames * 1000) / (timestamp - state.lastFpsUpdate));
    DOM.fpsCounter().textContent = state.fps;
    state.fpsFrames = 0;
    state.lastFpsUpdate = timestamp;
  }

  // 1. Fetch AI detections if model is loaded and ready, respect frames throttler
  let detections = [];
  if (state.modelLoaded && !state.isProcessing && (state.frameCount % state.inferenceThrottle === 0)) {
    state.isProcessing = true;
    try {
      // Pass video element to run on GPU WebGL
      detections = await state.model.detect(state.videoEl);
    } catch (e) {
      console.error('[Vision Loop] Inference failed', e);
    } finally {
      state.isProcessing = false;
    }
  }

  // Filter for sports ball and players
  const ballDetections = detections.filter(d => d.class === 'sports ball' && d.score >= 0.25);
  const playerDetections = detections.filter(d => d.class === 'person' && d.score >= 0.35);
  
  if (detections.length > 0) {
    DOM.targetCounter().textContent = ballDetections.length + playerDetections.length;
  }

  // 2. EXECUTE BALL TRACKING AND MEMORY PREDICTION ALGORITHMS
  let activeBall = null;
  
  if (ballDetections.length > 0) {
    // Pick the most confident ball detection
    const ball = ballDetections.sort((a, b) => b.score - a.score)[0];
    const bx = ball.bbox[0] + ball.bbox[2]/2;
    const by = ball.bbox[1] + ball.bbox[3]/2;
    
    // Compute current velocity (speed of pass or bounce!)
    if (state.ballTracker.lastX !== null) {
      state.ballTracker.vx = bx - state.ballTracker.lastX;
      state.ballTracker.vy = by - state.ballTracker.lastY;
    } else {
      state.ballTracker.vx = 0;
      state.ballTracker.vy = 0;
    }

    state.ballTracker.lastX = bx;
    state.ballTracker.lastY = by;
    state.ballTracker.lostFrames = 0;
    
    activeBall = { x: bx, y: by, w: ball.bbox[2], h: ball.bbox[3], score: ball.score, prediction: false };
  } else {
    // If ball is lost, trigger ball memory prediction
    if (state.ballTracker.lastX !== null && state.ballTracker.lostFrames < state.ballTracker.maxMemoryFrames) {
      // Predict ball's position using simple physics (inertia + friction decay)
      state.ballTracker.lastX += state.ballTracker.vx;
      state.ballTracker.lastY += state.ballTracker.vy;
      
      // Decay velocity slowly so the camera doesn't fly off-screen
      state.ballTracker.vx *= 0.92;
      state.ballTracker.vy *= 0.92;
      state.ballTracker.lostFrames++;

      activeBall = {
        x: state.ballTracker.lastX,
        y: state.ballTracker.lastY,
        w: 40, // Standard size
        h: 40,
        score: 0.5,
        prediction: true // Custom flag to draw a dashed reticle
      };
    } else {
      // Ball fully lost, clear tracker memory
      state.ballTracker.lastX = null;
      state.ballTracker.lastY = null;
      state.ballTracker.vx = 0;
      state.ballTracker.vy = 0;
    }
  }

  // 2.5 EXECUTE PLAYER POSSESSION DETECTION HEURISTIC
  let ballHolder = null;
  
  if (activeBall) {
    let minDistance = Infinity;
    playerDetections.forEach(p => {
      const px = p.bbox[0];
      const py = p.bbox[1];
      const pw = p.bbox[2];
      const ph = p.bbox[3];
      
      // Possession overlap: Pad horizontally by 15% and vertically by 10%
      const padW = pw * 0.15;
      const padH = ph * 0.10;
      
      const bx = activeBall.x;
      const by = activeBall.y;
      
      if (bx >= px - padW && bx <= px + pw + padW &&
          by >= py - padH && by <= py + ph + padH) {
        
        const pCenterX = px + pw / 2;
        const pCenterY = py + ph / 2;
        const dist = Math.hypot(pCenterX - bx, pCenterY - by);
        
        if (dist < minDistance) {
          minDistance = dist;
          ballHolder = p;
        }
      }
    });
    
    if (ballHolder) {
      state.possession.holder = ballHolder;
      state.possession.lastHolder = {
        x: ballHolder.bbox[0] + ballHolder.bbox[2] / 2,
        y: ballHolder.bbox[1] + ballHolder.bbox[3] / 2,
        w: ballHolder.bbox[2],
        h: ballHolder.bbox[3],
        timestamp: Date.now()
      };
      state.possession.isBallInAir = false;
      state.possession.airTime = 0;
    } else {
      // Ball is detected, but no player is holding it (it is in the air!)
      state.possession.holder = null;
      state.possession.isBallInAir = true;
      state.possession.airTime++;
    }
  } else {
    // Ball is lost. If it was in the air, let it stay in the air for up to 30 frames (1s) of tracking memory
    state.possession.holder = null;
    if (state.possession.isBallInAir) {
      state.possession.airTime++;
      if (state.possession.airTime > 30) {
        state.possession.isBallInAir = false;
        state.possession.airTime = 0;
      }
    }
  }

  // 3. CENTER STAGE MATH - COMPUTE TARGET VIEWPORT
  const aspect = state.canvasEl.width / state.canvasEl.height;
  
  let targetX = state.videoWidth / 2;
  let targetY = state.videoHeight / 2;
  let targetW = state.videoWidth;

  const mode = state.viewport.mode;
  
  if (mode === 'ball' && activeBall) {
    // Pan strictly around ball
    targetX = activeBall.x;
    targetY = activeBall.y;
    targetW = state.videoWidth / state.viewport.maxZoom; // Crop closely
  } 
  else if (mode === 'players' && playerDetections.length > 0) {
    // Pan to center of active players
    let sumX = 0, sumY = 0;
    playerDetections.forEach(p => {
      sumX += p.bbox[0] + p.bbox[2]/2;
      sumY += p.bbox[1] + p.bbox[3]/2;
    });
    targetX = sumX / playerDetections.length;
    targetY = sumY / playerDetections.length;
    
    // Zoom out based on players spread
    let minPX = Math.min(...playerDetections.map(p => p.bbox[0]));
    let maxPX = Math.max(...playerDetections.map(p => p.bbox[0] + p.bbox[2]));
    let spread = maxPX - minPX;
    
    // Dynamic viewport width containing all players with 30% padding
    let minW = state.videoWidth / state.viewport.maxZoom;
    targetW = Math.max(spread * 1.3, minW);
  } 
  else if (mode === 'player_with_ball') {
    // Track player currently holding the ball. Zoom out during passes/shots!
    if (ballHolder) {
      // Focus strictly on the player holding the ball (tight zoom!)
      targetX = ballHolder.bbox[0] + ballHolder.bbox[2] / 2;
      targetY = ballHolder.bbox[1] + ballHolder.bbox[3] / 2;
      targetW = state.videoWidth / state.viewport.maxZoom; // Crop closely
    } else if (activeBall && state.possession.isBallInAir) {
      // Ball is in the air (pass/shot). Zoom out to capture flight path & receiver!
      targetX = activeBall.x;
      targetY = activeBall.y;
      
      // Dynamic zoom out: set viewport to 1.25x zoom for cinematic court capture
      targetW = state.videoWidth / 1.25;
      
      // Frame the pass: bias center slightly towards the pass origin (last holder) so both are in frame
      if (state.possession.lastHolder && (Date.now() - state.possession.lastHolder.timestamp < 1500)) {
        targetX = activeBall.x * 0.7 + state.possession.lastHolder.x * 0.3;
        targetY = activeBall.y * 0.7 + state.possession.lastHolder.y * 0.3;
      }
    } else if (playerDetections.length > 0) {
      // Fallback: track center of players
      let sumX = 0, sumY = 0;
      playerDetections.forEach(p => {
        sumX += p.bbox[0] + p.bbox[2]/2;
        sumY += p.bbox[1] + p.bbox[3]/2;
      });
      targetX = sumX / playerDetections.length;
      targetY = sumY / playerDetections.length;
      targetW = state.videoWidth / 1.5; // Intermediate zoom
    }
  }
  else if (mode === 'combined') {
    // Dynamic combined mode (Smart Camera)
    if (activeBall) {
      targetX = activeBall.x;
      targetY = activeBall.y;
      
      // Look for players near the ball to include in crop
      const nearbyPlayers = playerDetections.filter(p => {
        const px = p.bbox[0] + p.bbox[2]/2;
        const py = p.bbox[1] + p.bbox[3]/2;
        const dist = Math.hypot(px - activeBall.x, py - activeBall.y);
        return dist < 450; // Dribblers and defenders within 450 pixels
      });

      if (nearbyPlayers.length > 0) {
        // Shift camera center slightly towards players to frame the duel!
        let sumX = activeBall.x;
        let sumY = activeBall.y;
        nearbyPlayers.forEach(p => {
          sumX += p.bbox[0] + p.bbox[2]/2;
          sumY += p.bbox[1] + p.bbox[3]/2;
        });
        targetX = sumX / (nearbyPlayers.length + 1);
        targetY = sumY / (nearbyPlayers.length + 1);

        // Zoom viewport dynamically to capture ball and adjacent players
        let minX = Math.min(activeBall.x, ...nearbyPlayers.map(p => p.bbox[0]));
        let maxX = Math.max(activeBall.x, ...nearbyPlayers.map(p => p.bbox[0] + p.bbox[2]));
        let minW = state.videoWidth / state.viewport.maxZoom;
        
        // If ball is in the air, force a wider crop to capture pass dynamics
        if (state.possession.isBallInAir) {
          targetW = Math.max((maxX - minX) * 1.5, state.videoWidth / 1.25);
        } else {
          targetW = Math.max((maxX - minX) * 1.5, minW);
        }
      } else {
        // Just ball, zoom in or out based on air status
        if (state.possession.isBallInAir) {
          targetW = state.videoWidth / 1.25;
        } else {
          targetW = state.videoWidth / state.viewport.maxZoom;
        }
      }
    } else if (playerDetections.length > 0) {
      // Fallback to tracking player center
      let sumX = 0, sumY = 0;
      playerDetections.forEach(p => {
        sumX += p.bbox[0] + p.bbox[2]/2;
        sumY += p.bbox[1] + p.bbox[3]/2;
      });
      targetX = sumX / playerDetections.length;
      targetY = sumY / playerDetections.length;
      targetW = state.videoWidth / 1.3; // Default intermediate zoom
    }
  }

  // 4. LERP INTERPOLATION (Cinematic Smoothing Equations)
  state.viewport.x += (targetX - state.viewport.x) * state.viewport.panLerp;
  state.viewport.y += (targetY - state.viewport.y) * state.viewport.panLerp;
  state.viewport.w += (targetW - state.viewport.w) * state.viewport.zoomLerp;
  
  // Guard rails - maintain canvas crop aspect ratio
  state.viewport.w = Math.min(state.viewport.w, state.videoWidth);
  state.viewport.h = state.viewport.w / aspect;

  if (state.viewport.h > state.videoHeight) {
    state.viewport.h = state.videoHeight;
    state.viewport.w = state.viewport.h * aspect;
  }

  // Clamp crop rectangle inside physical camera sensor boundaries (no black borders!)
  let cropLeft = state.viewport.x - state.viewport.w / 2;
  let cropTop = state.viewport.y - state.viewport.h / 2;

  cropLeft = Math.max(0, Math.min(cropLeft, state.videoWidth - state.viewport.w));
  cropTop = Math.max(0, Math.min(cropTop, state.videoHeight - state.viewport.h));

  // 5. DRAW VIRTUAL PANNING VIEWPORT TO CANVAS
  state.ctx.clearRect(0, 0, state.canvasEl.width, state.canvasEl.height);
  
  // Render cropped portion of the high-res raw stream to canvas
  state.ctx.drawImage(
    state.videoEl,
    cropLeft, cropTop, state.viewport.w, state.viewport.h, // Source (Cropped CropBox)
    0, 0, state.canvasEl.width, state.canvasEl.height      // Destination (Full Canvas)
  );

  // 6. RENDER DYNAMIC AI HUD / TELEMETRY OVERLAYS
  if (state.showDebug) {
    const scaleX = state.canvasEl.width / state.viewport.w;
    const scaleY = state.canvasEl.height / state.viewport.h;

    // Draw active player bounding boxes (holographic cyan overlay)
    playerDetections.forEach(p => {
      // Check if this player is currently the ball holder
      const isHolder = ballHolder &&
                       Math.abs(p.bbox[0] - ballHolder.bbox[0]) < 1 &&
                       Math.abs(p.bbox[1] - ballHolder.bbox[1]) < 1;

      // Translate coordinates from raw feed coordinates to local canvas crop coordinates
      const px = (p.bbox[0] - cropLeft) * scaleX;
      const py = (p.bbox[1] - cropTop) * scaleY;
      const pw = p.bbox[2] * scaleX;
      const ph = p.bbox[3] * scaleY;

      // Draw active possession ring under the player's feet (NBA style!)
      if (isHolder) {
        state.ctx.fillStyle = 'rgba(255, 85, 0, 0.25)';
        state.ctx.strokeStyle = '#ff5500';
        state.ctx.lineWidth = 3;
        state.ctx.beginPath();
        state.ctx.ellipse(px + pw / 2, py + ph, pw * 0.4, 8, 0, 0, Math.PI * 2);
        state.ctx.fill();
        state.ctx.stroke();
      }

      // Draw futuristic glass-border player box (orange for holder, cyan for other players)
      state.ctx.strokeStyle = isHolder ? '#ff5500' : '#00f0ff';
      state.ctx.lineWidth = isHolder ? 3 : 2;
      state.ctx.strokeRect(px, py, pw, ph);

      // Box corners accents
      drawCornerBrackets(state.ctx, px, py, pw, ph, 10, isHolder ? '#ff5500' : '#00f0ff');

      // Label background card
      state.ctx.fillStyle = isHolder ? 'rgba(255, 85, 0, 0.85)' : 'rgba(10, 11, 14, 0.7)';
      state.ctx.fillRect(px, py - 20, Math.max(90, pw * 0.5), 20);
      
      // Label text
      state.ctx.fillStyle = '#ffffff';
      state.ctx.font = "bold 10px 'Inter', sans-serif";
      state.ctx.fillText(isHolder ? `POSSESSION ${Math.round(p.score*100)}%` : `PLAYER ${Math.round(p.score*100)}%`, px + 6, py - 6);
    });

    // Draw sports ball neon orange lock-on HUD reticle
    if (activeBall) {
      const bx = (activeBall.x - cropLeft) * scaleX;
      const by = (activeBall.y - cropTop) * scaleY;
      const bw = activeBall.w * scaleX;
      const bh = activeBall.h * scaleY;
      const radius = Math.max(16, (bw + bh) / 4);

      // Neon glowing lock circular dial
      state.ctx.strokeStyle = activeBall.prediction ? '#ff7b00' : '#ff5500';
      state.ctx.lineWidth = 3;
      state.ctx.setLineDash(activeBall.prediction ? [4, 4] : []);
      
      // Draw outer target ring
      state.ctx.beginPath();
      state.ctx.arc(bx, by, radius + 6, 0, Math.PI * 2);
      state.ctx.stroke();

      // Crosshairs locking
      state.ctx.lineWidth = 1.5;
      state.ctx.setLineDash([]);
      state.ctx.beginPath();
      // Horizontal crosshair lines
      state.ctx.moveTo(bx - radius - 12, by); state.ctx.lineTo(bx - radius + 2, by);
      state.ctx.moveTo(bx + radius - 2, by); state.ctx.lineTo(bx + radius + 12, by);
      // Vertical crosshair lines
      state.ctx.moveTo(bx, by - radius - 12); state.ctx.lineTo(bx, by - radius + 2);
      state.ctx.moveTo(bx, by + radius - 2); state.ctx.lineTo(bx, by + radius + 12);
      state.ctx.stroke();

      // Tracking state label
      state.ctx.fillStyle = activeBall.prediction ? '#ff7b00' : '#ff3b30';
      state.ctx.font = "bold 9px 'Outfit', sans-serif";
      state.ctx.textAlign = 'center';
      state.ctx.fillText(
        activeBall.prediction ? 'PREDICTED LOCK' : 'BALL LOCKED',
        bx,
        by - radius - 18
      );
      state.ctx.textAlign = 'left'; // Reset
    }
  }

  // Continue rendering recursively
  requestAnimationFrame(processFrame);
}

// Helper to draw futuristic photo-bracket borders on active targets
function drawCornerBrackets(ctx, x, y, w, h, len, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.setLineDash([]);

  // Top Left Corner
  ctx.beginPath();
  ctx.moveTo(x + len, y); ctx.lineTo(x, y); ctx.lineTo(x, y + len);
  ctx.stroke();

  // Top Right Corner
  ctx.beginPath();
  ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
  ctx.stroke();

  // Bottom Left Corner
  ctx.beginPath();
  ctx.moveTo(x + len, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - len);
  ctx.stroke();

  // Bottom Right Corner
  ctx.beginPath();
  ctx.moveTo(x + w - len, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - len);
  ctx.stroke();
}

// 8. HIGH QUALITY ON-DEVICE VIDEO RECORDER
async function startRecording() {
  if (state.isRecording) return;
  
  state.recordedChunks = [];

  // Capture canvas output at a highly stable 30fps
  const canvasStream = state.canvasEl.captureStream(30);
  let finalStream = canvasStream;

  try {
    // Add microphone audio to record court ambiance and crowd reactions!
    const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    // Combine video from Canvas and Audio from Microphone
    finalStream = new MediaStream([
      ...canvasStream.getVideoTracks(),
      ...audioStream.getAudioTracks()
    ]);
  } catch (err) {
    console.warn('[Recorder] No microphone permission. Recording video without audio.', err);
    showToast('Recording video ONLY (microphone access denied)', 'info');
  }

  // Determine standard supported video container format
  let options = { mimeType: 'video/webm;codecs=vp8,opus' };
  if (!MediaRecorder.isTypeSupported(options.mimeType)) {
    options = { mimeType: 'video/webm' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
      options = { mimeType: 'video/mp4' }; // Fallback container
    }
  }

  try {
    state.mediaRecorder = new MediaRecorder(finalStream, options);
    
    state.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        state.recordedChunks.push(event.data);
      }
    };

    state.mediaRecorder.onstop = saveRecordedVideo;

    // Start recording, chunking data every 1 second
    state.mediaRecorder.start(1000);
    state.isRecording = true;
    state.recordStartTime = Date.now();

    // Trigger UI effects (record buttons turning into squares, glowing pulses)
    DOM.recordBtn().classList.add('recording');
    DOM.recordTimer().classList.add('visible');
    
    // Live record timer countdown
    state.recordTimerInterval = setInterval(updateRecordTimer, 1000);
    
    showToast('Recording Started', 'success');
  } catch (error) {
    console.error('[Recorder] Failed to start MediaRecorder:', error);
    showToast('Failed to start video recording', 'error');
  }
}

function updateRecordTimer() {
  const elapsed = Date.now() - state.recordStartTime;
  const totalSecs = Math.floor(elapsed / 1000);
  const mins = Math.floor(totalSecs / 60).toString().padStart(2, '0');
  const secs = (totalSecs % 60).toString().padStart(2, '0');
  DOM.recordTimer().textContent = `${mins}:${secs}`;
}

function stopRecording() {
  if (!state.isRecording) return;
  
  state.mediaRecorder.stop();
  state.isRecording = false;

  // Cleanup timers & UI recording styles
  clearInterval(state.recordTimerInterval);
  DOM.recordBtn().classList.remove('recording');
  DOM.recordTimer().classList.remove('visible');
  DOM.recordTimer().textContent = '00:00';
  
  showToast('Recording Stopped. Rendering video...', 'info');
}

// 9. RENDER RECORDER AND SAVE VIDEO FILE INSTANTLY
function saveRecordedVideo() {
  const blob = new Blob(state.recordedChunks, { type: 'video/webm' });
  const url = URL.createObjectURL(blob);
  
  // Format standard date/time filename
  const now = new Date();
  const dateStr = now.toISOString().split('T')[0];
  const timeStr = now.toTimeString().split(' ')[0].replace(/:/g, '');
  const fileName = `antigravity-track-${dateStr}-${timeStr}.webm`;

  // Trigger browser downloader
  const a = document.createElement('a');
  document.body.appendChild(a);
  a.style.display = 'none';
  a.href = url;
  a.download = fileName;
  a.click();
  
  // Revoke to prevent browser memory leaks
  setTimeout(() => {
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }, 100);

  showToast('Video saved successfully!', 'success');
}

// UI Alert system helper (Glass toast alerts)
function showToast(message, type = 'info') {
  const container = DOM.toastContainer();
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;

  container.appendChild(toast);

  // Automatically garbage-collect toast nodes after animation finishes
  setTimeout(() => {
    toast.remove();
  }, 3100); // Fits css keyframe animation timer perfectly
}
