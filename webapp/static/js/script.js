const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const ctx = canvas.getContext('2d');

// UI Elements
const charDisplay = document.getElementById('current-char');
const confBar = document.getElementById('conf-bar');
const confText = document.getElementById('conf-text');
const sentenceOut = document.getElementById('sentence-output');
const toggleBtn = document.getElementById('toggleCam');
const scanLine = document.querySelector('.scan-line'); // Added selector for scan line

let isStreaming = false;
let lastPrediction = "";
let holdCount = 0;
const HOLD_THRESHOLD = 5; // Frames to hold before typing

// 1. Camera Toggle
toggleBtn.addEventListener('click', async () => {
    if (!isStreaming) {
        try {
            // Request camera
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            video.srcObject = stream;
            isStreaming = true;
            
            // Visual Update
            toggleBtn.innerText = "TERMINATE CAM";
            toggleBtn.style.borderColor = "var(--neon-pink)";
            toggleBtn.style.color = "var(--neon-pink)";
            scanLine.classList.add('active'); // Show scan line animation
            
            // Start Loop
            requestAnimationFrame(sendFrame); 
        } catch (err) {
            console.error("Camera Error:", err);
            alert("Could not access camera. Please allow permissions.");
        }
    } else {
        // Stop Camera
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
        isStreaming = false;
        
        // Visual Update
        toggleBtn.innerText = "INITIALIZE CAM";
        toggleBtn.style.borderColor = "var(--neon-blue)";
        toggleBtn.style.color = "var(--neon-blue)";
        scanLine.classList.remove('active'); // Hide scan line animation
    }
});

// 2. Main Loop: Send Frame to Python
async function sendFrame() {
    if (!isStreaming) return;

    // Draw video to hidden canvas
    if (video.videoWidth === 0 || video.videoHeight === 0) {
        requestAnimationFrame(sendFrame);
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Convert to base64
    const dataURL = canvas.toDataURL('image/jpeg', 0.5);

    try {
        const response = await fetch('/predict_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL })
        });
        
        const data = await response.json();
        
        // Check for valid prediction
        if(data.prediction) {
            // FIX: Use 'confidence' not 'confidence_val'
            updateUI(data.prediction, data.confidence);
        }

    } catch (error) {
        console.error("Prediction error:", error);
    }

    // Throttle loop to ~10 FPS
    setTimeout(() => requestAnimationFrame(sendFrame), 100);
}

// 3. UI Updates
function updateUI(prediction, confidence) {
    charDisplay.innerText = prediction;
    
    // Handle confidence bar
    // Handle case where confidence might be null/undefined
    const confVal = confidence ? confidence : 0;
    const confPercent = Math.round(confVal * 100);
    
    confBar.style.width = `${confPercent}%`;
    confText.innerText = `${confPercent}%`;

    // Typewriter Logic
    if (prediction === lastPrediction && prediction !== "--" && prediction !== "ERR") {
        holdCount++;
        
        if (holdCount === HOLD_THRESHOLD) {
            if (prediction === "SPACE") {
                sentenceOut.innerText += " ";
            } else if (prediction === "DEL") {
                sentenceOut.innerText = sentenceOut.innerText.slice(0, -1);
            } else {
                sentenceOut.innerText += prediction;
            }
            
            // Flash Effect
            charDisplay.style.textShadow = "0 0 30px #fff";
            setTimeout(() => charDisplay.style.textShadow = "0 0 20px var(--neon-blue)", 200);
        }
    } else {
        holdCount = 0;
        lastPrediction = prediction;
    }
}

// 4. Utility Buttons
document.getElementById('clearBtn').addEventListener('click', () => {
    sentenceOut.innerText = "";
});

document.getElementById('speakBtn').addEventListener('click', () => {
    const text = sentenceOut.innerText;
    if(text) {
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    }
});