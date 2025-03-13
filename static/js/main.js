// Initialize Socket.IO connection
const socket = io();

// DOM Elements
const videoCanvas = document.getElementById('video-canvas');
const videoPlaceholder = document.querySelector('.video-placeholder');
const startStreamBtn = document.getElementById('start-stream');
const stopStreamBtn = document.getElementById('stop-stream');
const carXDisplay = document.getElementById('car-x');
const carYDisplay = document.getElementById('car-y');
const carSpeedDisplay = document.getElementById('car-speed');
const cameraHDisplay = document.getElementById('camera-h');
const cameraVDisplay = document.getElementById('camera-v');
const mapPlaceholder = document.getElementById('map-placeholder');
const mapDisplay = document.getElementById('map-display');
const mapImage = document.getElementById('map-image');
const startSlamBtn = document.getElementById('start-slam');
const stopSlamBtn = document.getElementById('stop-slam');
const posXDisplay = document.getElementById('pos-x');
const posYDisplay = document.getElementById('pos-y');
const posZDisplay = document.getElementById('pos-z');

// Canvas context
const ctx = videoCanvas.getContext('2d');

// Global variables
let isStreaming = false;
let isSlamActive = false;
let carJoystick = null;
let cameraJoystick = null;
let carControlInterval = null;
let cameraControlInterval = null;
let carJoystickData = { x: 0, y: 0 };
let cameraJoystickData = { x: 0, y: 0 };

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    stopStream();
    if (isSlamActive) {
        stopSlam();
    }
});

socket.on('status_update', (data) => {
    updateStatusIndicators(data);
    
    // Update SLAM button state
    startSlamBtn.disabled = !data.slam_available;
    
    // Update SLAM status
    if (data.slam_active) {
        isSlamActive = true;
        startSlamBtn.disabled = true;
        stopSlamBtn.disabled = false;
        mapPlaceholder.style.display = 'none';
        mapDisplay.style.display = 'block';
    }
});

socket.on('video_frame', (data) => {
    if (isStreaming) {
        displayVideoFrame(data.frame);
    }
});

socket.on('stream_status', (data) => {
    if (data.status === 'started') {
        isStreaming = true;
        videoPlaceholder.style.display = 'none';
        videoCanvas.style.display = 'block';
        startStreamBtn.disabled = true;
        stopStreamBtn.disabled = false;
    } else if (data.status === 'stopped') {
        isStreaming = false;
        videoPlaceholder.style.display = 'flex';
        videoCanvas.style.display = 'none';
        startStreamBtn.disabled = false;
        stopStreamBtn.disabled = true;
    } else if (data.status === 'error') {
        alert(`Stream error: ${data.message}`);
        isStreaming = false;
        startStreamBtn.disabled = false;
        stopStreamBtn.disabled = true;
    }
});

socket.on('slam_status', (data) => {
    if (data.status === 'started') {
        isSlamActive = true;
        startSlamBtn.disabled = true;
        stopSlamBtn.disabled = false;
        mapPlaceholder.style.display = 'none';
        mapDisplay.style.display = 'block';
    } else if (data.status === 'stopped') {
        isSlamActive = false;
        startSlamBtn.disabled = false;
        stopSlamBtn.disabled = true;
        mapPlaceholder.style.display = 'flex';
        mapDisplay.style.display = 'none';
    } else if (data.status === 'error') {
        alert(`SLAM error: ${data.message}`);
        isSlamActive = false;
        startSlamBtn.disabled = false;
        stopSlamBtn.disabled = true;
    }
});

socket.on('slam_map', (data) => {
    if (isSlamActive) {
        // Update map image
        mapImage.src = 'data:image/png;base64,' + data.map;
    }
});

socket.on('slam_position', (data) => {
    if (isSlamActive) {
        // Update position display
        posXDisplay.textContent = data.position[0].toFixed(2);
        posYDisplay.textContent = data.position[1].toFixed(2);
        posZDisplay.textContent = data.position[2].toFixed(2);
    }
});

socket.on('control_response', (data) => {
    if (data.status === 'success') {
        carSpeedDisplay.textContent = data.speed;
    } else if (data.status === 'error') {
        console.error(`Control error: ${data.message}`);
    }
});

socket.on('gimbal_response', (data) => {
    if (data.status === 'success') {
        cameraHDisplay.textContent = `${data.h_angle}°`;
        cameraVDisplay.textContent = `${data.v_angle}°`;
    } else if (data.status === 'error') {
        console.error(`Gimbal error: ${data.message}`);
    }
});

// Function to update status indicators
function updateStatusIndicators(data) {
    const robotStatus = document.getElementById('robot-status');
    const cameraStatus = document.getElementById('camera-status');
    const mpuStatus = document.getElementById('mpu-status');
    const slamStatus = document.getElementById('slam-status');
    
    if (data.robot_available) {
        robotStatus.innerHTML = '<i class="fas fa-check-circle status-ok"></i> Connected';
    } else {
        robotStatus.innerHTML = '<i class="fas fa-times-circle status-error"></i> Disconnected';
    }
    
    if (data.camera_available) {
        cameraStatus.innerHTML = '<i class="fas fa-check-circle status-ok"></i> Connected';
    } else {
        cameraStatus.innerHTML = '<i class="fas fa-times-circle status-error"></i> Disconnected';
    }
    
    if (data.mpu6050_available) {
        mpuStatus.innerHTML = '<i class="fas fa-check-circle status-ok"></i> Connected';
    } else {
        mpuStatus.innerHTML = '<i class="fas fa-times-circle status-error"></i> Disconnected';
    }
    
    if (data.slam_available) {
        slamStatus.innerHTML = '<i class="fas fa-check-circle status-ok"></i> Available';
        if (data.slam_active) {
            slamStatus.innerHTML = '<i class="fas fa-check-circle status-ok"></i> Active';
        }
    } else {
        slamStatus.innerHTML = '<i class="fas fa-times-circle status-error"></i> Unavailable';
    }
}

// Function to display video frame
function displayVideoFrame(frameData) {
    const img = new Image();
    img.onload = () => {
        // Clear canvas
        ctx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
        
        // Draw image
        videoCanvas.width = img.width;
        videoCanvas.height = img.height;
        ctx.drawImage(img, 0, 0, videoCanvas.width, videoCanvas.height);
    };
    
    // Use base64 encoded frame directly
    img.src = 'data:image/jpeg;base64,' + frameData;
}

// Initialize joysticks
function initJoysticks() {
    // Car control joystick (left)
    const carJoystickOptions = {
        zone: document.getElementById('car-joystick'),
        mode: 'static',
        position: { left: '50%', top: '50%' },
        size: 120,
        color: '#3498db',
        lockX: false,
        lockY: false
    };
    
    carJoystick = nipplejs.create(carJoystickOptions);
    
    carJoystick.on('move', (evt, data) => {
        const x = parseFloat((-data.vector.x).toFixed(2));
        const y = parseFloat((data.vector.y).toFixed(2));
        
        carJoystickData = { x, y };
        
        // Update display
        carXDisplay.textContent = x;
        carYDisplay.textContent = y;
    });
    
    carJoystick.on('end', () => {
        // Auto-centering: reset to zero when released
        carJoystickData = { x: 0, y: 0 };
        
        // Update display
        carXDisplay.textContent = '0';
        carYDisplay.textContent = '0';
        
        // Send stop command immediately
        socket.emit('car_control', { x: 0, y: 0 });
    });
    
    // Camera control joystick (right)
    const cameraJoystickOptions = {
        zone: document.getElementById('camera-joystick'),
        mode: 'static',
        position: { left: '50%', top: '50%' },
        size: 120,
        color: '#e74c3c',
        lockX: false,
        lockY: false
    };
    
    cameraJoystick = nipplejs.create(cameraJoystickOptions);
    
    cameraJoystick.on('move', (evt, data) => {
        const x = parseFloat((-data.vector.x).toFixed(2));
        const y = parseFloat((data.vector.y).toFixed(2));
        
        cameraJoystickData = { x, y };
    });
    
    cameraJoystick.on('end', () => {
        // Auto-centering: reset to zero when released
        cameraJoystickData = { x: 0, y: 0 };
        
        // Send reset command immediately
        socket.emit('gimbal_control', { x: 0, y: 0 });
    });
}

// Start control intervals
function startControlIntervals() {
    // Car control interval (100ms)
    carControlInterval = setInterval(() => {
        if (carJoystickData.x !== 0 || carJoystickData.y !== 0) {
            socket.emit('car_control', carJoystickData);
        }
    }, 100);
    
    // Camera control interval (200ms)
    cameraControlInterval = setInterval(() => {
        if (cameraJoystickData.x !== 0 || cameraJoystickData.y !== 0) {
            socket.emit('gimbal_control', cameraJoystickData);
        }
    }, 200);
}

// Stop control intervals
function stopControlIntervals() {
    clearInterval(carControlInterval);
    clearInterval(cameraControlInterval);
}

// Start video stream
function startStream() {
    socket.emit('start_stream');
}

// Stop video stream
function stopStream() {
    socket.emit('stop_stream');
    isStreaming = false;
    videoPlaceholder.style.display = 'flex';
    videoCanvas.style.display = 'none';
    startStreamBtn.disabled = false;
    stopStreamBtn.disabled = true;
}

// Start SLAM
function startSlam() {
    socket.emit('start_slam');
}

// Stop SLAM
function stopSlam() {
    socket.emit('stop_slam');
    isSlamActive = false;
    mapPlaceholder.style.display = 'flex';
    mapDisplay.style.display = 'none';
    startSlamBtn.disabled = false;
    stopSlamBtn.disabled = true;
}

// Event listeners
startStreamBtn.addEventListener('click', startStream);
stopStreamBtn.addEventListener('click', stopStream);
startSlamBtn.addEventListener('click', startSlam);
stopSlamBtn.addEventListener('click', stopSlam);

// Helper function to convert ArrayBuffer to Base64 (不需要此函数，直接使用base64)
// 保留此函数以兼容可能的历史代码，但实际上不再使用它
function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    
    return window.btoa(binary);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Initialize joysticks
    initJoysticks();
    
    // Start control intervals
    startControlIntervals();
    
    // Request status update
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            updateStatusIndicators(data);
            
            // Update SLAM button state
            startSlamBtn.disabled = !data.slam_available;
            
            // Update SLAM status
            if (data.slam_active) {
                isSlamActive = true;
                startSlamBtn.disabled = true;
                stopSlamBtn.disabled = false;
                mapPlaceholder.style.display = 'none';
                mapDisplay.style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Error fetching status:', error);
        });
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, stop controls
        socket.emit('car_control', { x: 0, y: 0 });
        stopControlIntervals();
    } else {
        // Page is visible again, restart control intervals
        startControlIntervals();
    }
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    // Stop the car when leaving the page
    socket.emit('car_control', { x: 0, y: 0 });
}); 