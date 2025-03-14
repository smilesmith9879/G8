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
// MPU6050 sensor data displays
const accelXDisplay = document.getElementById('accel-x');
const accelYDisplay = document.getElementById('accel-y');
const accelZDisplay = document.getElementById('accel-z');
const gyroXDisplay = document.getElementById('gyro-x');
const gyroYDisplay = document.getElementById('gyro-y');
const gyroZDisplay = document.getElementById('gyro-z');
const temperatureDisplay = document.getElementById('temperature');

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

// 添加性能监控变量
let frameStats = {
    received: 0,
    displayed: 0,
    errors: 0,
    lastFrameTime: 0,
    frameTimes: [],  // 存储最近10帧的处理时间
    avgFps: 0,
    bufferSize: 0,
    lastStatsUpdate: Date.now()
};

// 用于处理视频流的帧缓冲区
const frameBuffer = {
    maxSize: 3,  // 最多缓存3帧
    frames: [],
    add: function(frame) {
        if (this.frames.length >= this.maxSize) {
            this.frames.shift(); // 移除最旧的帧
        }
        this.frames.push(frame);
        frameStats.bufferSize = this.frames.length;
    },
    getNext: function() {
        if (this.frames.length > 0) {
            return this.frames.shift();
        }
        return null;
    },
    clear: function() {
        this.frames = [];
        frameStats.bufferSize = 0;
    }
};

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to server');
    console.log('Waiting for server to start video stream...');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    isStreaming = false;
    if (isSlamActive) {
        isSlamActive = false;
    }
    console.log('Connection lost - stream states reset');
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
    const receiveTime = Date.now();
    try {
        if (data && data.frame) {
            frameStats.received++;
            
            // 记录帧元数据
            if (data.count && data.size) {
                console.log(`帧 #${data.count} 接收, 大小: ${Math.round(data.size/1024)}KB`);
            }
            
            // 计算自上一帧的时间差
            if (frameStats.lastFrameTime > 0) {
                const timeDiff = receiveTime - frameStats.lastFrameTime;
                frameStats.frameTimes.push(timeDiff);
                // 只保留最近10个时间差
                if (frameStats.frameTimes.length > 10) {
                    frameStats.frameTimes.shift();
                }
                
                // 计算平均FPS
                if (frameStats.frameTimes.length > 0) {
                    const avgTime = frameStats.frameTimes.reduce((a, b) => a + b, 0) / frameStats.frameTimes.length;
                    frameStats.avgFps = Math.round(1000 / avgTime * 10) / 10;
                }
            }
            frameStats.lastFrameTime = receiveTime;
            
            // 更新性能统计
            const now = Date.now();
            if (now - frameStats.lastStatsUpdate > 2000) { // 每2秒更新一次状态
                console.log(`视频性能: 已接收=${frameStats.received}, 已显示=${frameStats.displayed}, 平均FPS=${frameStats.avgFps}, 错误=${frameStats.errors}`);
                updateVideoStats();
                frameStats.lastStatsUpdate = now;
            }
            
            // 将帧添加到缓冲区
            frameBuffer.add(data.frame);
            
            // 如果这是第一帧，立即处理
            if (frameStats.received === 1 || frameBuffer.frames.length === 1) {
                processNextFrame();
            }
        } else {
            console.error('收到无效的视频帧数据');
            frameStats.errors++;
        }
    } catch (error) {
        console.error('处理视频帧时出错:', error);
        frameStats.errors++;
    }
});

socket.on('stream_status', (data) => {
    console.log('Received stream_status:', data.status);
    
    if (data.status === 'started') {
        console.log('Video streaming started successfully');
        isStreaming = true;
        videoPlaceholder.style.display = 'none';
        videoCanvas.style.display = 'block';
    } else if (data.status === 'stopped') {
        console.log('Video streaming stopped by server');
        isStreaming = false;
        videoPlaceholder.style.display = 'flex';
        videoCanvas.style.display = 'none';
    } else if (data.status === 'error') {
        console.error(`Stream error: ${data.message}`);
        alert(`Stream error: ${data.message}`);
        isStreaming = false;
        videoPlaceholder.style.display = 'flex';
        videoCanvas.style.display = 'none';
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

socket.on('mpu6050_data', (data) => {
    // Update MPU6050 data displays
    if (accelXDisplay) accelXDisplay.textContent = data.accel_x;
    if (accelYDisplay) accelYDisplay.textContent = data.accel_y;
    if (accelZDisplay) accelZDisplay.textContent = data.accel_z;
    if (gyroXDisplay) gyroXDisplay.textContent = data.gyro_x;
    if (gyroYDisplay) gyroYDisplay.textContent = data.gyro_y;
    if (gyroZDisplay) gyroZDisplay.textContent = data.gyro_z;
    if (temperatureDisplay) temperatureDisplay.textContent = data.temperature;
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

// 创建一个处理帧的函数，可以独立调用
function processNextFrame() {
    const frameData = frameBuffer.getNext();
    if (frameData) {
        displayVideoFrame(frameData);
        
        // 如果缓冲区中还有帧，安排下一个帧的处理
        if (frameBuffer.frames.length > 0) {
            // 使用requestAnimationFrame来优化渲染性能
            requestAnimationFrame(processNextFrame);
        }
    }
}

// 添加视频状态更新到UI
function updateVideoStats() {
    // 如果有视频状态显示元素，则更新它
    const statsElement = document.getElementById('video-stats');
    if (statsElement) {
        statsElement.innerHTML = `
            <div>FPS: ${frameStats.avgFps}</div>
            <div>缓冲: ${frameStats.bufferSize}/${frameBuffer.maxSize}</div>
            <div>已接收: ${frameStats.received}</div>
        `;
    }
}

// Function to display video frame
function displayVideoFrame(frameData) {
    if (!frameData) {
        console.error("收到空帧数据");
        frameStats.errors++;
        return;
    }
    
    const processStart = Date.now();
    
    // 更新显示状态，但不触发任何事件
    if (!isStreaming) {
        console.log("收到第一帧，更新UI");
        isStreaming = true;
        if (videoPlaceholder && videoPlaceholder.style) {
            videoPlaceholder.style.display = 'none';
        }
        if (videoCanvas && videoCanvas.style) {
            videoCanvas.style.display = 'block';
        }
        
        // 添加视频状态显示元素
        if (!document.getElementById('video-stats')) {
            const statsDiv = document.createElement('div');
            statsDiv.id = 'video-stats';
            statsDiv.style.position = 'absolute';
            statsDiv.style.top = '5px';
            statsDiv.style.right = '5px';
            statsDiv.style.backgroundColor = 'rgba(0,0,0,0.5)';
            statsDiv.style.color = 'white';
            statsDiv.style.padding = '5px';
            statsDiv.style.fontSize = '12px';
            statsDiv.style.zIndex = '1000';
            document.body.appendChild(statsDiv);
        }
    }
    
    // 确保videoCanvas和ctx已初始化
    if (!videoCanvas || !ctx) {
        console.error("视频画布或上下文未初始化");
        frameStats.errors++;
        return;
    }
    
    // 确保canvas已初始化
    if (!videoCanvas.width || !videoCanvas.height) {
        videoCanvas.width = 320;
        videoCanvas.height = 240;
        console.log("画布初始化为320x240");
    }
    
    try {
        const img = new Image();
        
        // 图像加载错误处理
        img.onerror = (err) => {
            console.error("无法从base64数据加载图像:", err);
            frameStats.errors++;
        };
        
        img.onload = () => {
            try {
                // 清除画布
                ctx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
                
                // 绘制图像
                ctx.drawImage(img, 0, 0, videoCanvas.width, videoCanvas.height);
                
                // 记录显示成功
                frameStats.displayed++;
                
                // 绘制性能信息到画布上
                ctx.fillStyle = 'rgba(0,0,0,0.5)';
                ctx.fillRect(5, 5, 100, 20);
                ctx.fillStyle = 'white';
                ctx.font = '12px Arial';
                ctx.fillText(`FPS: ${frameStats.avgFps}`, 10, 20);
                
                // 计算处理时间
                const processTime = Date.now() - processStart;
                if (processTime > 100) {
                    console.warn(`帧处理耗时较长: ${processTime}ms`);
                }
                
            } catch (drawError) {
                console.error("绘制图像到画布时出错:", drawError);
                frameStats.errors++;
            }
        };
        
        // 使用base64编码帧直接与明确的MIME类型
        const src = 'data:image/jpeg;base64,' + frameData;
        img.src = src;
        
        // 调试帮助 - 只在帧数据异常小时警告
        if (frameData.length < 1000) {
            console.warn("接收到可疑的小帧数据:", frameData.length, "字节");
        }
    } catch (e) {
        console.error("显示视频帧时出错:", e);
        frameStats.errors++;
    }
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

// Stop video stream - 只在用户主动点击停止按钮时调用
function stopStream() {
    console.log('User requested to stop stream');
    socket.emit('stop_stream');
    // 状态更新会在接收到服务器的回复后处理
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
    console.log('Page loaded, initializing interface');
    
    // 完全隐藏视频控制按钮，因为我们现在自动处理视频流
    const videoControls = document.querySelector('.video-controls');
    if (videoControls) {
        videoControls.style.display = 'none';
    }
    
    // 移除不必要的事件监听器，防止意外触发
    if (stopStreamBtn) {
        stopStreamBtn.removeEventListener('click', stopStream);
    }
    
    // 如果视频控制按钮仍然可见，更新它们的状态
    if (startStreamBtn) {
        startStreamBtn.disabled = true;
        startStreamBtn.textContent = 'Stream Auto-Started';
    }
    if (stopStreamBtn) {
        stopStreamBtn.disabled = true;
    }
    
    // Initialize joysticks
    initJoysticks();
    
    // Start control intervals
    startControlIntervals();
    
    // Request status update
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            console.log('Received status from server:', data);
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
    console.log('Page visibility changed:', document.hidden ? 'hidden' : 'visible');
    
    if (document.hidden) {
        // Page is hidden, stop controls but don't stop video
        if (carJoystickData.x !== 0 || carJoystickData.y !== 0) {
            socket.emit('car_control', { x: 0, y: 0 });
            console.log('Car stopped due to page hidden');
        }
        stopControlIntervals();
    } else {
        // Page is visible again, restart control intervals
        startControlIntervals();
    }
});

// Handle page unload - only stop the car, don't stop the stream
window.addEventListener('beforeunload', () => {
    // Stop the car when leaving the page
    socket.emit('car_control', { x: 0, y: 0 });
    console.log('Car stopped due to page unload');
    // No need to stop the stream here, the socket disconnect event will handle cleanup
}); 