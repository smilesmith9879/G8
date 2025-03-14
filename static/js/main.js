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
    
    // 开始定期测量延迟
    setInterval(measureLatency, 3000);
    
    // 更新诊断面板
    updateDiagnosticPanel();
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
    const receiveStartTime = Date.now();
    
    try {
        if (data && data.frame) {
            frameStats.received++;
            
            // 记录帧元数据
            if (data.count && data.size) {
                console.log(`帧 #${data.count} 接收, 大小: ${Math.round(data.size/1024)}KB`);
                frameStats.lastFrameSize = data.size;
            }
            
            // 记录帧接收时间
            frameStats.lastReceiveTime = Date.now() - receiveStartTime;
            
            // 计算自上一帧的时间差
            if (frameStats.lastFrameTime > 0) {
                const timeDiff = receiveStartTime - frameStats.lastFrameTime;
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
            frameStats.lastFrameTime = receiveStartTime;
            
            // 更新性能统计
            const now = Date.now();
            if (now - frameStats.lastStatsUpdate > 2000) { // 每2秒更新一次状态
                console.log(`视频性能: 已接收=${frameStats.received}, 已显示=${frameStats.displayed}, 平均FPS=${frameStats.avgFps}, 错误=${frameStats.errors}`);
                updateVideoStats();
                updateDiagnosticPanel();  // 更新诊断面板
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
    
    // 更新最后活动时间
    const lastActivityElement = document.getElementById('last-activity');
    if (lastActivityElement) {
        const now = new Date();
        lastActivityElement.textContent = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
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
        const renderStartTime = Date.now();
        displayVideoFrame(frameData);
        frameStats.lastRenderTime = Date.now() - renderStartTime;
        
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

// 添加服务器资源监控UI创建函数
function createResourceMonitorUI() {
    // 检查是否已存在资源监控面板
    if (document.getElementById('resource-monitor')) {
        return;
    }
    
    // 创建资源监控面板
    const resourceMonitor = document.createElement('div');
    resourceMonitor.id = 'resource-monitor';
    resourceMonitor.className = 'resource-monitor';
    resourceMonitor.innerHTML = `
        <div class="resource-title">服务器资源监控</div>
        <div class="resource-content">
            <div class="resource-item">
                <div class="resource-label">CPU使用率:</div>
                <div class="resource-value" id="cpu-percent">--</div>
                <div class="resource-progress-container">
                    <div class="resource-progress" id="cpu-progress"></div>
                </div>
            </div>
            <div class="resource-item">
                <div class="resource-label">进程CPU:</div>
                <div class="resource-value" id="process-cpu-percent">--</div>
                <div class="resource-progress-container">
                    <div class="resource-progress" id="process-cpu-progress"></div>
                </div>
            </div>
            <div class="resource-item">
                <div class="resource-label">内存使用率:</div>
                <div class="resource-value" id="memory-percent">--</div>
                <div class="resource-progress-container">
                    <div class="resource-progress" id="memory-progress"></div>
                </div>
            </div>
            <div class="resource-item">
                <div class="resource-label">进程内存:</div>
                <div class="resource-value" id="process-memory">--</div>
            </div>
            <div class="resource-item">
                <div class="resource-label">磁盘使用率:</div>
                <div class="resource-value" id="disk-percent">--</div>
                <div class="resource-progress-container">
                    <div class="resource-progress" id="disk-progress"></div>
                </div>
            </div>
            <div class="resource-item">
                <div class="resource-label">网络流量:</div>
                <div class="resource-value" id="network-traffic">--</div>
            </div>
            <div class="resource-item">
                <div class="resource-label">最后更新:</div>
                <div class="resource-value" id="last-update">--</div>
            </div>
        </div>
    `;
    
    // 添加样式
    const style = document.createElement('style');
    style.textContent = `
        .resource-monitor {
            position: fixed;
            top: 10px;
            right: 10px;
            width: 300px;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            border-radius: 5px;
            padding: 10px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
        }
        
        .resource-title {
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
            font-size: 14px;
            border-bottom: 1px solid #555;
            padding-bottom: 5px;
        }
        
        .resource-content {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .resource-item {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
        }
        
        .resource-label {
            width: 100px;
            font-weight: normal;
        }
        
        .resource-value {
            width: 60px;
            text-align: right;
            font-weight: bold;
        }
        
        .resource-progress-container {
            flex: 1;
            height: 8px;
            background-color: #444;
            border-radius: 4px;
            margin-left: 10px;
            overflow: hidden;
        }
        
        .resource-progress {
            height: 100%;
            width: 0%;
            background-color: #4CAF50;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
    `;
    
    // 添加到文档
    document.head.appendChild(style);
    document.body.appendChild(resourceMonitor);
    
    console.log('Resource monitor UI created.');
}

// 更新资源监控UI函数
function updateResourceMonitorUI(data) {
    if (!document.getElementById('resource-monitor')) {
        createResourceMonitorUI();
    }
    
    // 更新CPU使用率
    const cpuPercent = document.getElementById('cpu-percent');
    const cpuProgress = document.getElementById('cpu-progress');
    if (cpuPercent && cpuProgress) {
        cpuPercent.textContent = `${data.cpu.percent}%`;
        cpuProgress.style.width = `${data.cpu.percent}%`;
        
        // 根据使用率更改颜色
        if (data.cpu.percent > 80) {
            cpuProgress.style.backgroundColor = '#FF5252';
        } else if (data.cpu.percent > 60) {
            cpuProgress.style.backgroundColor = '#FFC107';
        } else {
            cpuProgress.style.backgroundColor = '#4CAF50';
        }
    }
    
    // 更新进程CPU使用率
    const processCpuPercent = document.getElementById('process-cpu-percent');
    const processCpuProgress = document.getElementById('process-cpu-progress');
    if (processCpuPercent && processCpuProgress) {
        processCpuPercent.textContent = `${data.cpu.process_percent}%`;
        processCpuProgress.style.width = `${data.cpu.process_percent}%`;
        
        // 根据使用率更改颜色
        if (data.cpu.process_percent > 80) {
            processCpuProgress.style.backgroundColor = '#FF5252';
        } else if (data.cpu.process_percent > 60) {
            processCpuProgress.style.backgroundColor = '#FFC107';
        } else {
            processCpuProgress.style.backgroundColor = '#4CAF50';
        }
    }
    
    // 更新内存使用率
    const memoryPercent = document.getElementById('memory-percent');
    const memoryProgress = document.getElementById('memory-progress');
    if (memoryPercent && memoryProgress) {
        memoryPercent.textContent = `${data.memory.percent}%`;
        memoryProgress.style.width = `${data.memory.percent}%`;
        
        // 根据使用率更改颜色
        if (data.memory.percent > 80) {
            memoryProgress.style.backgroundColor = '#FF5252';
        } else if (data.memory.percent > 60) {
            memoryProgress.style.backgroundColor = '#FFC107';
        } else {
            memoryProgress.style.backgroundColor = '#4CAF50';
        }
    }
    
    // 更新进程内存使用
    const processMemory = document.getElementById('process-memory');
    if (processMemory) {
        processMemory.textContent = `${data.memory.process_mb.toFixed(1)} MB`;
    }
    
    // 更新磁盘使用率
    const diskPercent = document.getElementById('disk-percent');
    const diskProgress = document.getElementById('disk-progress');
    if (diskPercent && diskProgress) {
        diskPercent.textContent = `${data.disk.percent}%`;
        diskProgress.style.width = `${data.disk.percent}%`;
        
        // 根据使用率更改颜色
        if (data.disk.percent > 85) {
            diskProgress.style.backgroundColor = '#FF5252';
        } else if (data.disk.percent > 70) {
            diskProgress.style.backgroundColor = '#FFC107';
        } else {
            diskProgress.style.backgroundColor = '#4CAF50';
        }
    }
    
    // 更新网络流量
    const networkTraffic = document.getElementById('network-traffic');
    if (networkTraffic) {
        networkTraffic.textContent = `↑${data.network.sent_mb.toFixed(1)} | ↓${data.network.recv_mb.toFixed(1)} MB`;
    }
    
    // 更新最后更新时间
    const lastUpdate = document.getElementById('last-update');
    if (lastUpdate) {
        const now = new Date();
        lastUpdate.textContent = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
    }
}

// 添加Socket.IO事件处理程序接收资源更新
socket.on('resource_update', (data) => {
    console.log('Received resource update:', data);
    updateResourceMonitorUI(data);
});

// 在文档加载完成后初始化资源监控UI
document.addEventListener('DOMContentLoaded', function() {
    // 初始化其他UI...
    
    // 创建资源监控UI
    createResourceMonitorUI();
});

// 添加诊断面板相关函数
// 扩展frameStats对象添加更多性能指标
frameStats.lastFrameSize = 0;
frameStats.lastReceiveTime = 0;
frameStats.lastRenderTime = 0;
frameStats.connectionIssues = 0;

// 诊断面板创建函数
function createDiagnosticPanel() {
    // 检查是否已存在诊断面板
    if (document.getElementById('diagnostic-panel')) {
        return;
    }
    
    // 创建诊断面板
    const diagnosticPanel = document.createElement('div');
    diagnosticPanel.id = 'diagnostic-panel';
    diagnosticPanel.className = 'diagnostic-panel';
    diagnosticPanel.innerHTML = `
        <div class="diagnostic-title">视频流诊断面板</div>
        <div class="diagnostic-content">
            <div class="diagnostic-section">
                <div class="section-title">视频流性能</div>
                <div class="diagnostic-item">
                    <div class="diagnostic-label">帧接收率:</div>
                    <div class="diagnostic-value" id="frame-rate">0 FPS</div>
                </div>
                <div class="diagnostic-item">
                    <div class="diagnostic-label">帧缓冲:</div>
                    <div class="diagnostic-value" id="frame-buffer">0/3</div>
                </div>
                <div class="diagnostic-item">
                    <div class="diagnostic-label">帧大小:</div>
                    <div class="diagnostic-value" id="frame-size">0 KB</div>
                </div>
                <div class="diagnostic-item">
                    <div class="diagnostic-label">接收耗时:</div>
                    <div class="diagnostic-value" id="receive-time">0 ms</div>
                </div>
                <div class="diagnostic-item">
                    <div class="diagnostic-label">渲染耗时:</div>
                    <div class="diagnostic-value" id="render-time">0 ms</div>
                </div>
                <div class="diagnostic-item">
                    <div class="diagnostic-label">总帧数:</div>
                    <div class="diagnostic-value" id="total-frames">0</div>
                </div>
                <div class="diagnostic-item">
                    <div class="diagnostic-label">解码错误:</div>
                    <div class="diagnostic-value" id="decode-errors">0</div>
                </div>
            </div>
            <div class="diagnostic-section">
                <div class="section-title">网络状态</div>
                <div class="diagnostic-item">
                    <div class="diagnostic-label">连接状态:</div>
                    <div class="diagnostic-value" id="connection-status">连接中</div>
                </div>
                <div class="diagnostic-item">
                    <div class="diagnostic-label">延迟:</div>
                    <div class="diagnostic-value" id="latency">-- ms</div>
                </div>
                <div class="diagnostic-item">
                    <div class="diagnostic-label">最后活动:</div>
                    <div class="diagnostic-value" id="last-activity">--:--:--</div>
                </div>
            </div>
        </div>
        <div class="diagnostic-footer">
            <button id="toggle-diagnostic" class="diagnostic-button">隐藏诊断</button>
            <button id="clear-diagnostic" class="diagnostic-button">重置统计</button>
        </div>
    `;
    
    // 添加样式
    const style = document.createElement('style');
    style.textContent = `
        .diagnostic-panel {
            position: fixed;
            bottom: 10px;
            left: 10px;
            width: 280px;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            border-radius: 5px;
            padding: 10px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
            transition: transform 0.3s ease;
        }
        
        .diagnostic-panel.collapsed {
            transform: translateY(calc(100% - 30px));
        }
        
        .diagnostic-title {
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
            font-size: 14px;
            border-bottom: 1px solid #555;
            padding-bottom: 5px;
            cursor: pointer;
        }
        
        .diagnostic-content {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 10px;
        }
        
        .diagnostic-section {
            border: 1px solid #555;
            border-radius: 4px;
            padding: 8px;
        }
        
        .section-title {
            font-weight: bold;
            margin-bottom: 8px;
            color: #4CAF50;
        }
        
        .diagnostic-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        
        .diagnostic-label {
            font-weight: normal;
        }
        
        .diagnostic-value {
            font-weight: bold;
        }
        
        .diagnostic-footer {
            display: flex;
            justify-content: space-between;
        }
        
        .diagnostic-button {
            background-color: #333;
            color: white;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 11px;
        }
        
        .diagnostic-button:hover {
            background-color: #444;
        }
    `;
    
    // 添加到文档
    document.head.appendChild(style);
    document.body.appendChild(diagnosticPanel);
    
    // 添加折叠功能
    const title = diagnosticPanel.querySelector('.diagnostic-title');
    title.addEventListener('click', function() {
        diagnosticPanel.classList.toggle('collapsed');
        
        // 更新按钮文本
        const toggleButton = document.getElementById('toggle-diagnostic');
        if (toggleButton) {
            if (diagnosticPanel.classList.contains('collapsed')) {
                toggleButton.textContent = '显示诊断';
            } else {
                toggleButton.textContent = '隐藏诊断';
            }
        }
    });
    
    // 添加按钮功能
    const toggleButton = document.getElementById('toggle-diagnostic');
    if (toggleButton) {
        toggleButton.addEventListener('click', function() {
            diagnosticPanel.classList.toggle('collapsed');
            this.textContent = diagnosticPanel.classList.contains('collapsed') ? '显示诊断' : '隐藏诊断';
        });
    }
    
    // 重置统计按钮
    const clearButton = document.getElementById('clear-diagnostic');
    if (clearButton) {
        clearButton.addEventListener('click', function() {
            resetDiagnosticStats();
        });
    }
    
    console.log('Diagnostic panel created');
}

// 重置诊断统计数据
function resetDiagnosticStats() {
    frameStats.received = 0;
    frameStats.displayed = 0;
    frameStats.errors = 0;
    frameStats.frameTimes = [];
    frameStats.avgFps = 0;
    frameStats.bufferSize = 0;
    frameStats.lastFrameTime = 0;
    frameStats.lastStatsUpdate = Date.now();
    frameStats.lastFrameSize = 0;
    frameStats.lastReceiveTime = 0;
    frameStats.lastRenderTime = 0;
    frameStats.connectionIssues = 0;
    
    // 更新UI
    updateDiagnosticPanel();
    
    console.log('Diagnostic statistics reset');
}

// 测量网络延迟
let lastPingTime = 0;
let currentLatency = 0;

function measureLatency() {
    lastPingTime = Date.now();
    socket.emit('ping_request');
}

// 更新诊断面板的值
function updateDiagnosticPanel() {
    // 检查面板是否已创建
    if (!document.getElementById('diagnostic-panel')) {
        createDiagnosticPanel();
    }
    
    // 更新帧率
    const frameRateElement = document.getElementById('frame-rate');
    if (frameRateElement) {
        frameRateElement.textContent = `${frameStats.avgFps.toFixed(1)} FPS`;
        
        // 根据帧率设置颜色
        if (frameStats.avgFps < 5) {
            frameRateElement.style.color = '#FF5252';
        } else if (frameStats.avgFps < 10) {
            frameRateElement.style.color = '#FFC107';
        } else {
            frameRateElement.style.color = '#4CAF50';
        }
    }
    
    // 更新帧缓冲
    const frameBufferElement = document.getElementById('frame-buffer');
    if (frameBufferElement) {
        frameBufferElement.textContent = `${frameStats.bufferSize}/${frameBuffer.maxSize}`;
    }
    
    // 更新帧大小
    const frameSizeElement = document.getElementById('frame-size');
    if (frameSizeElement && frameStats.lastFrameSize) {
        frameSizeElement.textContent = `${(frameStats.lastFrameSize / 1024).toFixed(1)} KB`;
    }
    
    // 更新接收耗时
    const receiveTimeElement = document.getElementById('receive-time');
    if (receiveTimeElement && frameStats.lastReceiveTime) {
        receiveTimeElement.textContent = `${frameStats.lastReceiveTime.toFixed(1)} ms`;
    }
    
    // 更新渲染耗时
    const renderTimeElement = document.getElementById('render-time');
    if (renderTimeElement && frameStats.lastRenderTime) {
        renderTimeElement.textContent = `${frameStats.lastRenderTime.toFixed(1)} ms`;
    }
    
    // 更新总帧数
    const totalFramesElement = document.getElementById('total-frames');
    if (totalFramesElement) {
        totalFramesElement.textContent = frameStats.received.toString();
    }
    
    // 更新解码错误
    const decodeErrorsElement = document.getElementById('decode-errors');
    if (decodeErrorsElement) {
        decodeErrorsElement.textContent = frameStats.errors.toString();
        
        // 根据错误数设置颜色
        if (frameStats.errors > 10) {
            decodeErrorsElement.style.color = '#FF5252';
        } else if (frameStats.errors > 0) {
            decodeErrorsElement.style.color = '#FFC107';
        } else {
            decodeErrorsElement.style.color = '#4CAF50';
        }
    }
    
    // 更新连接状态
    const connectionStatusElement = document.getElementById('connection-status');
    if (connectionStatusElement) {
        if (socket.connected) {
            connectionStatusElement.textContent = '已连接';
            connectionStatusElement.style.color = '#4CAF50';
        } else {
            connectionStatusElement.textContent = '断开连接';
            connectionStatusElement.style.color = '#FF5252';
        }
    }
    
    // 更新延迟
    const latencyElement = document.getElementById('latency');
    if (latencyElement) {
        latencyElement.textContent = `${currentLatency} ms`;
        
        // 根据延迟设置颜色
        if (currentLatency > 200) {
            latencyElement.style.color = '#FF5252';
        } else if (currentLatency > 100) {
            latencyElement.style.color = '#FFC107';
        } else if (currentLatency > 0) {
            latencyElement.style.color = '#4CAF50';
        }
    }
    
    // 更新最后活动时间
    const lastActivityElement = document.getElementById('last-activity');
    if (lastActivityElement) {
        const now = new Date();
        lastActivityElement.textContent = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
    }
}

// 添加延迟测量响应处理器
socket.on('ping_response', () => {
    const pingTime = Date.now() - lastPingTime;
    currentLatency = pingTime;
    
    // 更新诊断面板中的延迟
    const latencyElement = document.getElementById('latency');
    if (latencyElement) {
        latencyElement.textContent = `${currentLatency} ms`;
        
        // 根据延迟设置颜色
        if (currentLatency > 200) {
            latencyElement.style.color = '#FF5252';
        } else if (currentLatency > 100) {
            latencyElement.style.color = '#FFC107';
        } else {
            latencyElement.style.color = '#4CAF50';
        }
    }
}); 