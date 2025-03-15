// Initialize Socket.IO connection
const socket = io({
    transports: ['websocket', 'polling'], // 首选websocket，但支持fallback到polling
    reconnectionAttempts: 10,            // 增加重连尝试次数到10次
    reconnectionDelay: 1000,             // 初始重连延迟1秒
    reconnectionDelayMax: 10000,         // 最大重连延迟10秒
    timeout: 20000,                      // 连接超时20秒
    pingTimeout: 60000,                  // 匹配服务器的ping超时配置
    pingInterval: 25000                  // 匹配服务器的ping间隔配置
});

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
const videoFeed = document.getElementById('video-feed');
const toggleMapViewBtn = document.getElementById('toggle-map-view');
const map3dDisplay = document.getElementById('map3d-display');

// Canvas context
const ctx = videoCanvas.getContext('2d');

// Global variables
let isStreaming = false;
let isSlamActive = false;
let is3dViewActive = false;
let carJoystick;
let cameraJoystick;
let carInterval;
let cameraInterval;
let carData = { x: 0, y: 0, client_type: 'web' };
let cameraData = { x: 0, y: 0, client_type: 'web' };
let currentLatency = 0;
let pingStart = 0;
let frameBuffer = {
    frames: [],
    maxSize: 3,
    add: function(frame) {
        if (this.frames.length >= this.maxSize) {
            this.frames.shift();
        }
        this.frames.push(frame);
    },
    get: function() {
        return this.frames.shift();
    },
    isEmpty: function() {
        return this.frames.length === 0;
    },
    size: function() {
        return this.frames.length;
    }
};

// 添加性能监控变量
let frameStats = {
    received: 0,
    displayed: 0,
    errors: 0,
    fps: 0,
    avgFps: 0,
    fpsHistory: [],
    lastFrameTime: 0,
    bufferSize: 0
};

// 服务器资源信息
let serverResources = {
    cpu: {
        percent: 0,
        process_percent: 0
    },
    memory: {
        percent: 0,
        used_mb: 0,
        total_mb: 0,
        process_mb: 0
    },
    timestamp: 0
};

// Three.js variables
let scene, camera, renderer, controls, pointCloud;

// Initialize Three.js scene
function initThreeJs() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    // Create camera
    camera = new THREE.PerspectiveCamera(75, map3dDisplay.clientWidth / map3dDisplay.clientHeight, 0.1, 1000);
    camera.position.set(5, 5, 5);
    camera.lookAt(0, 0, 0);
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(map3dDisplay.clientWidth, map3dDisplay.clientHeight);
    map3dDisplay.querySelector('#map3d-container').appendChild(renderer.domElement);
    
    // Create controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    
    // Add grid helper
    const gridHelper = new THREE.GridHelper(10, 10);
    scene.add(gridHelper);
    
    // Add axes helper
    const axesHelper = new THREE.AxesHelper(2);
    scene.add(axesHelper);
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);
    
    // Create empty point cloud
    createEmptyPointCloud();
    
    // Start animation loop
    animate();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

// Create empty point cloud
function createEmptyPointCloud() {
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true
    });
    
    // Create empty buffers
    const positions = new Float32Array(0);
    const colors = new Float32Array(0);
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    // Create point cloud
    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
}

// Update point cloud with new data
function updatePointCloud(data) {
    if (!data || !data.positions || !data.colors || data.count === 0) {
        return;
    }
    
    // Create geometry
    const geometry = new THREE.BufferGeometry();
    
    // Create position and color arrays
    const positions = new Float32Array(data.positions);
    const colors = new Float32Array(data.colors);
    
    // Set attributes
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    // Remove old point cloud
    scene.remove(pointCloud);
    
    // Create new point cloud
    const material = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true
    });
    
    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    // Update controls
    if (controls) {
        controls.update();
    }
    
    // Render scene
    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

// Handle window resize
function onWindowResize() {
    if (camera && renderer && map3dDisplay) {
        camera.aspect = map3dDisplay.clientWidth / map3dDisplay.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(map3dDisplay.clientWidth, map3dDisplay.clientHeight);
    }
}

// Toggle between 2D and 3D map view
function toggleMapView() {
    is3dViewActive = !is3dViewActive;
    
    if (is3dViewActive) {
        // Switch to 3D view
        mapDisplay.style.display = 'none';
        map3dDisplay.style.display = 'block';
        
        // Initialize Three.js if not already initialized
        if (!scene) {
            initThreeJs();
        }
        
        // Update button icon
        toggleMapViewBtn.innerHTML = '<i class="fas fa-map"></i> Toggle 2D View';
    } else {
        // Switch to 2D view
        mapDisplay.style.display = 'flex';
        map3dDisplay.style.display = 'none';
        
        // Update button icon
        toggleMapViewBtn.innerHTML = '<i class="fas fa-cube"></i> Toggle 3D View';
    }
}

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
    console.log(`收到视频帧 #${data.count}, 时间戳: ${new Date().toISOString()}`);
    
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
                frameStats.fpsHistory.push(timeDiff);
                // 只保留最近10个时间差
                if (frameStats.fpsHistory.length > 10) {
                    frameStats.fpsHistory.shift();
                }
                
                // 计算平均FPS
                if (frameStats.fpsHistory.length > 0) {
                    const avgTime = frameStats.fpsHistory.reduce((a, b) => a + b, 0) / frameStats.fpsHistory.length;
                    frameStats.avgFps = Math.round(1000 / avgTime * 10) / 10;
                }
            }
            frameStats.lastFrameTime = receiveStartTime;
            
            // 验证base64数据
            if (!validateBase64Data(data.frame)) {
                console.error(`帧 #${data.count} 的Base64数据无效`);
                frameStats.errors++;
                return;
            }
            
            // 将帧添加到缓冲区
            frameBuffer.add(data.frame);
            
            // 如果这是第一帧，立即处理
            if (frameStats.received === 1 || frameBuffer.frames.length === 1) {
                processNextFrame();
            }
            
            // 更新性能统计
            const now = Date.now();
            if (now - frameStats.lastStatsUpdate > 2000) { // 每2秒更新一次状态
                console.log(`视频性能: 已接收=${frameStats.received}, 已显示=${frameStats.displayed}, 平均FPS=${frameStats.avgFps}, 错误=${frameStats.errors}`);
                updateVideoStats();
                updateDiagnosticPanel();  // 更新诊断面板
                frameStats.lastStatsUpdate = now;
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
        
        // Show the appropriate map view
        if (is3dViewActive) {
            map3dDisplay.style.display = 'block';
        } else {
            mapDisplay.style.display = 'flex';
        }
    } else if (data.status === 'stopped') {
        isSlamActive = false;
        startSlamBtn.disabled = false;
        stopSlamBtn.disabled = true;
        mapPlaceholder.style.display = 'flex';
        mapDisplay.style.display = 'none';
        map3dDisplay.style.display = 'none';
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

socket.on('slam_map_3d', (data) => {
    if (isSlamActive && is3dViewActive) {
        // Update 3D point cloud
        updatePointCloud(data);
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
    const frameData = frameBuffer.get();
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
            <div>缓冲: ${frameBuffer.size()}/${frameBuffer.maxSize}</div>
            <div>已接收: ${frameStats.received}</div>
        `;
    }
}

// Function to display video frame
function displayVideoFrame(frameData) {
    // 检查是否初始化了Canvas
    if (!videoCanvas || !ctx) {
        console.error('视频画布未初始化');
        return;
    }
    
    // 创建图像对象
    const img = new Image();
    
    // 添加更详细的图像加载完成回调
    img.onload = () => {
        try {
            console.log(`图像加载成功: ${img.width}x${img.height}`);
            
            // 清除画布
            ctx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
            
            // 绘制一个边框，确认画布正在被绘制
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.strokeRect(1, 1, videoCanvas.width - 2, videoCanvas.height - 2);
            
            // 绘制图像，保持纵横比
            const canvasRatio = videoCanvas.width / videoCanvas.height;
            const imgRatio = img.width / img.height;
            
            let drawWidth, drawHeight, offsetX, offsetY;
            
            if (canvasRatio > imgRatio) {
                // Canvas更宽，图像高度将填满
                drawHeight = videoCanvas.height;
                drawWidth = img.width * (drawHeight / img.height);
                offsetX = (videoCanvas.width - drawWidth) / 2;
                offsetY = 0;
            } else {
                // Canvas更高，图像宽度将填满
                drawWidth = videoCanvas.width;
                drawHeight = img.height * (drawWidth / img.width);
                offsetX = 0;
                offsetY = (videoCanvas.height - drawHeight) / 2;
            }
            
            // 绘制图像
            ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
            
            // 添加诊断叠加信息
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.fillRect(5, 5, 160, 60);
            
            ctx.font = '12px monospace';
            ctx.fillStyle = '#ffffff';
            ctx.fillText(`FPS: ${frameStats.avgFps.toFixed(1)}`, 10, 20);
            ctx.fillText(`帧: ${frameStats.received}/${frameStats.displayed}`, 10, 35);
            ctx.fillText(`缓冲: ${frameBuffer.size()}/${frameBuffer.maxSize}`, 10, 50);
            
            // 更新统计
            frameStats.displayed++;
            
            // 如果缓冲区中还有帧，处理下一帧
            if (frameBuffer.frames.length > 0) {
                requestAnimationFrame(processNextFrame);
            }
            
        } catch (renderError) {
            console.error('渲染图像时出错:', renderError);
            frameStats.errors++;
            
            // 显示错误信息
            ctx.fillStyle = 'red';
            ctx.fillRect(0, 0, videoCanvas.width, videoCanvas.height);
            ctx.fillStyle = 'white';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('图像渲染错误', videoCanvas.width/2, videoCanvas.height/2);
            ctx.fillText(renderError.message, videoCanvas.width/2, videoCanvas.height/2 + 30);
            
            // 如果缓冲区中还有帧，处理下一帧
            if (frameBuffer.frames.length > 0) {
                requestAnimationFrame(processNextFrame);
            }
        }
    };
    
    img.onerror = (error) => {
        console.error('加载图像时出错:', error);
        frameStats.errors++;
        
        // 显示错误信息在Canvas上
        ctx.fillStyle = 'red';
        ctx.fillRect(0, 0, videoCanvas.width, videoCanvas.height);
        ctx.fillStyle = 'white';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('图像加载错误', videoCanvas.width/2, videoCanvas.height/2);
        ctx.fillText('请检查网络连接或刷新页面', videoCanvas.width/2, videoCanvas.height/2 + 30);
        
        // 如果缓冲区中还有帧，处理下一帧
        if (frameBuffer.frames.length > 0) {
            requestAnimationFrame(processNextFrame);
        }
    };
    
    // 添加超时处理
    const imageLoadTimeout = setTimeout(() => {
        console.error('图像加载超时');
        img.src = ''; // 取消当前加载
        frameStats.errors++;
        
        // 显示超时信息
        ctx.fillStyle = 'orange';
        ctx.fillRect(0, 0, videoCanvas.width, videoCanvas.height);
        ctx.fillStyle = 'black';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('图像加载超时', videoCanvas.width/2, videoCanvas.height/2);
        ctx.fillText('请检查网络连接', videoCanvas.width/2, videoCanvas.height/2 + 30);
        
        // 如果缓冲区中还有帧，处理下一帧
        if (frameBuffer.frames.length > 0) {
            requestAnimationFrame(processNextFrame);
        }
    }, 5000); // 5秒超时
    
    // 设置图像源，并在完成后清除超时
    img.src = 'data:image/jpeg;base64,' + frameData;

    // 保存原始回调函数引用
    const originalOnload = img.onload;
    const originalOnerror = img.onerror;

    // 覆盖回调函数
    img.onload = () => {
        clearTimeout(imageLoadTimeout);
        originalOnload(); // 调用原始函数
    };

    img.onerror = (error) => {
        clearTimeout(imageLoadTimeout);
        originalOnerror(error); // 调用原始函数
    };
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
        
        carData = { x, y, client_type: 'web' };
        
        // Update display
        carXDisplay.textContent = x;
        carYDisplay.textContent = y;
    });
    
    carJoystick.on('end', () => {
        // Auto-centering: reset to zero when released
        carData = { x: 0, y: 0, client_type: 'web' };
        
        // Update display
        carXDisplay.textContent = '0';
        carYDisplay.textContent = '0';
        
        // Send stop command immediately
        socket.emit('car_control', { x: 0, y: 0, client_type: 'web' });
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
        
        cameraData = { x, y, client_type: 'web' };
    });
    
    cameraJoystick.on('end', () => {
        // Auto-centering: reset to zero when released
        cameraData = { x: 0, y: 0, client_type: 'web' };
        
        // Send reset command immediately
        socket.emit('gimbal_control', { x: 0, y: 0, client_type: 'web' });
    });
}

// Start control intervals
function startControlIntervals() {
    // Car control interval (100ms)
    carInterval = setInterval(() => {
        if (carData.x !== 0 || carData.y !== 0) {
            socket.emit('car_control', carData);
        }
    }, 100);
    
    // Camera control interval (200ms)
    cameraInterval = setInterval(() => {
        if (cameraData.x !== 0 || cameraData.y !== 0) {
            socket.emit('gimbal_control', cameraData);
        }
    }, 200);
}

// Stop control intervals
function stopControlIntervals() {
    clearInterval(carInterval);
    clearInterval(cameraInterval);
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
if (toggleMapViewBtn) {
    toggleMapViewBtn.addEventListener('click', toggleMapView);
}

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
        if (carData.x !== 0 || carData.y !== 0) {
            socket.emit('car_control', { x: 0, y: 0, client_type: 'web' });
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
    socket.emit('car_control', { x: 0, y: 0, client_type: 'web' });
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
    
    // 添加标题栏和折叠按钮
    const titleBar = document.createElement('div');
    titleBar.className = 'diagnostic-title-bar';
    
    const title = document.createElement('div');
    title.className = 'diagnostic-title';
    title.textContent = '视频流诊断面板';
    
    const toggleButton = document.createElement('button');
    toggleButton.className = 'diagnostic-toggle';
    toggleButton.innerHTML = '<i class="fas fa-chevron-down"></i>';
    toggleButton.title = '隐藏/显示诊断面板';
    
    titleBar.appendChild(title);
    titleBar.appendChild(toggleButton);
    
    // 创建内容区域
    const content = document.createElement('div');
    content.className = 'diagnostic-content';
    content.innerHTML = `
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
                <div class="diagnostic-label">渲染时间:</div>
                <div class="diagnostic-value" id="render-time">0 ms</div>
            </div>
        </div>
        <div class="diagnostic-section">
            <div class="section-title">网络状态</div>
            <div class="diagnostic-item">
                <div class="diagnostic-label">延迟:</div>
                <div class="diagnostic-value" id="latency">0 ms</div>
            </div>
            <div class="diagnostic-item">
                <div class="diagnostic-label">连接状态:</div>
                <div class="diagnostic-value" id="connection-status">已连接</div>
            </div>
            <div class="diagnostic-item">
                <div class="diagnostic-label">最后活动:</div>
                <div class="diagnostic-value" id="last-activity">刚刚</div>
            </div>
        </div>
        <div class="diagnostic-section">
            <div class="section-title">服务器资源</div>
            <div class="diagnostic-item">
                <div class="diagnostic-label">CPU使用率:</div>
                <div class="diagnostic-value" id="cpu-usage">0%</div>
            </div>
            <div class="diagnostic-item">
                <div class="diagnostic-label">内存使用率:</div>
                <div class="diagnostic-value" id="memory-usage">0%</div>
            </div>
        </div>
    `;
    
    // 组装面板
    diagnosticPanel.appendChild(titleBar);
    diagnosticPanel.appendChild(content);
    
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
            padding: 0;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
            transition: transform 0.3s ease, height 0.3s ease;
            overflow: hidden;
        }
        
        .diagnostic-panel.collapsed {
            height: 30px !important;
        }
        
        .diagnostic-title-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 10px;
            background-color: rgba(0, 122, 255, 0.7);
            border-radius: 5px 5px 0 0;
            cursor: pointer;
        }
        
        .diagnostic-title {
            font-weight: bold;
            font-size: 14px;
        }
        
        .diagnostic-toggle {
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            padding: 0;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.3s ease;
        }
        
        .diagnostic-panel.collapsed .diagnostic-toggle i {
            transform: rotate(180deg);
        }
        
        .diagnostic-content {
            padding: 10px;
        }
        
        .diagnostic-section {
            border: 1px solid #555;
            border-radius: 4px;
            padding: 8px;
            margin-bottom: 10px;
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
    `;
    
    // 添加到文档
    document.head.appendChild(style);
    document.body.appendChild(diagnosticPanel);
    
    // 添加折叠功能
    const titleBarElement = diagnosticPanel.querySelector('.diagnostic-title-bar');
    titleBarElement.addEventListener('click', function(e) {
        // 如果点击的是按钮本身，不处理（让按钮自己的事件处理）
        if (e.target.closest('.diagnostic-toggle')) {
            return;
        }
        
        diagnosticPanel.classList.toggle('collapsed');
        
        // 更新按钮图标
        const toggleButton = diagnosticPanel.querySelector('.diagnostic-toggle i');
        if (toggleButton) {
            toggleButton.className = diagnosticPanel.classList.contains('collapsed') 
                ? 'fas fa-chevron-up' 
                : 'fas fa-chevron-down';
        }
    });
    
    // 添加按钮功能
    const toggleButtonElement = diagnosticPanel.querySelector('.diagnostic-toggle');
    if (toggleButtonElement) {
        toggleButtonElement.addEventListener('click', function(e) {
            e.stopPropagation(); // 阻止事件冒泡
            diagnosticPanel.classList.toggle('collapsed');
            
            // 更新按钮图标
            const icon = this.querySelector('i');
            if (icon) {
                icon.className = diagnosticPanel.classList.contains('collapsed') 
                    ? 'fas fa-chevron-up' 
                    : 'fas fa-chevron-down';
            }
        });
    }
    
    // 从本地存储中恢复面板状态
    const isPanelCollapsed = localStorage.getItem('diagnosticPanelCollapsed') === 'true';
    if (isPanelCollapsed) {
        diagnosticPanel.classList.add('collapsed');
        const icon = diagnosticPanel.querySelector('.diagnostic-toggle i');
        if (icon) {
            icon.className = 'fas fa-chevron-up';
        }
    }
    
    // 保存面板状态到本地存储
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === 'class') {
                localStorage.setItem('diagnosticPanelCollapsed', diagnosticPanel.classList.contains('collapsed'));
            }
        });
    });
    
    observer.observe(diagnosticPanel, { attributes: true });
    
    return diagnosticPanel;
}

// 重置诊断统计数据
function resetDiagnosticStats() {
    frameStats.received = 0;
    frameStats.displayed = 0;
    frameStats.errors = 0;
    frameStats.fpsHistory = [];
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

function measureLatency() {
    lastPingTime = Date.now();
    socket.emit('ping_request');
}

// 更新诊断面板
function updateDiagnosticPanel() {
    const panel = document.getElementById('diagnostic-panel');
    if (!panel) return;
    
    // 更新帧率
    const frameRateElement = document.getElementById('frame-rate');
    if (frameRateElement) {
        frameRateElement.textContent = `${frameStats.avgFps.toFixed(1)} FPS`;
        
        // 根据帧率设置颜色
        if (frameStats.avgFps >= 15) {
            frameRateElement.style.color = '#4cd964'; // 绿色
        } else if (frameStats.avgFps >= 10) {
            frameRateElement.style.color = '#ffcc00'; // 黄色
        } else {
            frameRateElement.style.color = '#ff3b30'; // 红色
        }
    }
    
    // 更新帧大小
    const frameSizeElement = document.getElementById('frame-size');
    if (frameSizeElement && frameStats.lastFrameSize) {
        const sizeKB = (frameStats.lastFrameSize / 1024).toFixed(1);
        frameSizeElement.textContent = `${sizeKB} KB`;
    }
    
    // 更新渲染时间
    const renderTimeElement = document.getElementById('render-time');
    if (renderTimeElement && frameStats.lastRenderTime) {
        renderTimeElement.textContent = `${frameStats.lastRenderTime.toFixed(1)} ms`;
        
        // 根据渲染时间设置颜色
        if (frameStats.lastRenderTime < 20) {
            renderTimeElement.style.color = '#4cd964'; // 绿色
        } else if (frameStats.lastRenderTime < 50) {
            renderTimeElement.style.color = '#ffcc00'; // 黄色
        } else {
            renderTimeElement.style.color = '#ff3b30'; // 红色
        }
    }
    
    // 更新延迟
    const latencyElement = document.getElementById('latency');
    if (latencyElement && frameStats.latency) {
        latencyElement.textContent = `${frameStats.latency} ms`;
        
        // 根据延迟设置颜色
        if (frameStats.latency < 100) {
            latencyElement.style.color = '#4cd964'; // 绿色
        } else if (frameStats.latency < 300) {
            latencyElement.style.color = '#ffcc00'; // 黄色
        } else {
            latencyElement.style.color = '#ff3b30'; // 红色
        }
    }
    
    // 更新连接状态
    const connectionStatusElement = document.getElementById('connection-status');
    if (connectionStatusElement) {
        if (socket.connected) {
            connectionStatusElement.textContent = '已连接';
            connectionStatusElement.style.color = '#4cd964'; // 绿色
        } else {
            connectionStatusElement.textContent = '已断开';
            connectionStatusElement.style.color = '#ff3b30'; // 红色
        }
    }
    
    // 更新最后活动时间
    const lastActivityElement = document.getElementById('last-activity');
    if (lastActivityElement && frameStats.lastFrameTime) {
        const now = Date.now();
        const diff = now - frameStats.lastFrameTime;
        
        let timeText;
        if (diff < 1000) {
            timeText = '刚刚';
        } else if (diff < 60000) {
            timeText = `${Math.floor(diff / 1000)}秒前`;
        } else if (diff < 3600000) {
            timeText = `${Math.floor(diff / 60000)}分钟前`;
        } else {
            timeText = `${Math.floor(diff / 3600000)}小时前`;
        }
        
        lastActivityElement.textContent = timeText;
        
        // 根据时间差设置颜色
        if (diff < 5000) {
            lastActivityElement.style.color = '#4cd964'; // 绿色
        } else if (diff < 30000) {
            lastActivityElement.style.color = '#ffcc00'; // 黄色
        } else {
            lastActivityElement.style.color = '#ff3b30'; // 红色
        }
    }
    
    // 更新CPU和内存使用率
    const cpuUsageElement = document.getElementById('cpu-usage');
    const memoryUsageElement = document.getElementById('memory-usage');
    
    if (cpuUsageElement && typeof serverResources.cpu !== 'undefined') {
        cpuUsageElement.textContent = `${serverResources.cpu.percent}%`;
        
        // 根据CPU使用率设置颜色
        if (serverResources.cpu.percent < 50) {
            cpuUsageElement.style.color = '#4cd964'; // 绿色
        } else if (serverResources.cpu.percent < 80) {
            cpuUsageElement.style.color = '#ffcc00'; // 黄色
        } else {
            cpuUsageElement.style.color = '#ff3b30'; // 红色
        }
    }
    
    if (memoryUsageElement && typeof serverResources.memory !== 'undefined') {
        memoryUsageElement.textContent = `${serverResources.memory.percent}%`;
        
        // 根据内存使用率设置颜色
        if (serverResources.memory.percent < 50) {
            memoryUsageElement.style.color = '#4cd964'; // 绿色
        } else if (serverResources.memory.percent < 80) {
            memoryUsageElement.style.color = '#ffcc00'; // 黄色
        } else {
            memoryUsageElement.style.color = '#ff3b30'; // 红色
        }
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

// 添加base64数据验证函数
function validateBase64Data(base64String) {
    if (!base64String || typeof base64String !== 'string') {
        console.error('Base64数据为空或不是字符串');
        return false;
    }
    
    // 检查长度，太短的字符串可能无效
    if (base64String.length < 100) {
        console.error(`Base64数据过短: ${base64String.length}字符`);
        return false;
    }
    
    // 检查是否包含有效的Base64字符集
    const validBase64Regex = /^[A-Za-z0-9+/=]+$/;
    if (!validBase64Regex.test(base64String)) {
        console.error('Base64数据包含无效字符');
        return false;
    }
    
    // 检查填充字符的正确性
    if (base64String.length % 4 !== 0) {
        console.error('Base64数据长度不是4的倍数');
        return false;
    }
    
    console.log(`Base64数据验证通过，长度: ${base64String.length}字符`);
    return true;
}

// 处理ping响应，计算延迟
socket.on('ping_response', function() {
    currentLatency = Date.now() - lastPingTime;
    frameStats.latency = currentLatency;
    
    // 更新诊断面板中的延迟显示
    updateDiagnosticPanel();
});

// 处理服务器资源更新
socket.on('resource_update', function(data) {
    // 更新服务器资源信息
    serverResources = data;
    
    // 更新诊断面板
    updateDiagnosticPanel();
});

// 处理状态更新
socket.on('status_update', function(data) {
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

// Reset car controls
function resetCarControls() {
    carData = { x: 0, y: 0, client_type: 'web' };
    socket.emit('car_control', { x: 0, y: 0, client_type: 'web' });
}

// Reset camera controls
function resetCameraControls() {
    cameraData = { x: 0, y: 0, client_type: 'web' };
    socket.emit('gimbal_control', { x: 0, y: 0, client_type: 'web' });
} 