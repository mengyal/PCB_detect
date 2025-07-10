// PCB焊点检测系统 JavaScript
let autoScroll = true;

// 图片展示收起/展开功能配置
let expandedStates = {
    visualization: false,
    yolo: false
};
const maxDisplayItems = 3; // 默认显示的最大图片数量（修改为3张）

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    loadPreprocessingConfig();
    loadTrainingConfig();
    setupEventListeners();
    startLogPolling();
    refreshResults('yolo');
    refreshResults('visualization');
    loadQuickStats();
    
    // 初始化展开/收起状态
    initializeExpandStates();
});

// 全局通知弹窗函数，支持 success/error/warning/info
function showNotification(message, type = 'info') {
    let color = {
        success: '#4caf50',
        error: '#f44336',
        warning: '#ff9800',
        info: '#2196f3'
    }[type] || '#2196f3';

    let notification = document.createElement('div');
    notification.className = 'custom-notification';
    notification.style.position = 'fixed';
    notification.style.top = '30px';
    notification.style.right = '30px';
    notification.style.background = color;
    notification.style.color = '#fff';
    notification.style.padding = '12px 24px';
    notification.style.borderRadius = '6px';
    notification.style.boxShadow = '0 2px 8px rgba(0,0,0,0.15)';
    notification.style.zIndex = 9999;
    notification.style.fontSize = '16px';
    notification.innerText = message;

    document.body.appendChild(notification);
    setTimeout(() => {
        notification.remove();
    }, 2500);
}

// 初始化展开/收起状态
function initializeExpandStates() {
    addLogEntry('初始化图片展示收起/展开状态', 'INFO');
    
    // 重置所有展开状态为收起
    Object.keys(expandedStates).forEach(key => {
        expandedStates[key] = false;
    });
    
    addLogEntry('图片展示状态初始化完成，默认为收起模式', 'INFO');
}

// 初始化应用
function initializeApp() {
    console.log('PCB焊点检测系统初始化');
    updateSystemStatus('系统就绪', 'success');
}

// 设置事件监听器
function setupEventListeners() {
    // 文件上传
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('imageUpload');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    
    fileInput.addEventListener('change', handleFileSelect);
    
    // 监听resize输入变化
    document.getElementById('resizeWidth').addEventListener('change', validateSizeInput);
    document.getElementById('resizeHeight').addEventListener('change', validateSizeInput);
}

// 处理拖拽
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFiles(files);
    }
}

// 处理文件选择
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadFiles(files);
    }
}

// 验证尺寸输入
function validateSizeInput(e) {
    const value = parseInt(e.target.value);
    if (value < 64) e.target.value = 64;
    if (value > 2048) e.target.value = 2048;
    if (value % 32 !== 0) {
        e.target.value = Math.round(value / 32) * 32;
    }
}

// 更新系统状态
function updateSystemStatus(message, type = 'info') {
    const statusElement = document.getElementById('systemStatus');
    statusElement.textContent = message;
    statusElement.className = `status-indicator ${type}`;
}

// 显示加载状态
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'block';
}

// 隐藏加载状态
function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

// 显示模态窗口
function showModal(content) {
    document.getElementById('modalBody').innerHTML = content;
    document.getElementById('modal').style.display = 'block';
}

// 关闭模态窗口
function closeModal() {
    document.getElementById('modal').style.display = 'none';
}

// 关闭统计分析模态窗口
function closeStatisticsModal() {
    document.getElementById('statisticsModal').style.display = 'none';
}

// 关闭报告模态窗口
function closeReportsModal() {
    document.getElementById('reportsModal').style.display = 'none';
}

// 关闭报告详情模态窗口
function closeReportDetailModal() {
    document.getElementById('reportDetailModal').style.display = 'none';
}

// API调用封装
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        addLogEntry(`API调用失败: ${error.message}`, 'ERROR');
        throw error;
    }
}

// === 预处理相关功能 ===

// 加载预处理配置
async function loadPreprocessingConfig() {
    try {
        const config = await apiCall('/api/config/preprocessing');
        document.getElementById('resizeWidth').value = config.width;
        document.getElementById('resizeHeight').value = config.height;
    } catch (error) {
        addLogEntry('加载预处理配置失败', 'ERROR');
    }
}

// 更新预处理配置
async function updatePreprocessingConfig() {
    try {
        const width = parseInt(document.getElementById('resizeWidth').value);
        const height = parseInt(document.getElementById('resizeHeight').value);
        
        await apiCall('/api/config/preprocessing', {
            method: 'POST',
            body: JSON.stringify({ width, height })
        });
        
        addLogEntry(`预处理配置已更新: ${width}x${height}`, 'INFO');
        showNotification('预处理配置更新成功', 'success');
    } catch (error) {
        showNotification('预处理配置更新失败', 'error');
    }
}

// 运行预处理
async function runPreprocessing() {
    try {
        updateSystemStatus('正在预处理...', 'warning');
        const result = await apiCall('/api/run/preprocessing', { method: 'POST' });
        
        if (result.success) {
            addLogEntry('预处理任务已启动', 'INFO');
            showNotification('预处理任务已启动', 'success');
        }
    } catch (error) {
        updateSystemStatus('预处理失败', 'error');
        showNotification('预处理启动失败', 'error');
    }
}

// === 训练相关功能 ===

// 加载训练配置
async function loadTrainingConfig() {
    try {
        const config = await apiCall('/api/config/training');
        document.getElementById('imgDir').value = config.img_dir;
        document.getElementById('jsonDir').value = config.json_dir;
        document.getElementById('modelSavePath').value = config.model_save_path;
    } catch (error) {
        addLogEntry('加载训练配置失败', 'ERROR');
    }
}

// 更新训练配置
async function updateTrainingConfig() {
    try {
        const config = {
            img_dir: document.getElementById('imgDir').value,
            json_dir: document.getElementById('jsonDir').value,
            model_save_path: document.getElementById('modelSavePath').value
        };
        
        await apiCall('/api/config/training', {
            method: 'POST',
            body: JSON.stringify(config)
        });
        
        addLogEntry('训练配置已更新', 'INFO');
        showNotification('训练配置更新成功', 'success');
    } catch (error) {
        showNotification('训练配置更新失败', 'error');
    }
}

// 显示数据集结构
function showDatasetStructure(type) {
    apiCall('/api/upload/folder', {
        method: 'POST',
        body: JSON.stringify({ folder_type: type })
    }).then(result => {
        if (result.success) {
            const structure = result.structure;
            const content = `
                <h3>${structure.type}数据集结构</h3>
                <div class="dataset-structure">
                    <h4>标准目录结构：</h4>
                    <div class="structure-tree">
                        ${generateStructureHTML(structure.structure)}
                    </div>
                    <div class="highlight">
                        <strong>说明：</strong>${structure.description}
                    </div>
                </div>
            `;
            showModal(content);
        }
    });
}

// 生成结构HTML
function generateStructureHTML(structure, indent = 0) {
    let html = '';
    const indentStr = '&nbsp;'.repeat(indent * 4);
    
    for (const [key, value] of Object.entries(structure)) {
        if (typeof value === 'object') {
            html += `${indentStr}<span class="folder">📁 ${key}</span><br>`;
            html += generateStructureHTML(value, indent + 1);
        } else {
            html += `${indentStr}<span class="file">📄 ${key}</span> - <span class="description">${value}</span><br>`;
        }
    }
    
    return html;
}

// 运行训练
async function runTraining() {
    try {
        updateSystemStatus('正在训练...', 'warning');
        const result = await apiCall('/api/run/training', { method: 'POST' });
        
        if (result.success) {
            addLogEntry('训练任务已启动', 'INFO');
            showNotification('训练任务已启动', 'success');
        }
    } catch (error) {
        updateSystemStatus('训练失败', 'error');
        showNotification('训练启动失败', 'error');
    }
}

// === 文件上传功能 ===

// 上传文件
async function uploadFiles(files) {
    const formData = new FormData();
    
    for (let file of files) {
        if (file.type.startsWith('image/')) {
            formData.append('files', file);
        }
    }
    
    if (formData.get('files')) {
        try {
            updateSystemStatus('正在上传图片...', 'warning');
            
            const response = await fetch('/api/upload/image', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                addLogEntry(`成功上传 ${result.files.length} 个文件`, 'INFO');
                showNotification('图片上传成功', 'success');
                displayUploadedFiles(result.files);
                updateSystemStatus('图片上传完成', 'success');
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            addLogEntry(`图片上传失败: ${error.message}`, 'ERROR');
            showNotification('图片上传失败', 'error');
            updateSystemStatus('图片上传失败', 'error');
        }
    }
}

// 显示已上传文件
function displayUploadedFiles(files) {
    const container = document.getElementById('uploadedFiles');
    container.innerHTML = '';
    
    files.forEach(filename => {
        const fileDiv = document.createElement('div');
        fileDiv.className = 'uploaded-file';
        fileDiv.innerHTML = `
            <i class="fas fa-image"></i>
            <span>${filename}</span>
        `;
        container.appendChild(fileDiv);
    });
}

// === 检测功能 ===

// 运行检测
async function runDetection(modelType) {
    try {
        updateSystemStatus(`正在运行${modelType.toUpperCase()}检测...`, 'warning');
        const result = await apiCall('/api/run/detection', {
            method: 'POST',
            body: JSON.stringify({ model_type: modelType })
        });
        
        if (result.success) {
            addLogEntry(`${modelType.toUpperCase()}检测任务已启动`, 'INFO');
            showNotification(`${modelType.toUpperCase()}检测任务已启动`, 'success');
        }
    } catch (error) {
        updateSystemStatus('检测失败', 'error');
        showNotification('检测启动失败', 'error');
    }
}

// 运行可视化检测
function runVisualization() {
    if (!confirm('确定要运行可视化检测吗？这将处理data/users目录中的所有图片。')) {
        return;
    }
    
    updateSystemStatus('正在运行可视化检测...', 'warning');
    
    fetch('/api/run/visualization', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addLogEntry('可视化检测已启动', 'INFO');
            updateSystemStatus('可视化检测运行中...', 'info');
            
            // 5秒后自动刷新结果
            setTimeout(() => {
                refreshResults('visualization');
            }, 5000);
        } else {
            addLogEntry('启动可视化检测失败: ' + (data.error || '未知错误'), 'ERROR');
            updateSystemStatus('可视化检测启动失败', 'error');
        }
    })
    .catch(error => {
        console.error('启动可视化检测失败:', error);
        addLogEntry('启动可视化检测失败: ' + error.message, 'ERROR');
        updateSystemStatus('可视化检测启动失败', 'error');
    });
}

// 生成报告
function generateReports() {
    const mode = confirm('选择报告生成模式:\n确定 = 自动生成所有可视化结果的报告\n取消 = 手动生成指定报告');
    
    if (mode) {
        // 自动生成模式
        autoGenerateReports();
    } else {
        // 手动生成模式
        showManualReportGeneration();
    }
}

// 自动生成报告
function autoGenerateReports() {
    updateSystemStatus('正在自动生成报告...', 'warning');
    
    // 先获取现有报告列表，用于后续检查重复
    fetch('/api/reports')
    .then(response => response.json())
    .then(reportsData => {
        const existingReportImages = reportsData.reports.map(report => report.image_name);
        addLogEntry(`获取到现有报告列表，共 ${existingReportImages.length} 个报告`, 'INFO');
        
        return fetch('/api/reports/auto-generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                existingReportImages: existingReportImages,
                skipExisting: true  // 标记跳过已存在的报告
            })
        });
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const newCount = data.generated || 0;
            const skippedCount = data.skipped || 0;
            
            addLogEntry(`自动生成报告完成: 新增 ${newCount} 个报告, 跳过 ${skippedCount} 个已存在报告`, 'INFO');
            updateSystemStatus('报告生成完成', 'success');
            loadQuickStats();
            
            let message = `报告生成成功! 新增 ${newCount} 个报告`;
            if (skippedCount > 0) {
                message += `, 跳过 ${skippedCount} 个已存在报告`;
            }
            showNotification(message, 'success');
        } else {
            throw new Error(data.error || '自动生成报告失败');
        }
    })
    .catch(error => {
        console.error('自动生成报告失败:', error);
        addLogEntry('自动生成报告失败: ' + error.message, 'ERROR');
        updateSystemStatus('报告生成失败', 'error');
        showNotification('自动生成报告失败: ' + error.message, 'error');
    });
}

// 显示手动生成报告界面
function showManualReportGeneration() {
    // 获取可视化结果列表
    fetch('/api/results/visualization')
        .then(response => response.json())
        .then(data => {
            if (data.files && data.files.length > 0) {
                const fileOptions = data.files.map(file => 
                    `<option value="${file.name}">${file.name}</option>`
                ).join('');
                
                const formHtml = `
                    <div class="manual-report-form">
                        <h3>手动生成报告</h3>
                        <div class="form-group">
                            <label>选择可视化文件:</label>
                            <select id="selectedVisualization">
                                ${fileOptions}
                            </select>
                        </div>
                        <div class="form-group">
                            <label>PCB型号:</label>
                            <input type="text" id="manualPcbModel" value="PCB-MANUAL" placeholder="输入PCB型号">
                        </div>
                        <div class="form-group">
                            <label>批次号:</label>
                            <input type="text" id="manualBatchNumber" value="BATCH-MANUAL" placeholder="输入批次号">
                        </div>
                        <div class="form-group">
                            <label>备注:</label>
                            <textarea id="manualRemarks" placeholder="输入备注信息"></textarea>
                        </div>
                        <div class="form-actions">
                            <button class="btn btn-primary" onclick="submitManualReport()">生成报告</button>
                            <button class="btn btn-secondary" onclick="closeModal()">取消</button>
                        </div>
                    </div>
                `;
                
                document.getElementById('modalBody').innerHTML = formHtml;
                document.getElementById('modal').style.display = 'block';
            } else {
                alert('没有找到可视化结果文件，请先运行可视化检测。');
            }
        })
        .catch(error => {
            console.error('获取可视化文件列表失败:', error);
            alert('获取可视化文件列表失败: ' + error.message);
        });
}

// 提交手动报告
function submitManualReport() {
    const selectedFile = document.getElementById('selectedVisualization').value;
    const pcbModel = document.getElementById('manualPcbModel').value;
    const batchNumber = document.getElementById('manualBatchNumber').value;
    const remarks = document.getElementById('manualRemarks').value;
    
    if (!selectedFile) {
        alert('请选择可视化文件');
        return;
    }
    
    // 获取真实的分析数据
    const imageName = selectedFile.replace('_overlay.jpg', '.jpg');
    
    fetch(`/api/analysis/${imageName}`)
    .then(response => response.json())
    .then(analysisData => {
        if (analysisData.error) {
            throw new Error(analysisData.error);
        }
        
        // 使用真实的分析数据
        const total_points = analysisData.total_solder_points;
        const good_count = analysisData.good_solder_points;
        const defect_count = analysisData.defect_solder_points;
        const quality_rate = analysisData.quality_rate;
        
        const reportData = {
            image_name: imageName,
            pcb_model: pcbModel,
            batch_number: batchNumber,
            total_solder_points: total_points,
            good_count: good_count,
            excess_count: Math.floor(defect_count * 0.3),
            insufficient_count: Math.floor(defect_count * 0.3),
            shift_count: Math.floor(defect_count * 0.2),
            miss_count: Math.floor(defect_count * 0.2),
            defect_description: `手动生成: 良品${good_count}, 缺陷${defect_count}, 良品率${(quality_rate*100).toFixed(1)}%`,
            conclusion: quality_rate >= 0.95 ? 'PASS' : 'FAIL',
            remarks: `手动生成报告, 可视化文件: ${selectedFile}. ${remarks}`
        };
        
        return fetch('/api/reports/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(reportData)
        });
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addLogEntry(`手动生成报告成功: ID=${data.report_id}`, 'INFO');
            closeModal();
            loadQuickStats();
            alert('报告生成成功!');
        } else {
            addLogEntry('手动生成报告失败: ' + (data.error || '未知错误'), 'ERROR');
        }
    })
    .catch(error => {
        console.error('手动生成报告失败:', error);
        addLogEntry('手动生成报告失败: ' + error.message, 'ERROR');
        alert('报告生成失败: ' + error.message);
    });
}

// 清空日志
function clearLogs() {
    const logContainer = document.getElementById('logsContent');
    if (logContainer) {
        logContainer.innerHTML = '<div class="log-entry"><span class="log-time">[' + new Date().toLocaleString() + ']</span><span class="log-level info">[INFO]</span><span class="log-message">日志已清空</span></div>';
        addLogEntry('前端日志已清空', 'INFO');
    }
}

// 切换自动滚动
function toggleAutoScroll() {
    autoScroll = !autoScroll;
    const button = event.target.closest('button');
    const icon = button.querySelector('i');
    
    if (autoScroll) {
        icon.className = 'fas fa-arrow-down';
        button.title = '关闭自动滚动';
        addLogEntry('自动滚动已开启', 'INFO');
    } else {
        icon.className = 'fas fa-pause';
        button.title = '开启自动滚动';
        addLogEntry('自动滚动已关闭', 'INFO');
    }
}

// 刷新结果显示
async function refreshResults(modelType) {
    addLogEntry(`=== 开始刷新 ${modelType} 检测结果 ===`, 'INFO');
    
    try {
        addLogEntry(`请求URL: /api/results/${modelType}`, 'INFO');
        const response = await fetch(`/api/results/${modelType}`);
        const data = await response.json();
        
        addLogEntry(`API响应状态: ${response.status}`, 'INFO');
        addLogEntry(`返回文件数量: ${data.files ? data.files.length : 0}`, 'INFO');
        
        const gridId = modelType + 'ResultGrid';
        const grid = document.getElementById(gridId);
        
        if (!grid) {
            const errorMsg = `结果网格元素未找到: ${gridId}`;
            console.error(errorMsg);
            addLogEntry(errorMsg, 'ERROR');
            return;
        }
        
        if (data.files && data.files.length > 0) {
            addLogEntry(`开始渲染 ${data.files.length} 个文件`, 'INFO');
            
            // 检查是否需要收起部分图片
            const isExpanded = expandedStates[modelType];
            const displayFiles = isExpanded ? data.files : data.files.slice(0, maxDisplayItems);
            
            addLogEntry(`显示 ${displayFiles.length}/${data.files.length} 个文件 (${isExpanded ? '展开' : '收起'}模式)`, 'INFO');
            
            if (modelType === 'visualization') {
                // 可视化结果显示
                addLogEntry('渲染可视化检测结果界面', 'INFO');
                grid.innerHTML = displayFiles.map(file => {
                    const imageName = file.name.replace('_overlay.jpg', '');
                    return `
                        <div class="result-item visualization-item">
                            <div class="result-image">
                                <img src="/api/visualizations/${file.name}" 
                                     alt="${file.name}" 
                                     onclick="showImageModal('/api/visualizations/${file.name}', '${file.name}')">
                            </div>
                            <div class="result-info">
                                <h5>${imageName}</h5>
                                <p class="result-meta">
                                    <span class="file-size">${formatFileSize(file.size)}</span>
                                    <span class="file-date">${formatDate(file.modified)}</span>
                                </p>
                                <div class="result-actions">
                                    <button class="btn btn-sm btn-primary" 
                                            onclick="viewVisualizationDetails('${file.name}')">
                                        <i class="fas fa-eye"></i> 查看详情
                                    </button>
                                    <button class="btn btn-sm btn-success" 
                                            onclick="generateSingleReport('${file.name}')">
                                        <i class="fas fa-file-plus"></i> 生成报告
                                    </button>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');
            } else {
                // YOLO结果显示
                addLogEntry(`渲染 ${modelType.toUpperCase()} 检测结果界面`, 'INFO');
                grid.innerHTML = displayFiles.map(file => `
                    <div class="result-item">
                        <div class="result-image">
                            <img src="/api/results/${modelType}/${file.name}" 
                                 alt="${file.name}" 
                                 onclick="showImageModal('/api/results/${modelType}/${file.name}', '${file.name}')">
                        </div>
                        <div class="result-info">
                            <h5>${file.name}</h5>
                            <p class="result-meta">
                                <span class="file-size">${formatFileSize(file.size)}</span>
                                <span class="file-date">${formatDate(file.modified)}</span>
                            </p>
                            <div class="result-actions">
                                <button class="btn btn-sm btn-primary" onclick="downloadResult('${modelType}', '${file.name}')">
                                    <i class="fas fa-download"></i> 下载
                                </button>
                            </div>
                        </div>
                    </div>
                `).join('');
            }
            
            // 更新展开/收起按钮
            updateExpandButton(modelType, data.files.length);
            
            addLogEntry(`${modelType} 检测结果界面渲染完成`, 'INFO');
        } else {
            // 没有结果时显示空状态
            addLogEntry(`${modelType} 暂无检测结果，显示空状态`, 'INFO');
            const message = modelType === 'visualization' ? '暂无检测结果' : '暂无检测结果';
            grid.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-inbox"></i>
                    <p>${message}</p>
                </div>
            `;
        }
        
        addLogEntry(`=== ${modelType} 检测结果刷新完成 ===`, 'INFO');
    } catch (error) {
        const errorMsg = `刷新${modelType}结果失败: ${error.message}`;
        console.error(errorMsg, error);
        addLogEntry(errorMsg, 'ERROR');
    }
}

// 显示图片模态框
function showImageModal(imageSrc, imageName) {
    const modal = document.getElementById('modal');
    const modalBody = document.getElementById('modalBody');
    
    modalBody.innerHTML = `
        <div class="image-modal-content">
            <h3>${imageName}</h3>
            <div class="image-container">
                <img src="${imageSrc}" alt="${imageName}" style="max-width: 100%; max-height: 70vh; object-fit: contain;">
            </div>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeModal()">关闭</button>
                <a href="${imageSrc}" download="${imageName}" class="btn btn-primary">
                    <i class="fas fa-download"></i> 下载
                </a>
            </div>
        </div>
    `;
    
    modal.style.display = 'block';
}

// 查看可视化详情
function viewVisualizationDetails(fileName) {
    const imageName = fileName.replace('_overlay.jpg', '');
    
    // 显示详情模态框
    const modal = document.getElementById('modal');
    const modalBody = document.getElementById('modalBody');
    
    modalBody.innerHTML = `
        <div class="visualization-details">
            <h3>可视化详情 - ${imageName}</h3>
            <div class="detail-section">
                <h4>检测结果图片</h4>
                <div class="image-container">
                    <img src="/api/visualizations/${fileName}" alt="${fileName}" 
                         style="max-width: 100%; max-height: 50vh; object-fit: contain;">
                </div>
            </div>
            <div class="detail-section">
                <h4>文件信息</h4>
                <table class="detail-table">
                    <tr><td>文件名:</td><td>${fileName}</td></tr>
                    <tr><td>原始图片:</td><td>${imageName}.jpg</td></tr>
                    <tr><td>类型:</td><td>可视化检测结果</td></tr>
                </table>
            </div>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeModal()">关闭</button>
                <button class="btn btn-success" onclick="generateSingleReport('${fileName}')">
                    <i class="fas fa-file-plus"></i> 生成报告
                </button>
                <a href="/api/visualizations/${fileName}" download="${fileName}" class="btn btn-primary">
                    <i class="fas fa-download"></i> 下载图片
                </a>
            </div>
        </div>
    `;
    
    modal.style.display = 'block';
}

// 生成单个报告
function generateSingleReport(fileName) {
    if (!confirm(`确定要为 ${fileName} 生成检测报告吗？`)) {
        return;
    }
    
    const imageName = fileName.replace('_overlay.jpg', '.jpg');
    addLogEntry(`开始检查图片 ${imageName} 是否已有报告...`, 'INFO');
    
    // 先检查是否已经存在该图片的报告
    fetch('/api/reports')
    .then(response => response.json())
    .then(data => {
        // 检查是否已经存在该图片的报告
        const existingReport = data.reports.find(report => report.image_name === imageName);
        
        if (existingReport) {
            addLogEntry(`图片 ${imageName} 已存在报告 (ID: ${existingReport.id})`, 'WARNING');
            if (!confirm(`该图片已有报告(ID: ${existingReport.id})，是否重新生成？`)) {
                showNotification(`已取消重复生成报告`, 'warning');
                return Promise.reject(new Error('用户取消重复生成报告'));
            }
            addLogEntry(`用户确认重新生成报告`, 'INFO');
        }
        
        // 获取真实的分析数据
        return fetch(`/api/analysis/${imageName}`);
    })
    .then(response => response.json())
    .then(analysisData => {
        if (analysisData.error) {
            throw new Error(analysisData.error);
        }
        
        // 使用真实的分析数据
        const total_points = analysisData.total_solder_points;
        const good_count = analysisData.good_solder_points;
        const defect_count = analysisData.defect_solder_points;
        const quality_rate = analysisData.quality_rate;
        
        const reportData = {
            image_name: imageName,
            pcb_model: 'PCB-AUTO',
            batch_number: 'BATCH-AUTO',
            total_solder_points: total_points,
            good_count: good_count,
            excess_count: Math.floor(defect_count * 0.3),
            insufficient_count: Math.floor(defect_count * 0.3),
            shift_count: Math.floor(defect_count * 0.2),
            miss_count: Math.floor(defect_count * 0.2),
            defect_description: `自动检测: 良品${good_count}, 缺陷${defect_count}, 良品率${(quality_rate*100).toFixed(1)}%`,
            conclusion: quality_rate >= 0.95 ? 'PASS' : 'FAIL',
            remarks: `从可视化文件自动生成: ${fileName}`
        };
        
        return fetch('/api/reports/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(reportData)
        });
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addLogEntry(`生成报告成功: ID=${data.report_id}, 文件=${fileName}`, 'INFO');
            loadQuickStats();
            showNotification('报告生成成功!', 'success');
        } else {
            addLogEntry('生成报告失败: ' + (data.error || '未知错误'), 'ERROR');
            showNotification('报告生成失败', 'error');
        }
    })
    .catch(error => {
        // 如果是用户取消重复生成报告，不显示错误
        if (error.message === '用户取消重复生成报告') {
            return;
        }
        console.error('生成报告失败:', error);
        addLogEntry('生成报告失败: ' + error.message, 'ERROR');
        showNotification('报告生成失败: ' + error.message, 'error');
    });
}

// 下载结果文件
function downloadResult(modelType, fileName) {
    addLogEntry(`下载 ${modelType} 结果文件: ${fileName}`, 'INFO');
    
    try {
        const link = document.createElement('a');
        link.href = `/api/results/${modelType}/${fileName}`;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        addLogEntry(`文件下载成功: ${fileName}`, 'INFO');
    } catch (error) {
        addLogEntry(`文件下载失败: ${error.message}`, 'ERROR');
    }
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 格式化日期
function formatDate(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString('zh-CN');
}

// 页面加载时初始化日志系统
document.addEventListener('DOMContentLoaded', function() {
    addLogEntry('PCB焊点检测系统前端已加载', 'INFO');
    addLogEntry('日志系统已初始化', 'INFO');
});

// ==================== 日志系统 ====================

// 添加日志条目
function addLogEntry(message, level = 'INFO') {
    console.log(`[${level}] ${message}`);
    
    // 发送日志到后端（可选）
    fetch('/api/logs', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: message,
            level: level,
            timestamp: new Date().toISOString()
        })
    }).catch(err => console.error('发送日志失败:', err));
}

// 开始日志轮询
function startLogPolling() {
    console.log('开始日志轮询...');
    
    // 每2秒获取一次最新日志
    setInterval(() => {
        fetchLatestLogs();
    }, 2000);
}

// 获取最新日志
async function fetchLatestLogs() {
    try {
        const response = await fetch('/api/logs');
        const data = await response.json();
        
        if (data.logs && data.logs.length > 0) {
            updateLogDisplay(data.logs);
        }
    } catch (error) {
        console.error('获取日志失败:', error);
    }
}

// 更新日志显示
function updateLogDisplay(logs) {
    const logContainer = document.getElementById('logsContent');
    
    if (logContainer) {
        // 清空现有日志
        logContainer.innerHTML = '';
        
        // 添加新日志
        logs.forEach(log => {
            const logElement = document.createElement('div');
            logElement.className = 'log-entry';
            
            // 解析日志格式: [timestamp] [level] message
            const timestampMatch = log.match(/\[(.*?)\]/);
            const levelMatch = log.match(/\]\s*\[(.*?)\]/);
            
            let timestamp = timestampMatch ? timestampMatch[1] : new Date().toLocaleString();
            let level = levelMatch ? levelMatch[1] : 'INFO';
            let message = log.replace(/\[.*?\]\s*\[.*?\]\s*/, '');
            
            // 如果有第三个方括号（如[前端]），也要移除
            message = message.replace(/^\[.*?\]\s*/, '');
            
            // 设置级别样式
            const levelClass = level.toLowerCase();
            
            logElement.innerHTML = `
                <span class="log-time">[${timestamp}]</span>
                <span class="log-level ${levelClass}">[${level}]</span>
                <span class="log-message">${message}</span>
            `;
            
            logContainer.appendChild(logElement);
        });
        
        // 自动滚动到底部
        if (autoScroll) {
            logContainer.scrollTop = logContainer.scrollHeight;
        }
    }
}

// 加载并更新快速统计显示
function loadQuickStats() {
    addLogEntry('刷新快速统计数据', 'INFO');
    
    fetch('/api/reports/statistics')
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // 验证DOM元素是否存在
        const elements = {
            totalReports: document.getElementById('totalReports'),
            averageYield: document.getElementById('averageYield'),
            totalSolderPoints: document.getElementById('totalSolderPoints'),
            goodSolderPoints: document.getElementById('goodSolderPoints'),
            defectSolderPoints: document.getElementById('defectSolderPoints')
        };
        
        // 检查所有元素是否存在
        for (let [key, element] of Object.entries(elements)) {
            if (!element) {
                addLogEntry(`未找到DOM元素: ${key}`, 'ERROR');
                console.error(`Element not found: ${key}`);
                return;
            }
        }
        
        // 更新各个统计项目
        elements.totalReports.textContent = data.total_reports || 0;
        elements.averageYield.textContent = `${(data.avg_pass_rate || 0).toFixed(1)}%`;
        elements.totalSolderPoints.textContent = data.total_solder_points || 0;
        elements.goodSolderPoints.textContent = data.total_good || 0;
        elements.defectSolderPoints.textContent = data.total_defects || 0;
        
        addLogEntry('快速统计数据已更新', 'INFO');
        console.log('Quick stats updated:', data);
        console.log('Updated elements:', {
            totalReports: elements.totalReports.textContent,
            averageYield: elements.averageYield.textContent,
            totalSolderPoints: elements.totalSolderPoints.textContent,
            goodSolderPoints: elements.goodSolderPoints.textContent,
            defectSolderPoints: elements.defectSolderPoints.textContent
        });
    })
    .catch(error => {
        addLogEntry(`更新快速统计失败: ${error.message}`, 'ERROR');
        console.error('Failed to update quick stats:', error);
        
        // 设置错误状态
        document.getElementById('totalReports').textContent = '错误';
        document.getElementById('averageYield').textContent = '错误';
        document.getElementById('totalSolderPoints').textContent = '错误';
        document.getElementById('goodSolderPoints').textContent = '错误';
        document.getElementById('defectSolderPoints').textContent = '错误';
    });
}

// 刷新结果时添加日志并重新应用缩放
const originalRefreshResults = refreshResults;
refreshResults = async function(modelType) {
    addLogEntry(`开始刷新 ${modelType} 检测结果`, 'INFO');
    
    try {
        await originalRefreshResults(modelType);
        addLogEntry(`${modelType} 检测结果刷新完成`, 'INFO');
        
        // 刷新后重新应用缩放
        setTimeout(() => {
            if (modelType === 'visualization') {
                applyImageZoom('segment');
            } else if (modelType === 'yolo') {
                applyImageZoom('yolo');
            }
        }, 100);
        
    } catch (error) {
        addLogEntry(`刷新 ${modelType} 检测结果失败: ${error.message}`, 'ERROR');
        throw error;
    }
};

// 查看可视化详情时添加日志
const originalViewVisualizationDetails = viewVisualizationDetails;
viewVisualizationDetails = function(fileName) {
    addLogEntry(`查看可视化详情: ${fileName}`, 'INFO');
    originalViewVisualizationDetails(fileName);
};

// 生成单个报告时添加日志
const originalGenerateSingleReport = generateSingleReport;
generateSingleReport = function(fileName) {
    addLogEntry(`开始为 ${fileName} 生成单个报告`, 'INFO');
    originalGenerateSingleReport(fileName);
};

// 显示报告列表模态框
function showReportsModal() {
    addLogEntry('打开报告列表', 'INFO');
    
    fetch('/api/reports')
    .then(response => response.json())
    .then(data => {
        if (data.reports && data.reports.length > 0) {
            let reportsHtml = `
                <div class="reports-modal">
                    <h3>检测报告列表</h3>
                    <div class="reports-table-container">
                        <table class="reports-table">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>图片名称</th>
                                    <th>PCB型号</th>
                                    <th>批次号</th>
                                    <th>检测时间</th>
                                    <th>总焊点</th>
                                    <th>良品数</th>
                                    <th>缺陷数</th>
                                    <th>良品率</th>
                                    <th>结论</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody>
            `;
            
            data.reports.forEach(report => {
                // 添加通过率的颜色样式
                const passRateClass = report.pass_rate >= 95 ? 'high-pass-rate' : 
                                     report.pass_rate >= 85 ? 'medium-pass-rate' : 'low-pass-rate';
                
                reportsHtml += `
                    <tr>
                        <td>${report.id}</td>
                        <td class="image-name">${report.image_name}</td>
                        <td>${report.pcb_model}</td>
                        <td>${report.batch_number}</td>
                        <td>${new Date(report.detection_time).toLocaleString('zh-CN')}</td>
                        <td class="number-cell">${report.total_solder_points}</td>
                        <td class="number-cell good-count">${report.good_count}</td>
                        <td class="number-cell defect-count">${report.defect_count}</td>
                        <td class="number-cell ${passRateClass}">${report.pass_rate}%</td>
                        <td class="conclusion-${report.conclusion.toLowerCase()}">${report.conclusion}</td>
                        <td>
                            <button class="btn btn-sm btn-info" onclick="viewReportDetail(${report.id})" title="查看详细报告">
                                <i class="fas fa-eye"></i> 详情
                            </button>
                        </td>
                    </tr>
                `;
            });
            
            reportsHtml += `
                            </tbody>
                        </table>
                    </div>
                    <div class="modal-actions">
                        <button class="btn btn-secondary" onclick="closeModal()">关闭</button>
                    </div>
                </div>
            `;
            
            showModal(reportsHtml);
            addLogEntry(`显示 ${data.reports.length} 条报告记录`, 'INFO');
        } else {
            showModal(`
                <div class="reports-modal">
                    <h3>检测报告列表</h3>
                    <p class="no-data">暂无报告数据</p>
                    <div class="modal-actions">
                        <button class="btn btn-secondary" onclick="closeModal()">关闭</button>
                    </div>
                </div>
            `);
            addLogEntry('暂无报告数据', 'WARNING');
        }
    })
    .catch(error => {
        console.error('获取报告列表失败:', error);
        addLogEntry('获取报告列表失败: ' + error.message, 'ERROR');
    });
}

// 查看报告详情
function viewReportDetail(reportId) {
    addLogEntry(`查看报告详情: ID=${reportId}`, 'INFO');
    
    fetch(`/api/reports/${reportId}`)
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        const passRate = data.total_solder_points > 0 ? 
            (data.good_count / data.total_solder_points * 100).toFixed(1) : 0;
        
        const detailHtml = `
            <div class="report-detail">
                <h3>报告详情 - ID: ${data.id}</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <label>图片名称:</label>
                        <span>${data.image_name}</span>
                    </div>
                    <div class="detail-item">
                        <label>PCB型号:</label>
                        <span>${data.pcb_model}</span>
                    </div>
                    <div class="detail-item">
                        <label>批次号:</label>
                        <span>${data.batch_number}</span>
                    </div>
                    <div class="detail-item">
                        <label>检测时间:</label>
                        <span>${data.detection_time}</span>
                    </div>
                    <div class="detail-item">
                        <label>检测设备:</label>
                        <span>${data.detection_device}</span>
                    </div>
                    <div class="detail-item">
                        <label>总焊点数:</label>
                        <span>${data.total_solder_points}</span>
                    </div>
                    <div class="detail-item">
                        <label>良品数:</label>
                        <span>${data.good_count}</span>
                    </div>
                    <div class="detail-item">
                        <label>良品率:</label>
                        <span>${passRate}%</span>
                    </div>
                    <div class="detail-item">
                        <label>过量焊点:</label>
                        <span>${data.excess_count}</span>
                    </div>
                    <div class="detail-item">
                        <label>不足焊点:</label>
                        <span>${data.insufficient_count}</span>
                    </div>
                    <div class="detail-item">
                        <label>偏移焊点:</label>
                        <span>${data.shift_count}</span>
                    </div>
                    <div class="detail-item">
                        <label>缺失焊点:</label>
                        <span>${data.miss_count}</span>
                    </div>
                    <div class="detail-item full-width">
                        <label>缺陷描述:</label>
                        <span>${data.defect_description}</span>
                    </div>
                    <div class="detail-item">
                        <label>检测结论:</label>
                        <span class="conclusion-${data.conclusion.toLowerCase()}">${data.conclusion}</span>
                    </div>
                    <div class="detail-item full-width">
                        <label>备注:</label>
                        <span>${data.remarks || '无'}</span>
                    </div>
                </div>
                <div class="modal-actions">
                    <button class="btn btn-secondary" onclick="closeModal()">关闭</button>
                </div>
            </div>
        `;
        
        showModal(detailHtml);
        addLogEntry('报告详情显示成功', 'INFO');
    })
    .catch(error => {
        console.error('获取报告详情失败:', error);
        addLogEntry('获取报告详情失败: ' + error.message, 'ERROR');
    });
}

// 显示统计分析模态框
function showStatisticsModal() {
    addLogEntry('打开统计分析', 'INFO');
    
    fetch('/api/reports/statistics')
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        addLogEntry('统计数据获取成功，开始渲染界面', 'INFO');
        console.log('Statistics data:', data); // 调试日志
        
        // 验证数据完整性
        if (!data) {
            throw new Error('收到空数据');
        }
        
        console.log('各字段数据:', {
            total_reports: data.total_reports,
            total_solder_points: data.total_solder_points,
            total_good: data.total_good,
            total_defects: data.total_defects,
            avg_pass_rate: data.avg_pass_rate
        });
        
        const statsHtml = `
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-file-alt"></i>
                    </div>
                    <div class="stat-content">
                        <h4>${data.total_reports || 0}</h4>
                        <p>总报告数</p>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-circle"></i>
                    </div>
                    <div class="stat-content">
                        <h4>${data.total_solder_points || 0}</h4>
                        <p>总焊点数</p>
                    </div>
                </div>
                <div class="stat-card success">
                    <div class="stat-icon">
                        <i class="fas fa-check-circle"></i>
                    </div>
                    <div class="stat-content">
                        <h4>${data.total_good || 0}</h4>
                        <p>良品焊点</p>
                    </div>
                </div>
                <div class="stat-card error">
                    <div class="stat-icon">
                        <i class="fas fa-exclamation-circle"></i>
                    </div>
                    <div class="stat-content">
                        <h4>${data.total_defects || 0}</h4>
                        <p>缺陷焊点</p>
                    </div>
                </div>
                <div class="stat-card info">
                    <div class="stat-icon">
                        <i class="fas fa-percentage"></i>
                    </div>
                    <div class="stat-content">
                        <h4>${(data.avg_pass_rate || 0).toFixed(1)}%</h4>
                        <p>平均良品率</p>
                    </div>
                </div>
            </div>
                
                <div class="stats-details">
                    <div class="stats-section">
                        <h4>检测结论分布</h4>
                        <div class="conclusion-stats">
                            <div class="conclusion-item pass">
                                <span class="label">通过 (PASS):</span>
                                <span class="value">${data.conclusion_distribution?.PASS || 0}</span>
                            </div>
                            <div class="conclusion-item fail">
                                <span class="label">失败 (FAIL):</span>
                                <span class="value">${data.conclusion_distribution?.FAIL || 0}</span>
                            </div>
                        </div>
                    </div>
                    
                    ${data.trends && data.trends.length > 0 ? `
                    <div class="stats-section">
                        <h4>检测趋势</h4>
                        <div class="trends">
                            ${data.trends.map(trendData => `
                                <div class="trend-item">
                                    <span class="date">${trendData[0]}:</span>
                                    <span class="count">${trendData[1]} 个报告</span>
                                    <span class="rate">良品率 ${trendData[2].toFixed(1)}%</span>
                                    <span class="defects">${trendData[3]} 个缺陷</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    ` : ''}
                </div>
                
                <div class="modal-actions">
                    <button class="btn btn-secondary" onclick="closeStatisticsModal()">关闭</button>
                </div>
            `;
        
        // 直接将内容放入统计模态框
        document.getElementById('statisticsContent').innerHTML = statsHtml;
        document.getElementById('statisticsModal').style.display = 'block';
        addLogEntry('统计分析显示成功', 'INFO');
    })
    .catch(error => {
        console.error('获取统计数据失败:', error);
        addLogEntry('获取统计数据失败: ' + error.message, 'ERROR');
        
        // 显示错误提示
        const errorHtml = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>无法获取统计数据，请稍后重试</p>
                <p>错误详情: ${error.message}</p>
            </div>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeStatisticsModal()">关闭</button>
            </div>
        `;
        document.getElementById('statisticsContent').innerHTML = errorHtml;
        document.getElementById('statisticsModal').style.display = 'block';
    });
}

// 导出报告功能
function exportReports() {
    addLogEntry('用户点击导出报告按钮', 'INFO');
    
    const dateFilter = prompt('请输入日期筛选 (格式: YYYY-MM-DD，留空表示导出全部):', '');
    
    fetch('/api/reports/export', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            date_filter: dateFilter,
            filename: `pcb_reports_${new Date().toISOString().split('T')[0]}.csv`
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addLogEntry(`报告导出成功: ${data.filename}`, 'INFO');
            
            // 自动下载文件
            const downloadLink = document.createElement('a');
            downloadLink.href = data.download_url;
            downloadLink.download = data.filename;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            
            alert('报告导出成功！正在下载...');
        } else {
            addLogEntry('报告导出失败: ' + (data.error || '未知错误'), 'ERROR');
            alert('报告导出失败: ' + (data.error || '未知错误'));
        }
    })
    .catch(error => {
        console.error('导出报告失败:', error);
        addLogEntry('导出报告失败: ' + error.message, 'ERROR');
        alert('导出报告失败: ' + error.message);
    });
}

// ===== 检测结果标签切换功能 =====
function switchResultTab(tabType) {
    addLogEntry(`切换到 ${tabType} 检测结果标签`, 'INFO');
    
    // 更新标签状态
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // 激活当前标签
    const currentTab = document.getElementById(tabType + 'Tab');
    if (currentTab) {
        currentTab.classList.add('active');
    }
    
    // 切换面板显示
    document.querySelectorAll('.result-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    
    const currentPanel = document.getElementById(tabType + 'Results');
    if (currentPanel) {
        currentPanel.classList.add('active');
        addLogEntry(`成功切换到 ${tabType} 面板`, 'INFO');
    } else {
        addLogEntry(`未找到面板元素: ${tabType}Results`, 'ERROR');
    }
}

// ===== 图片缩放功能 =====
let zoomLevels = {
    segment: 1.0,
    yolo: 1.0
};

// 调整图片大小
function adjustImageSize(tabType, scale) {
    addLogEntry(`调整 ${tabType} 图片大小，比例: ${scale}`, 'INFO');
    
    // 更新缩放比例
    zoomLevels[tabType] *= scale;
    
    // 限制缩放范围 (0.2x - 5.0x)
    if (zoomLevels[tabType] < 0.2) {
        zoomLevels[tabType] = 0.2;
    } else if (zoomLevels[tabType] > 5.0) {
        zoomLevels[tabType] = 5.0;
    }
    
    // 应用缩放
    applyImageZoom(tabType);
    
    // 更新显示的缩放比例
    updateZoomDisplay(tabType);
    
    addLogEntry(`${tabType} 图片缩放比例更新为: ${(zoomLevels[tabType] * 100).toFixed(0)}%`, 'INFO');
}

// 重置图片大小
function resetImageSize(tabType) {
    addLogEntry(`重置 ${tabType} 图片大小`, 'INFO');
    
    zoomLevels[tabType] = 1.0;
    applyImageZoom(tabType);
    updateZoomDisplay(tabType);
    
    addLogEntry(`${tabType} 图片大小已重置为100%`, 'INFO');
}

// 应用图片缩放
function applyImageZoom(tabType) {
    const panelId = tabType + 'Results';
    const panel = document.getElementById(panelId);
    
    if (!panel) {
        addLogEntry(`未找到面板: ${panelId}`, 'ERROR');
        return;
    }
    
    const images = panel.querySelectorAll('.result-image img');
    const scale = zoomLevels[tabType];
    
    images.forEach(img => {
        img.style.transform = `scale(${scale})`;
        img.style.transformOrigin = 'center';
        img.style.transition = 'transform 0.3s ease';
        
        // 调整父容器以适应缩放
        const container = img.closest('.result-image');
        if (container) {
            if (scale > 1) {
                container.style.overflow = 'auto';
                container.style.maxHeight = '400px';
            } else {
                container.style.overflow = 'hidden';
                container.style.maxHeight = 'none';
            }
        }
    });
    
    addLogEntry(`应用 ${tabType} 图片缩放: ${(scale * 100).toFixed(0)}%`, 'INFO');
}

// 更新缩放比例显示
function updateZoomDisplay(tabType) {
    const zoomDisplayId = tabType + 'ZoomLevel';
    const zoomDisplay = document.getElementById(zoomDisplayId);
    
    if (zoomDisplay) {
        const percentage = (zoomLevels[tabType] * 100).toFixed(0);
        zoomDisplay.textContent = percentage + '%';
    }
}

// 切换收起/展开状态
function toggleExpandState(modelType) {
    expandedStates[modelType] = !expandedStates[modelType];
    const gridId = modelType + 'ResultGrid';
    const grid = document.getElementById(gridId);
    
    if (grid) {
        const items = grid.querySelectorAll('.result-item');
        
        if (expandedStates[modelType]) {
            // 展开所有
            items.forEach((item, index) => {
                item.style.display = 'flex';
                
                // 仅在展开状态下加载图片
                const img = item.querySelector('img');
                if (img && !img.src) {
                    img.src = img.getAttribute('data-src');
                }
            });
            
            addLogEntry(`已展开 ${modelType} 所有检测结果`, 'INFO');
        } else {
            // 收起，仅显示前N个
            const visibleItems = Array.from(items).slice(0, maxDisplayItems);
            const hiddenItems = Array.from(items).slice(maxDisplayItems);
            
            visibleItems.forEach(item => {
                item.style.display = 'flex';
            });
            
            hiddenItems.forEach(item => {
                item.style.display = 'none';
            });
            
            addLogEntry(`已收起 ${modelType} 检测结果，仅显示前${maxDisplayItems}个`, 'INFO');
        }
    } else {
        addLogEntry(`未找到网格元素: ${gridId}`, 'ERROR');
    }
}

// 切换展开/收起状态
function toggleExpanded(resultType) {
    expandedStates[resultType] = !expandedStates[resultType];
    addLogEntry(`${resultType} 图片展示状态切换为: ${expandedStates[resultType] ? '展开' : '收起'}`, 'INFO');
    
    // 刷新该类型的结果显示
    refreshResults(resultType);
}

// 更新展开/收起按钮
function updateExpandButton(resultType, totalCount) {
    const expandButtonId = `${resultType}ExpandBtn`;
    let expandButton = document.getElementById(expandButtonId);
    
    // 如果按钮不存在，创建它
    if (!expandButton) {
        const grid = document.getElementById(getGridId(resultType));
        if (grid && grid.parentNode) {
            expandButton = document.createElement('div');
            expandButton.id = expandButtonId;
            expandButton.className = 'expand-button-container';
            grid.parentNode.appendChild(expandButton);
        }
    }
    
    if (expandButton && totalCount > maxDisplayItems) {
        const isExpanded = expandedStates[resultType];
        const hiddenCount = totalCount - maxDisplayItems;
        
        expandButton.innerHTML = `
            <button class="btn btn-outline expand-btn" onclick="toggleExpanded('${resultType}')">
                <i class="fas fa-${isExpanded ? 'chevron-up' : 'chevron-down'}"></i>
                ${isExpanded ? '收起' : `展开 (还有${hiddenCount}张图片)`}
            </button>
        `;
        expandButton.style.display = 'block';
    } else if (expandButton) {
        expandButton.style.display = 'none';
    }
}

// 获取网格容器ID
function getGridId(resultType) {
    return resultType === 'visualization' ? 'visualizationResultGrid' : 'yoloResultGrid';
}

// 初始化时收起结果
document.addEventListener('DOMContentLoaded', function() {
    // 默认收起可视化结果
    toggleExpandState('visualization');
    
    // YOLO结果不自动收起，以便于查看
    expandedStates.yolo = true;
    const yoloGrid = document.getElementById('yoloResultGrid');
    if (yoloGrid) {
        yoloGrid.querySelectorAll('.result-item').forEach(item => {
            item.style.display = 'flex';
        });
    }
});
