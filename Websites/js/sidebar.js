let historyPollingTimer = null;
let lastHistoryData = null;
function checkLoginAndLoadHistory() {
    fetch('./api/users/check_login.php')
        .then(response => response.json())
        .then(data => {
            const historyList = document.getElementById('historyList');
            if (data.isLoggedIn) {
                historyList.innerHTML = '<div class="loading-history">Loading history...</div>';
                loadHistory();
            } else {
                historyList.innerHTML = '<div class="no-history">A free account is required to save results</div>';
            }
        });
}

// 加载历史记录
window.loadHistory = function() {
    fetch('./api/history/get_history.php')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 检查数据是否变化（转换为JSON字符串比较）
                const currentDataStr = JSON.stringify(data.tasks);
                const lastDataStr = JSON.stringify(lastHistoryData);
                
                // 如果数据变化或第一次加载，则渲染
                if (currentDataStr !== lastDataStr) {
                    renderHistory(data.tasks);
                    lastHistoryData = data.tasks; // 更新上次数据
                }
                
                // 检查是否有排队或处理中的任务
                const hasPendingTasks = data.tasks.some(task => 
                    task.status === 'queued' || task.status === 'processing'
                );
                
                // 清除之前的定时器
                if (historyPollingTimer) {
                    clearTimeout(historyPollingTimer);
                    historyPollingTimer = null;
                }
                
                // 如果有待处理任务，则设置定时轮询
                if (hasPendingTasks) {
                    historyPollingTimer = setTimeout(() => {
                        window.loadHistory();
                    }, 5000); // 5秒后再次请求
                }
            } else {
                console.error('获取历史记录失败:', data.message);
            }
        })
        .catch(error => {
            console.error('获取历史记录请求失败:', error);
        });
}

// 渲染历史记录
function renderHistory(tasks) {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = ''; // 清空当前内容

    if (tasks.length === 0) {
        historyList.innerHTML = '<div class="no-history">No analysis history</div>';
        return;
    }

    tasks.forEach(task => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.dataset.taskId = task.task_id; // 存储任务ID
        historyItem.dataset.cancerType = task.cancer_type; // 存储癌种类型

        // 任务名称和日期
        const title = document.createElement('div');
        title.className = 'history-title';
        title.textContent = task.task_name;
		
        const cancerType = document.createElement('div');
        cancerType.className = 'history-cancer';
        cancerType.textContent = `${task.cancer_type}`; 
        // 提交日期
        const date = document.createElement('div');
        date.className = 'history-date';
        date.textContent = `${task.date_display} ${task.time_display}`;
        
        // 文件信息
        const info = document.createElement('div');
        info.className = 'history-info';
        
        const files = document.createElement('span');
        files.className = 'history-files';
        files.textContent = `${task.file_count} file(s)`;
        
        // 状态标签
        const status = document.createElement('span');
        status.className = `history-status status-${task.status}`;
        status.textContent = getStatusText(task.status);
        info.appendChild(cancerType);
        info.appendChild(files);
        info.appendChild(status);
		
        historyItem.appendChild(title);
        
		historyItem.appendChild(date);
        historyItem.appendChild(info);
		

        
        // 将删除按钮添加到历史记录项中
		const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-task-btn';
        deleteBtn.dataset.taskId = task.task_id;
        deleteBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
        historyItem.appendChild(deleteBtn);
		deleteBtn.addEventListener('click', function(e) {
            e.stopPropagation(); // 阻止事件冒泡
            if (confirm('Are you sure you want to delete this task? This cannot be undone!')) {
                deleteTask(task.task_id, historyItem);
            }
        });
		
		
        // 点击事件 - 后续将用于展示结果
        historyItem.addEventListener('click', function() {
            const userInfo = document.getElementById('userInfo');
            if (!userInfo || !userInfo.dataset.userid) {
                alert('请先登录');
                document.getElementById('loginBtn').click();
                return;
            }
            
            // 显示结果区域
            document.getElementById('upload').style.display = 'none';
			document.getElementById('results').style.display = 'block';
            // 滚动到结果区域
            //document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
            
            // 显示任务名称
            //document.querySelector('.results-section .section-title').textContent = `分析结果 - ${task.task_name}`;
            
            // 渲染结果
            renderTaskResult(task.task_id, task.cancer_type);
        });

        historyList.appendChild(historyItem);
    });
}

function deleteTask(taskId, element) {
    fetch(`./api/history/delete_task.php?task_id=${taskId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 删除成功，从DOM中移除
            element.remove();
            
            // 如果没有历史记录了，显示提示
            if (document.querySelectorAll('.history-item').length === 0) {
                document.getElementById('historyList').innerHTML = '<div class="no-history">No analysis history</div>';
            }
        } else {
            alert('删除失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('删除任务出错:', error);
        alert('删除任务出错，请重试');
    });
}



async function renderTaskResult(taskId, cancerType) {
    const resultsSection = document.querySelector('.results-section');
    resultsSection.innerHTML = `
        <div class="loading-section">
            <i class="fas fa-spinner fa-spin"></i>
            <p>Loading result, please wait...</p>
        </div>
    `;
    
    try {
        // 首先获取任务状态
        const statusResponse = await fetch(`./api/history/get_task_status.php?task_id=${taskId}`);
        const statusData = await statusResponse.json();
        
        if (!statusData.success) {
            throw new Error(statusData.message || '获取任务状态失败');
        }
        
        const status = statusData.status;
        const resultJson = statusData.result_json;
        
        // 根据状态进行处理
        if (status === 'completed') {
            // 检查结果是否已存在
            //const existResponse = await fetch(`./api/history/check_results.php?task_id=${taskId}`);
            //const existData = await existResponse.json();
            
            //if (!existData.exists) {
                // 调用R脚本生成结果
                const generateResponse = await fetch(`./api/history/generate_results.php?task_id=${taskId}&cancer_type=${cancerType}`);
                const generateData = await generateResponse.json();
                
                if (!generateData.success) {
                    throw new Error(generateData.message || '结果生成失败');
                }
            //}
            
            // 渲染SVG和TXT
            renderResults(taskId,generateData.errors);
        } else if (status === 'failed') {
            // 显示失败原因
            let errorMessage = 'Analysis Failed';
            if (resultJson) {
                try {
                    const result = JSON.parse(resultJson);
                    errorMessage = result.error || errorMessage;
                } catch (e) {
                    errorMessage = resultJson; // 如果result_json不是JSON，直接显示
                }
            }
            resultsSection.innerHTML = `
                <div class="error-section">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>Analysis Failed:</h3>
                    <p>${errorMessage}</p>
                </div>
            `;
        } else if (status === 'queued') {
            const queueResponse = await fetch(`./api/history/get_queue_position.php?task_id=${taskId}`);
            const queueData = await queueResponse.json();
            
            let queueInfo = 'Your task is queued for analysis.';
            if (queueData.success) {
                queueInfo = `Your task is in position ${queueData.position} of ${queueData.total} in the queue.`;
            }
            
            resultsSection.innerHTML = `
                <div class="info-section">
                    <i class="fas fa-clock"></i>
                    <h3>Task Queued</h3>
                    <p>${queueInfo}</p>
                    <p>Estimated wait time: ${queueData.estimated_wait || 'calculating...'}</p>
                </div>
            `;
        } else if (status === 'processing') {
            resultsSection.innerHTML = `
                <div class="info-section">
                    <i class="fas fa-cog fa-spin"></i>
                    <h3>Analyzing</h3>
                    <p>Your task is being analyzed. Please wait patiently.</p>
                </div>
            `;
        } else {
            throw new Error('未知的任务状态');
        }
    } catch (error) {
        resultsSection.innerHTML = `
            <div class="error-section">
                <i class="fas fa-exclamation-triangle"></i>
                <p>加载任务状态失败: ${error.message}</p>
            </div>
        `;
    }
}

// 渲染最终结果
function renderResults(taskId,iserrors) {
    const resultsSection = document.querySelector('.results-section');
    console.log(iserrors)
	if(iserrors){
		errbtn=`<a href="results/${taskId}/errors.txt" download="errors_${taskId}.txt" class="download-btn  errors-btn">
                    <i class="fas fa-download"></i>Errors
                </a>`
	}else{
		errbtn=""
	}
    // 构建结果HTML
    resultsSection.innerHTML = `
        <div class="result-container">
            <div class="result-header">
                <h3>Analysis Results</h3>  <!-- 在这里添加标题 -->
                <a href="results/${taskId}/exprs.txt" download="exprs_${taskId}.txt" class="download-btn">
                    <i class="fas fa-download"></i>Expression Data
                </a>
				<a href="results/${taskId}/rawexprs.txt" download="rawexprs_${taskId}.txt" class="download-btn">
                    <i class="fas fa-download"></i>Expression Data(raw)
                </a>
				<a href="results/${taskId}/heatmap.pdf" download="heatmap_${taskId}.pdf" class="download-btn">
                    <i class="fas fa-download"></i>PDF
                </a>`+errbtn+`
            </div>
            <div class="result-content">
                <div class="svg-container">
                    <img src="results/${taskId}/heatmap.jpg"/>
                </div>
            </div>
        </div>
    `;
	document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// 状态文本转换
function getStatusText(status) {
    const statusMap = {
        'queued': 'Queued',
        'processing': 'Analyzing',
        'completed': 'Completed',
        'failed': 'Failed'
    };
    return statusMap[status] || status;
}

// 初始化历史记录
document.addEventListener('DOMContentLoaded', function() {
    // 用户登录后加载历史记录
    checkLoginAndLoadHistory();
    
    // 监听任务提交事件
    document.addEventListener('taskSubmitted', function() {
        // 1分钟后重新加载历史记录（给服务器处理时间）
        setTimeout(loadHistory, 60000);
    });
});

document.addEventListener('taskSubmitted', function() {
    // 显示提交成功提示
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '<div class="loading-history">Refreshing...</div>';
    
    // 2秒后重新加载历史记录（给服务器处理时间）
    setTimeout(loadHistory, 500);
});