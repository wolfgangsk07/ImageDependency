function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
// Tab切换功能
let selectedCancer = null;
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                // 移除所有激活状态
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // 设置当前激活状态
                tab.classList.add('active');
                const tabId = tab.getAttribute('data-tab');
                document.getElementById(`${tabId}-content`).classList.add('active');
            });
        });
        
        // 文件上传区域交互效果
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'rgba(52, 152, 219, 0.15)';
            uploadArea.style.borderColor = '#e74c3c';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.backgroundColor = 'rgba(52, 152, 219, 0.05)';
            uploadArea.style.borderColor = '#3498db';
        });
        
        uploadArea.addEventListener('click', () => {
            // 模拟文件选择
            uploadArea.style.backgroundColor = 'rgba(52, 152, 219, 0.1)';
            setTimeout(() => {
                uploadArea.style.backgroundColor = 'rgba(52, 152, 219, 0.05)';
            }, 300);
        });
        
        // 平滑滚动到锚点
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    window.scrollTo({
                        top: target.offsetTop - 80,
                        behavior: 'smooth'
                    });
                }
            });
        });
document.addEventListener('DOMContentLoaded', function() {
    // 生成并设置任务UUID
    const taskContainer = document.getElementById('taskContainer');
    const taskuuid = generateUUID();
    taskContainer.setAttribute('taskuuid', taskuuid);
    
    const fileInput = document.getElementById('fileInput');
    const selectFilesBtn = document.getElementById('selectFilesBtn');
    const fileList = document.getElementById('fileList');
    const uploadArea = document.getElementById('uploadArea');
    const submitBtn = document.querySelector('.submit-btn');
     // 打开癌种选择模态框
    //document.getElementById('selectCancerBtn').addEventListener('click', function() {
    //    document.getElementById('cancerModal').style.display = 'flex';
    //});
    
    // 关闭癌种选择模态框
    //document.querySelector('.close-cancer').addEventListener('click', function() {
    //    document.getElementById('cancerModal').style.display = 'none';
    //});
    
	/*
    // 选择癌种
    const cancerItems = document.querySelectorAll('.cancer-item');
    cancerItems.forEach(item => {
        item.addEventListener('click', function() {
            selectedCancer = this.dataset.cancer;
            document.getElementById('selectedCancerText').textContent = this.textContent;
            document.getElementById('cancerType').dataset.cancer = selectedCancer;
            document.getElementById('cancerModal').style.display = 'none';
        });
    });
	*/
	
	const cancerSelect = document.getElementById('cancerSelect');
	let selectedCancer = null;
	cancerSelect.addEventListener('change', function() {
		selectedCancer = this.value;
		//document.getElementById('selectedCancerText').textContent = 
		//	selectedCancer ? cancerSelect.options[cancerSelect.selectedIndex].text : '未选择';
	});
	
    // 点击选择文件按钮
    selectFilesBtn.addEventListener('click', () => {
        fileInput.click();
    });
    
    // 拖放功能
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFiles(fileInput.files);
        }
    });
    
    // 文件选择变化
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFiles(fileInput.files);
        }
    });
    
    // 处理文件上传
    function handleFiles(files) {
		const emptyHint = fileList.querySelector('.empty-hint');
		if (emptyHint) {
			emptyHint.style.display = 'none';
		}
		const totalFiles = files.length;
		
		for (let i = 0; i < totalFiles; i++) {
			const file = files[i];
			const fileItem = document.createElement('div');
			fileItem.className = 'file-item';
			fileItem.dataset.fileName = file.name;
			fileItem.innerHTML = `
				<div class="file-info">
					<span class="file-name">${file.name}</span>
					<span class="file-size">${formatFileSize(file.size)}</span>
				</div>
				<div class="progress-container">
					<div class="progress-bar" style="width: 0%"></div>
					<span class="progress-text">0%</span>
				</div>
				<button class="delete-file" data-filename="${file.name}">Delete</button>
			`;
			fileList.appendChild(fileItem);
			
			// 添加删除按钮事件
			const deleteBtn = fileItem.querySelector('.delete-file');
			deleteBtn.addEventListener('click', function() {
				const fileName = this.dataset.filename;
				deleteFile(fileName, fileItem);
			});
			
			uploadFile(file, i, totalFiles, fileItem);
		}
	}
	

	
    function deleteFile(fileName, fileItem) {
		// 中止上传请求（如果正在上传）
		if (fileItem.xhr) {
			fileItem.xhr.abort();
		}

		// 从DOM中移除文件条目
		fileItem.remove();

		// 检查文件列表是否为空
		const fileItems = fileList.querySelectorAll('.file-item');
		if (fileItems.length === 0) {
			showEmptyHint();
		}
		fileInput.value = '';
		// 发送删除请求到后端
		fetch('./api/history/delete_file.php', {
			method: 'POST',
			headers: {'Content-Type': 'application/json'},
			body: JSON.stringify({
				filename: fileName,
				taskuuid: document.getElementById('taskContainer').getAttribute('taskuuid')
			})
		})
		.then(response => response.json())
		.then(result => {
			if (!result.success) {
				console.error('删除文件失败:', result.message);
			}
		})
		.catch(error => {
			console.error('删除文件请求失败:', error);
		});
	}

	
    // 上传单个文件
    function uploadFile(file, index, totalFiles, fileItem) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('taskuuid', document.getElementById('taskContainer').getAttribute('taskuuid'));
        formData.append('fileIndex', index);
        formData.append('totalFiles', totalFiles);
        
        const progressBar = fileItem.querySelector('.progress-bar');
        const progressText = fileItem.querySelector('.progress-text');
        
        const xhr = new XMLHttpRequest();
        fileItem.xhr = xhr; // 保存xhr引用以便删除时中止
		
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                progressBar.style.width = `${percent}%`;
				if(percent==100){
					progressText.textContent = 'Waiting';
				}else{
					progressText.textContent = `${percent}%`;
				}
            }
        });
        
        xhr.addEventListener('load', () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                if (response.success) {
                    progressBar.style.backgroundColor = '#2ecc71';
                    progressText.textContent = 'Completed';
                } else {
                    progressBar.style.backgroundColor = '#e74c3c';
                    progressText.textContent = 'Failed';
                    fileItem.querySelector('.file-name').innerHTML += 
                        `<span class="error-msg"> - ${response.message || '上传错误'}</span>`;
                }
				fileInput.value = '';
            } else {
                progressBar.style.backgroundColor = '#e74c3c';
                progressText.textContent = 'Failed';
                fileItem.querySelector('.file-name').innerHTML += 
                    '<span class="error-msg"> - 服务器错误</span>';
            }
        });
        
        xhr.addEventListener('error', () => {
            progressBar.style.backgroundColor = '#e74c3c';
            progressText.textContent = 'Failed';
            fileItem.querySelector('.file-name').innerHTML += 
                '<span class="error-msg"> - 网络错误</span>';
        });
        
        xhr.open('POST', './api/history/upload.php', true);
        xhr.send(formData);
    }
    
    // 文件大小格式化
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    

	submitBtn.addEventListener('click', async (e) => {
		e.preventDefault();
		if (!selectedCancer) {
			alert('Please select cancer type first');
			return;
		}
		const userInfo = document.getElementById('userInfo');
		
		// 检查用户是否登录
		if (!userInfo || !userInfo.dataset.userid) {
			// 显示登录模态框而不是简单的alert
			document.getElementById('loginBtn').click();
			return;
		}
		
		const userId = parseInt(userInfo.dataset.userid);
		
		const taskId = taskContainer.getAttribute('taskuuid');
		const taskName = document.getElementById('taskName').value || 'New task';
		
		// 检查是否有文件上传
		const fileItems = document.querySelectorAll('.file-item');
		if (fileItems.length === 0) {
			alert('Please upload at least one file');
			return;
		}
		
		// 检查所有文件是否上传完成
		let allFilesUploaded = true;
		let errorMessage = '';
		
		fileItems.forEach(item => {
			const progressText = item.querySelector('.progress-text');
			const errorMsg = item.querySelector('.error-msg');
			
			if (errorMsg) {
				allFilesUploaded = false;
				errorMessage = `File ${item.dataset.fileName} failed to upload`;
				return;
			}
			
			if (progressText.textContent !== 'Completed') {
				allFilesUploaded = false;
				errorMessage = 'Please wait for all files to finish uploading';
			}
		});
		
		if (!allFilesUploaded) {
			alert(errorMessage || 'Please wait for all files to finish uploading');
			return;
		}
		
		// 显示加载状态
		submitBtn.disabled = true;
		submitBtn.textContent = 'Submitting...';
		
		try {
			const response = await fetch('./api/history/submit_analysis.php', {
				method: 'POST',
				headers: {'Content-Type': 'application/json'},
				body: JSON.stringify({
					user_id: parseInt(userId),
					task_id: taskId,
					task_name: taskName,
					cancer_type: selectedCancer  // 添加癌种
				})
			});
			
			const result = await response.json();
			
			if (response.ok) {
				alert(`Task submitted, queue position: ${result.queue_position}`);
				
				// 创建自定义事件通知任务提交
				const taskSubmittedEvent = new CustomEvent('taskSubmitted');
				document.dispatchEvent(taskSubmittedEvent);
				console.log(1)
				// 重置表单
				resetUploadForm();
				console.log(2)
				// 不显示结果区域，保持在上传界面
				// document.getElementById('upload').style.display = 'none';
				// document.getElementById('results').style.display = 'block';
			} else {
				alert(`提交失败: ${result.error || result.message || '未知错误'}`);
			}
		} catch (error) {
			alert(`请求失败: ${error.message}`);
		} finally {
			submitBtn.disabled = false;
			submitBtn.textContent = 'Start Analysis';
		}
	});
	
	document.getElementById('newTaskBtn').addEventListener('click', function() {
		// 显示上传区域，隐藏结果区域
		document.getElementById('upload').style.display = 'block';
		document.getElementById('results').style.display = 'none';
		
		// 重置表单
		resetUploadForm();
	});

});

function resetUploadForm() {
    // 清空文件列表
    document.getElementById('fileList').innerHTML = '';
    
    // 重置任务名称
    document.getElementById('taskName').value = '';
    
    // 重置癌种选择
	document.getElementById('cancerSelect').value=""
    selectedCancer = null;
    document.getElementById('cancerType').dataset.cancer = '';
    
    // 生成新的任务UUID
    const taskContainer = document.getElementById('taskContainer');
    const taskuuid = generateUUID();
    taskContainer.setAttribute('taskuuid', taskuuid);
	showEmptyHint();
}

function showEmptyHint() {
    let emptyHint = fileList.querySelector('.empty-hint');
    if (!emptyHint) {
        emptyHint = document.createElement('div');
        emptyHint.className = 'empty-hint';
        emptyHint.textContent = 'No SVS files uploaded yet';
        fileList.appendChild(emptyHint);
    }
    emptyHint.style.display = 'block';
}