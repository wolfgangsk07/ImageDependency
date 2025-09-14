// 在register.js中添加登录功能

document.getElementById('loginBtn').addEventListener('click', () => {
    modalTitle.textContent = 'Login';
    modalForm.innerHTML = `
        <div class="form-group">
            <label for="loginEmail">Email</label>
            <input type="email" id="loginEmail" placeholder="Enter your registered email">
        </div>
        <div class="form-group">
            <label for="loginPassword">Password</label>
            <div class="password-container">
                <input type="password" id="loginPassword" placeholder="Enter your password">
                <i class="fas fa-eye password-toggle"></i>
            </div>
        </div>
        <div class="form-options">
            <label>
                <input type="checkbox" id="rememberMe"> Remember me
            </label>
            <a href="#" class="forgot-password">Forgot password?</a>
        </div>
        <button id="submitLogin" class="auth-btn">Login</button>
        <div class="auth-footer">
            Don't have an account? <a href="#" id="switchToRegister">Register now</a>
        </div>
    `;
    modal.style.display = 'flex';
    
    // 密码可见性切换
    document.querySelector('.password-toggle').addEventListener('click', function() {
        const passwordInput = document.getElementById('loginPassword');
        if (passwordInput.type === 'password') {
            passwordInput.type = 'text';
            this.classList.replace('fa-eye', 'fa-eye-slash');
        } else {
            passwordInput.type = 'password';
            this.classList.replace('fa-eye-slash', 'fa-eye');
        }
    });
    
    // 提交登录
    document.getElementById('submitLogin').addEventListener('click', () => {
        const data = {
            email: document.getElementById('loginEmail').value,
            password: document.getElementById('loginPassword').value,
            remember: document.getElementById('rememberMe').checked
        };
        
        fetch('./api/users/login.php', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                // 登录成功处理
                modal.style.display = 'none';
                updateUserState({
					id: result.user.id,
					email: result.user.email
				});
				if (window.loadHistory) {
					const historyList = document.getElementById('historyList');
					historyList.innerHTML = '<div class="loading-history">Loading history...</div>';
					window.loadHistory();
				}
            } else {
                showError(result.message);
            }
        });
    });
    
    // 切换到注册表单
    document.getElementById('switchToRegister').addEventListener('click', (e) => {
        e.preventDefault();
        document.getElementById('registerBtn').click();
    });
});

// 更新用户状态显示
function updateUserState(user) {
    const userAuthDiv = document.querySelector('.user-auth');
    userAuthDiv.innerHTML = `
        <span class="welcome-msg">Welcome, ${user.email}</span>
        <button id="logoutBtn">Logout</button>
    `;
    
	// 添加隐藏的用户ID存储
    const userInfoSpan = document.createElement('span');
    userInfoSpan.id = 'userInfo';
    userInfoSpan.style.display = 'none';
    userInfoSpan.dataset.userid = user.id;
    userAuthDiv.prepend(userInfoSpan);
	
    // 添加退出功能
    document.getElementById('logoutBtn').addEventListener('click', logoutUser);
	window.loadHistory = loadHistory;
}

// 用户退出函数
function logoutUser() {
	if (historyPollingTimer) {
        clearTimeout(historyPollingTimer);
        historyPollingTimer = null;
    }
    fetch('./api/users/logout.php')
    .then(() => {
        const userAuthDiv = document.querySelector('.user-auth');
        userAuthDiv.innerHTML = `
            <button id="loginBtn">Login</button>
            <button id="registerBtn">Register</button>
        `;
        
        // 重新绑定事件
        document.getElementById('loginBtn').addEventListener('click', () => {
            document.getElementById('loginBtn').dispatchEvent(new Event('click'));
        });
        document.getElementById('registerBtn').addEventListener('click', () => {
            document.getElementById('registerBtn').dispatchEvent(new Event('click'));
        });
		const historyList = document.getElementById('historyList');
		historyList.innerHTML = '<div class="no-history">Please login to view history</div>';
    });
}

// 显示错误消息
function showError(message) {
    let errorDiv = document.querySelector('.error-message');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        modalForm.prepend(errorDiv);
    }
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}
