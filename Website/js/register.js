// 修改注册表单HTML
const modal = document.getElementById('authModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalForm = document.getElementById('modalForm');
	const closeBtn = document.querySelector('.close');
document.getElementById('registerBtn').addEventListener('click', () => {
    modalTitle.textContent = 'Create Free Account';
    modalForm.innerHTML = `
		<p class="register-description">Create a free account to securely save your analysis results</p>
        <div class="form-group">
            <label for="regName">Nickname</label>
            <input type="text" id="regName" placeholder="Enter your nickname">
        </div>
        <div class="form-group">
            <label for="regEmail">Email</label>
            <input type="email" id="regEmail" placeholder="Enter your email address">
        </div>
        <div class="form-group">
            <label for="regPassword">Password</label>
            <div class="password-container">
                <input type="password" id="regPassword" placeholder="Set password (at least 8 characters)">
                <i class="fas fa-eye password-toggle"></i>
            </div>
            <div class="password-strength">
                <div class="strength-bar"></div>
                <span class="strength-text">Password strength: weak</span>
            </div>
        </div>
        <div class="form-group">
            <label for="confirmPassword">Confirm password</label>
            <input type="password" id="confirmPassword" placeholder="Re-enter password">
        </div>
        <div class="form-group">
            <button id="sendCodeBtn" class="send-code-btn">Send verification code</button>
            <input type="text" id="verificationCode" placeholder="Enter 6-digit code">
        </div>
        <div class="form-agreement">
            <label>
                <input type="checkbox" id="agreeTerms"> 
                <span class="agreement-text">I have read and agree to the <a href="#" target="_blank">Terms of Service</a> and <a href="#" target="_blank">Privacy Policy</a></span>
            </label>
        </div>
        <button id="submitRegister" class="auth-btn">Register</button>
        <div class="auth-footer">
            Already have an account? <a href="#" id="switchToLogin">Login</a>
        </div>
    `;
    modal.style.display = 'flex';
    
    // 密码强度检测
    document.getElementById('regPassword').addEventListener('input', checkPasswordStrength);
    
    // 密码可见性切换
    document.querySelector('.password-toggle').addEventListener('click', function() {
        const passwordInput = document.getElementById('regPassword');
        if (passwordInput.type === 'password') {
            passwordInput.type = 'text';
            this.classList.replace('fa-eye', 'fa-eye-slash');
        } else {
            passwordInput.type = 'password';
            this.classList.replace('fa-eye-slash', 'fa-eye');
        }
    });
    
    // 发送验证码
    document.getElementById('sendCodeBtn').addEventListener('click', () => {
        const email = document.getElementById('regEmail').value;
        
        // 简单邮箱验证
        if (!validateEmail(email)) {
            showError('Please enter a valid email');
            return;
        }
        
        // 禁用按钮避免重复发送
        const btn = document.getElementById('sendCodeBtn');
        btn.disabled = true;
        
        fetch('./api/users/send_verification.php', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ email })
        })
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                startCountdown();
            } else {
                showError(result.message || 'Failed, retry');
                btn.disabled = false;
            }
        });
    });
    
    // 提交注册
    document.getElementById('submitRegister').addEventListener('click', () => {
        const name = document.getElementById('regName').value;
        const email = document.getElementById('regEmail').value;
        const password = document.getElementById('regPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const code = document.getElementById('verificationCode').value;
        const agreed = document.getElementById('agreeTerms').checked;
        
        // 表单验证
        if (!name) {
            showError('Please enter your name');
            return;
        }
        if (!validateEmail(email)) {
            showError('Please enter a valid email');
            return;
        }
        if (password.length < 8) {
            showError('Password must be at least 8 characters');
            return;
        }
        if (password !== confirmPassword) {
            showError('Passwords do not match');
            return;
        }
        if (!code) {
            showError('Please enter verification code');
            return;
        }
        if (!agreed) {
            showError('Please agree to terms and privacy policy');
            return;
        }
        
        // 提交数据
        const data = {
            name,
            email,
            password,
            code
        };
        
        fetch('./api/users/register.php', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                // 注册成功后自动登录
                modal.style.display = 'none';
                updateUserState({email: email});
            } else {
                showError(result.message || 'Registration failed');
            }
        });
    });
    
    // 切换到登录表单
    document.getElementById('switchToLogin').addEventListener('click', (e) => {
        e.preventDefault();
        document.getElementById('loginBtn').click();
    });
});

// 密码强度检测
function checkPasswordStrength() {
    const password = this.value;
    const strengthBar = document.querySelector('.strength-bar');
    const strengthText = document.querySelector('.strength-text');
    
    // 重置样式
    strengthBar.className = 'strength-bar';
    strengthText.textContent = 'Password strength: ';
    
    if (password.length === 0) {
        return;
    }
    
    let strength = 0;
    if (password.length >= 8) strength += 1;
    if (/[A-Z]/.test(password)) strength += 1;
    if (/[a-z]/.test(password)) strength += 1;
    if (/[0-9]/.test(password)) strength += 1;
    if (/[^A-Za-z0-9]/.test(password)) strength += 1;
    
    let strengthClass = 'weak';
    let text = 'weak';
    
    if (strength >= 4) {
        strengthClass = 'strong';
        text = 'strong';
    } else if (strength >= 3) {
        strengthClass = 'medium';
        text = 'medium';
    }
    
    strengthBar.classList.add(strengthClass);
    strengthText.textContent = `Password strength: ${text}`;
}

// 邮箱格式验证
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(String(email).toLowerCase());
}

// 验证码发送倒计时
function startCountdown() {
    const btn = document.getElementById('sendCodeBtn');
    let count = 60;
    
    btn.textContent = `Send again(${count})`;
    
    const timer = setInterval(() => {
        count--;
        btn.textContent = `Send again(${count})`;
        
        if (count <= 0) {
            clearInterval(timer);
            btn.textContent = '发送验证码';
            btn.disabled = false;
        }
    }, 1000);
}

closeBtn.addEventListener('click', () => modal.style.display = 'none');