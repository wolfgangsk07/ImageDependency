<?php
define('DB_PATH', __DIR__ . '/../database/database.db');
define('MAX_LOGIN_ATTEMPTS', 5);
define('VERIFICATION_EXPIRY', 900); // 15分钟



// SMTP配置 (根据您的邮件服务提供商修改)
define('SMTP_HOST', 'smtp.qq.com'); // SMTP服务器
define('SMTP_PORT', 465); // SMTP端口
define('SMTP_USERNAME', '415742258@qq.com'); // 邮箱账号
define('SMTP_PASSWORD', 'kwwectgfqystbihd'); // 邮箱密码或授权码
define('SMTP_ENCRYPTION', 'ssl'); // 加密方式 (tls/ssl)
define('SMTP_FROM_EMAIL', '415742258@qq.com'); // 发件人邮箱
define('SMTP_FROM_NAME', 'ImageDependency 系统'); // 发件人名称