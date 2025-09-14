<?php
require_once __DIR__ . '/PHPMailer/Exception.php';
require_once __DIR__ . '/PHPMailer/PHPMailer.php';
require_once __DIR__ . '/PHPMailer/SMTP.php';
use PHPMailer\PHPMailer\PHPMailer;
use PHPMailer\PHPMailer\Exception;
use PHPMailer\PHPMailer\SMTP;
function getClientIP() {
    if (!empty($_SERVER['HTTP_CLIENT_IP'])) {
        return $_SERVER['HTTP_CLIENT_IP'];
    } elseif (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
        return $_SERVER['HTTP_X_FORWARDED_FOR'];
    } else {
        return $_SERVER['REMOTE_ADDR'];
    }
}

function sendVerificationEmail_demo($email, $code) {
    // 实际实现需要SMTP配置
    $subject = "您的验证码";
    $message = "您的验证码是: $code\n15分钟内有效";
    // mail($email, $subject, $message);
    error_log("验证码发送到 $email: $code"); // 测试时输出到日志
    return true;
}

function sendVerificationEmail($email, $code) {
    $mail = new PHPMailer(true);
    
    try {
        // SMTP配置
        $mail->isSMTP();
        $mail->Host       = SMTP_HOST;
        $mail->SMTPAuth   = true;
        $mail->Username   = SMTP_USERNAME;
        $mail->Password   = SMTP_PASSWORD;
        $mail->SMTPSecure = SMTP_ENCRYPTION;
        $mail->Port       = SMTP_PORT;
        $mail->CharSet    = 'UTF-8';
        // 发件人和收件人
        $mail->setFrom(SMTP_FROM_EMAIL, SMTP_FROM_NAME);
        $mail->addAddress($email);
        // 邮件内容
        $mail->isHTML(true);
        $mail->Subject = '您的ImageDependency验证码';
		$mail->Body    = "感谢您使用ImageDependency!<br>您的验证码是: <b>{$code}</b><br>请在15分钟内使用此验证码完成注册。";
        $mail->AltBody = "您的验证码是: {$code}\n15分钟内有效";
        $mail->send();
        return true;
    } catch (Exception $e) {
        error_log("邮件发送失败: {$mail->ErrorInfo}");
        return false;
    }
}

function generateCode($length = 6) {
    return substr(str_shuffle("0123456789"), 0, $length);
}

function restore_session_if_remembered() {
    if (session_status() === PHP_SESSION_NONE) {
        session_start();
    }
    
    if (!isset($_SESSION['user_id']) && isset($_COOKIE['remember_token'])) {
        $token = $_COOKIE['remember_token'];
        $db = new DB();
        $pdo = $db->getConnection();
        
        $stmt = $pdo->prepare("SELECT id, email FROM users WHERE remember_token = ?");
        $stmt->execute([$token]);
        $user = $stmt->fetch(PDO::FETCH_ASSOC);
        
        if ($user) {
            $_SESSION['user_id'] = $user['id'];
            $_SESSION['email'] = $user['email'];
        }
    }
}
