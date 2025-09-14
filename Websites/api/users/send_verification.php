<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);
require_once __DIR__ . '/../../includes/utils.php';
require_once __DIR__ . '/../../includes/db.php';

header('Content-Type: application/json');

$data = json_decode(file_get_contents('php://input'), true);
$email = filter_var($data['email'], FILTER_SANITIZE_EMAIL);
$ip = getClientIP();

if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
    echo json_encode(['success' => false, 'message' => 'Invalid e-mail']);
    exit;
}

$db = new DB();
$pdo = $db->getConnection();

// 检查最近是否已发送过验证码
$stmt = $pdo->prepare("SELECT * FROM verification_codes 
                      WHERE email = ? AND created_at > datetime('now', '-60 seconds') 
                      ORDER BY created_at DESC LIMIT 1");
$stmt->execute([$email]);
$recentCode = $stmt->fetch();

if ($recentCode) {
    echo json_encode(['success' => false, 'message' => 'Please retry after 60 seconds.']);
    exit;
}

// 生成并保存验证码
$code = generateCode();
try {
    $stmt = $pdo->prepare("INSERT INTO verification_codes (email, code, ip) VALUES (?, ?, ?)");
    $stmt->execute([$email, $code, $ip]);
    
    // 发送邮件
    if (sendVerificationEmail($email, $code)) {
        echo json_encode(['success' => true]);
    } else {
        echo json_encode(['success' => false, 'message' => 'Failed, retry later.']);
    }
} catch (PDOException $e) {
    echo json_encode(['success' => false, 'message' => 'Failed: ' . $e->getMessage()]);
}
