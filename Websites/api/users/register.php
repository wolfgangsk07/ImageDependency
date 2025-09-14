<?php
require_once __DIR__ . '/../../includes/db.php';
require_once __DIR__ . '/../../includes/utils.php';

header('Content-Type: application/json');

$db = new DB();
$pdo = $db->getConnection();

$data = json_decode(file_get_contents('php://input'), true);
$name = filter_var($data['name'], FILTER_SANITIZE_STRING);
$email = filter_var($data['email'], FILTER_SANITIZE_EMAIL);
$password = $data['password'];
$code = $data['code'];
$ip = getClientIP();

// 基本验证
if (empty($name) || empty($email) || empty($password) || empty($code)) {
    echo json_encode(['success' => false, 'message' => '所有字段均为必填项']);
    exit;
}

if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
    echo json_encode(['success' => false, 'message' => '无效的邮箱格式']);
    exit;
}

if (strlen($password) < 8) {
    echo json_encode(['success' => false, 'message' => '密码长度至少为8位']);
    exit;
}

// 验证码检查
$stmt = $pdo->prepare("SELECT * FROM verification_codes 
                      WHERE email = ? AND code = ? 
                      AND created_at > datetime('now', '-900 seconds') 
                      ORDER BY created_at DESC LIMIT 1");
$stmt->execute([$email, $code]);
$validCode = $stmt->fetch();

if (!$validCode) {
    echo json_encode(['success' => false, 'message' => '验证码无效或已过期']);
    exit;
}

// 检查邮箱是否已注册
$stmt = $pdo->prepare("SELECT id FROM users WHERE email = ?");
$stmt->execute([$email]);
if ($stmt->fetch()) {
    echo json_encode(['success' => false, 'message' => '该邮箱已注册']);
    exit;
}

// 创建用户
$hashedPassword = password_hash($password, PASSWORD_DEFAULT);
try {
    $stmt = $pdo->prepare("INSERT INTO users (name, email, password, last_ip) VALUES (?, ?, ?, ?)");
    $stmt->execute([$name, $email, $hashedPassword, $ip]);
    
    // 删除使用过的验证码
    $pdo->prepare("DELETE FROM verification_codes WHERE email = ?")->execute([$email]);
    
    echo json_encode(['success' => true, 'userId' => $pdo->lastInsertId()]);
} catch (PDOException $e) {
    echo json_encode(['success' => false, 'message' => '注册失败: ' . $e->getMessage()]);
}
