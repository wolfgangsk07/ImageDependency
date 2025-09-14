<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);
require_once __DIR__ . '/../../includes/db.php';
require_once __DIR__ . '/../../includes/utils.php';

header('Content-Type: application/json');

session_start();

$data = json_decode(file_get_contents('php://input'), true);
$email = filter_var($data['email'], FILTER_SANITIZE_EMAIL);
$password = $data['password'];
$remember = $data['remember'] ?? false;
$ip = getClientIP();

$db = new DB();
$pdo = $db->getConnection();

// 获取用户
$stmt = $pdo->prepare("SELECT * FROM users WHERE email = ?");
$stmt->execute([$email]);
$user = $stmt->fetch(PDO::FETCH_ASSOC);

if (!$user) {
    echo json_encode(['success' => false, 'message' => '用户不存在']);
    exit;
}

if (!password_verify($password, $user['password'])) {
    echo json_encode(['success' => false, 'message' => '密码不正确']);
    exit;
}

// 登录成功
$_SESSION['user_id'] = $user['id'];
$_SESSION['email'] = $user['email'];

if ($remember) {
    // 创建记住我token (简化实现)
    $token = bin2hex(random_bytes(32));
    $expiry = time() + 60 * 60 * 24 * 30; // 30天
    
    setcookie('remember_token', $token, $expiry, '/');
    
    // 存储token到数据库 (实际实现需要更安全)
    $stmt = $pdo->prepare("UPDATE users SET remember_token = ? WHERE id = ?");
    $stmt->execute([$token, $user['id']]);
}

echo json_encode([
    'success' => true,
    'user' => [
        'id' => $user['id'],
        'email' => $user['email']
    ]
]);
