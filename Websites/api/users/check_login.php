<?php
require_once __DIR__ . '/../../includes/db.php';
require_once __DIR__ . '/../../includes/utils.php';
session_start();
restore_session_if_remembered();
header('Content-Type: application/json');

$response = ['isLoggedIn' => false];

if (isset($_SESSION['user_id'])) {
    $db = new DB();
    $pdo = $db->getConnection();
    
    $stmt = $pdo->prepare("SELECT id, email FROM users WHERE id = ?");
    $stmt->execute([$_SESSION['user_id']]);
    $user = $stmt->fetch(PDO::FETCH_ASSOC);
    
    if ($user) {
        $response = [
            'isLoggedIn' => true,
            'user' => [
                'id' => $user['id'],
                'email' => $user['email']
            ]
        ];
    }
}

// 检查记住我功能
if (!$response['isLoggedIn'] && isset($_COOKIE['remember_token'])) {
    $token = $_COOKIE['remember_token'];
    $db = new DB();
    $pdo = $db->getConnection();
    
    $stmt = $pdo->prepare("SELECT id, email FROM users WHERE remember_token = ?");
    $stmt->execute([$token]);
    $user = $stmt->fetch(PDO::FETCH_ASSOC);
    
    if ($user) {
        // 重新生成session
        $_SESSION['user_id'] = $user['id'];
        $_SESSION['email'] = $user['email'];
        
        $response = [
            'isLoggedIn' => true,
            'user' => [
				'id' => $user['id'], // 增加用户ID
				'email' => $user['email']
			]
        ];
    }
}

echo json_encode($response);
