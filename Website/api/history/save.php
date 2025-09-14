<?php
require_once __DIR__ . '/../../includes/db.php';
require_once __DIR__ . '/../../includes/utils.php';

header('Content-Type: application/json');

$data = json_decode(file_get_contents('php://input'), true);
$userId = $data['userId'];
$files = $data['files']; // 文件URL数组
$ip = getClientIP();

$fileCount = count($files);
$filesJson = json_encode($files);

$db = new DB();
$pdo = $db->getConnection();

try {
    $stmt = $pdo->prepare("INSERT INTO history (user_id, files_json, file_count, ip_address) 
                          VALUES (?, ?, ?, ?)");
    $stmt->execute([$userId, $filesJson, $fileCount, $ip]);
    
    echo json_encode(['success' => true, 'historyId' => $pdo->lastInsertId()]);
} catch (PDOException $e) {
    echo json_encode(['success' => false, 'message' => '保存失败: ' . $e->getMessage()]);
}
