<?php
require_once __DIR__ . '/../../includes/db.php';
require_once __DIR__ . '/../../includes/utils.php';
session_start();
restore_session_if_remembered();
header('Content-Type: application/json');

if (!isset($_SESSION['user_id'])) {
    echo json_encode(['success' => false, 'message' => '未登录']);
    exit;
}

$taskId = $_GET['task_id'] ?? '';

if (empty($taskId)) {
    echo json_encode(['success' => false, 'message' => '缺少任务ID']);
    exit;
}

$userId = $_SESSION['user_id'];
$db = new DB();
$pdo = $db->getConnection();

try {
    // 获取当前任务的位置
    $stmt = $pdo->prepare("
        SELECT COUNT(*) as position 
        FROM analysis_tasks 
        WHERE status = 'queued' 
        AND submitted_at <= (SELECT submitted_at FROM analysis_tasks WHERE task_id = ?)
    ");
    $stmt->execute([$taskId]);
    $position = $stmt->fetch(PDO::FETCH_ASSOC)['position'];
    
    // 获取队列中的总任务数
    $stmt = $pdo->prepare("SELECT COUNT(*) as total FROM analysis_tasks WHERE status = 'queued'");
    $stmt->execute();
    $total = $stmt->fetch(PDO::FETCH_ASSOC)['total'];
    
    // 简单估算等待时间（假设每个任务平均处理5分钟）
    $estimatedWait = $position * 5;
    
    echo json_encode([
        'success' => true,
        'position' => $position,
        'total' => $total,
        'estimated_wait' => $estimatedWait . ' minutes'
    ]);
    
} catch (PDOException $e) {
    echo json_encode(['success' => false, 'message' => '数据库错误: '.$e->getMessage()]);
}
