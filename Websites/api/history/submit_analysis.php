<?php
require_once __DIR__ . '/../../includes/db.php';
require_once __DIR__ . '/../../includes/utils.php';
restore_session_if_remembered();
header('Content-Type: application/json');

// 获取POST数据
$data = json_decode(file_get_contents('php://input'), true);
$userId = $data['user_id'] ?? 0;
$taskId = $data['task_id'] ?? '';
$taskName = $data['task_name'] ?? '';
$cancerType = $data['cancer_type'] ?? 'unknown';

if (empty($taskId) || empty($userId)) {
    echo json_encode(['success' => false, 'message' => '缺少必要参数']);
    exit;
}

$db = new DB();
$pdo = $db->getConnection();
$ip = getClientIP();

try {
	// 获取当前队列中的任务数量（状态为 queued 的任务）
    $stmt = $pdo->prepare("SELECT COUNT(*) as queue_count FROM analysis_tasks WHERE status = 'queued'");
    $stmt->execute();
    $queueCount = $stmt->fetch(PDO::FETCH_ASSOC)['queue_count'];
    
    // 计算队列位置（当前任务将是队列中的下一个）
    $queuePosition = $queueCount + 1;
    // 直接插入任务到数据库（状态为queued）
    $stmt = $pdo->prepare("INSERT INTO analysis_tasks 
                          (task_id, user_id, task_name, cancer_type, status, submitted_at, ip_address) 
                          VALUES (?, ?, ?, ?, ?, datetime('now'), ?)");
    $stmt->execute([
        $taskId, 
        $userId, 
        $taskName, 
        $cancerType,
        'queued',  // 初始状态设置为排队中
        $ip
    ]);
    
    // 返回成功响应（无需调用Flask服务）
    echo json_encode([
        'success' => true,
        'message' => '任务已提交，等待处理',
        'queue_position' => $queuePosition
    ]);
    
} catch (PDOException $e) {
    echo json_encode([
        'success' => false,
        'message' => '数据库操作失败: ' . $e->getMessage()
    ]);
}
