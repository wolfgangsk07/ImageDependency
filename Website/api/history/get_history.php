<?php
require_once __DIR__ . '/../../includes/db.php';
require_once __DIR__ . '/../../includes/utils.php';
restore_session_if_remembered();
session_start();
header('Content-Type: application/json');

if (!isset($_SESSION['user_id'])) {
    echo json_encode(['success' => false, 'message' => '未登录']);
    exit;
}

$userId = $_SESSION['user_id'];
$db = new DB();
$pdo = $db->getConnection();

try {
    $stmt = $pdo->prepare("
        SELECT 
            t.task_id, 
            t.task_name, 
            t.submitted_at,
            t.status,
            t.cancer_type,
            COUNT(u.id) AS file_count
        FROM analysis_tasks t
        LEFT JOIN uploads u ON t.task_id = u.taskuuid
        WHERE t.user_id = ?
        GROUP BY t.task_id
        ORDER BY t.submitted_at DESC
    ");
    $stmt->execute([$userId]);
    $tasks = $stmt->fetchAll(PDO::FETCH_ASSOC);
    
    // 格式化日期
    foreach ($tasks as &$task) {
        $dateObj = new DateTime($task['submitted_at']);
        $task['date_display'] = $dateObj->format('Y年m月d日');
        $task['time_display'] = $dateObj->format('H:i');
    }
    
    echo json_encode(['success' => true, 'tasks' => $tasks]);
} catch (PDOException $e) {
    echo json_encode(['success' => false, 'message' => '数据库错误: '.$e->getMessage()]);
}
