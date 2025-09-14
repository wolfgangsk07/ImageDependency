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
    $stmt = $pdo->prepare("
        SELECT status, result_json 
        FROM analysis_tasks 
        WHERE task_id = ? AND user_id = ?
    ");
    $stmt->execute([$taskId, $userId]);
    $task = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($task) {
		if($task['status'] == "failed"){
			echo json_encode([
				'success' => true,
				'status' => $task['status'],
				'result_json' => $task['result_json'] // 如果是失败，这里会有错误信息
			]);
		}else{
			echo json_encode([
				'success' => true,
				'status' => $task['status'],
				'result_json' => "" // 如果是失败，这里会有错误信息
			]);

		}
    } else {
        echo json_encode(['success' => false, 'message' => '任务不存在']);
    }
} catch (PDOException $e) {
    echo json_encode(['success' => false, 'message' => '数据库错误: '.$e->getMessage()]);
}