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

if ($_SERVER['REQUEST_METHOD'] !== 'DELETE') {
    echo json_encode(['success' => false, 'message' => '无效的请求方法']);
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
    // 开启事务
    $pdo->beginTransaction();
    
    // 1. 删除任务记录
    $stmt = $pdo->prepare("DELETE FROM analysis_tasks WHERE task_id = ? AND user_id = ?");
    $stmt->execute([$taskId, $userId]);
    
    // 2. 删除上传文件记录
    $stmt = $pdo->prepare("DELETE FROM uploads WHERE taskuuid = ?");
    $stmt->execute([$taskId]);
    
    // 3. 删除实际文件
    $uploadDir = __DIR__ . '/../../uploads/' . $taskId;
    $resultDir = __DIR__ . '/../../results/' . $taskId;
    
    // 删除上传的文件
    if (is_dir($uploadDir)) {
        array_map('unlink', glob("$uploadDir/*.*"));
        rmdir($uploadDir);
    }
    
    // 删除结果文件
    if (is_dir($resultDir)) {
        array_map('unlink', glob("$resultDir/*.*"));
        rmdir($resultDir);
    }
    
    // 提交事务
    $pdo->commit();
    
    echo json_encode(['success' => true, 'message' => '任务删除成功']);
    
} catch (PDOException $e) {
    $pdo->rollBack();
    echo json_encode(['success' => false, 'message' => '数据库错误: '.$e->getMessage()]);
    
} catch (Exception $e) {
    $pdo->rollBack();
    echo json_encode(['success' => false, 'message' => '文件删除错误: '.$e->getMessage()]);
}
