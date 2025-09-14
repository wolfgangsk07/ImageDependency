<?php
include_once '../config.php';
include_once '../../includes/db.php';
include_once '../../includes/utils.php';
restore_session_if_remembered();
header('Content-Type: application/json');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(['success' => false, 'message' => '无效的请求方法']);
    exit;
}

$data = json_decode(file_get_contents('php://input'), true);
$filename = $data['filename'] ?? '';
$taskuuid = $data['taskuuid'] ?? '';

if (empty($filename) || empty($taskuuid)) {
    echo json_encode(['success' => false, 'message' => '参数不完整']);
    exit;
}

$db = new DB();
$pdo = $db->getConnection();

try {
    // 查询数据库获取存储的文件名
    $stmt = $pdo->prepare("SELECT stored_name FROM uploads WHERE taskuuid = ? AND original_name = ?");
    $stmt->execute([$taskuuid, $filename]);
    $file = $stmt->fetch(PDO::FETCH_ASSOC);

    $uploadDir = '../../uploads/' . $taskuuid . '/';
    $fileDeleted = false;
    $recordDeleted = false;

    if ($file) {
        $filePath = $uploadDir . $file['stored_name'];
        // 从数据库中删除记录
        $stmt = $pdo->prepare("DELETE FROM uploads WHERE taskuuid = ? AND original_name = ?");
        $stmt->execute([$taskuuid, $filename]);
        $recordDeleted = true;

        // 如果文件存在，则删除文件
        if (file_exists($filePath)) {
            if (unlink($filePath)) {
                $fileDeleted = true;
            }
        } else {
            // 文件不存在，但仍然视为成功
            $fileDeleted = true;
        }
    } else {
        // 没有记录但仍然视为成功
        $recordDeleted = true;
        $fileDeleted = true;
    }

    // 总是返回成功，即使文件不存在
    echo json_encode([
        'success' => true,
        'message' => '删除操作完成',
        'file_deleted' => $fileDeleted,
        'record_deleted' => $recordDeleted
    ]);
    
} catch (PDOException $e) {
    echo json_encode(['success' => false, 'message' => '数据库错误: '.$e->getMessage()]);
}
