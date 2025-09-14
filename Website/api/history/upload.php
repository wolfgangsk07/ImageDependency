<?php
require_once __DIR__ . '/../../includes/db.php';
require_once __DIR__ . '/../../includes/utils.php';
restore_session_if_remembered();
header('Content-Type: application/json');

$taskuuid = $_POST['taskuuid'] ?? '';
$fileIndex = $_POST['fileIndex'] ?? 0;
$totalFiles = $_POST['totalFiles'] ?? 1;

if (empty($taskuuid)) {
    echo json_encode(['success' => false, 'message' => '缺少任务ID']);
    exit;
}

if (!isset($_FILES['file'])) {
    echo json_encode(['success' => false, 'message' => '未接收到文件']);
    exit;
}

$file = $_FILES['file'];
$originalName = $file['name'];
$extension = pathinfo($originalName, PATHINFO_EXTENSION);
$storedName = uniqid() . '.' . $extension;
$uploadDir = __DIR__ . '/../../uploads/' . $taskuuid . '/';

if (!is_dir($uploadDir)) {
    mkdir($uploadDir, 0777, true);
}

$targetPath = $uploadDir . $storedName;

if (move_uploaded_file($file['tmp_name'], $targetPath)) {
    $db = new DB();
    $pdo = $db->getConnection();
    
    try {
        $stmt = $pdo->prepare("INSERT INTO uploads (taskuuid, original_name, stored_name) VALUES (?, ?, ?)");
        $stmt->execute([$taskuuid, $originalName, $storedName]);
        
        echo json_encode([
            'success' => true,
            'originalName' => $originalName,
            'storedName' => $storedName,
            'fileIndex' => $fileIndex,
            'totalFiles' => $totalFiles
        ]);
    } catch (PDOException $e) {
        error_log("Database error: " . $e->getMessage());
        echo json_encode(['success' => false, 'message' => '数据库错误']);
    }
} else {
    echo json_encode(['success' => false, 'message' => '文件上传失败']);
}
