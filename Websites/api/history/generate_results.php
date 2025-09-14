<?php
require_once __DIR__ . '/../../includes/utils.php';
restore_session_if_remembered();
header('Content-Type: application/json');

$taskId = $_GET['task_id'] ?? '';
$cancerType = $_GET['cancer_type'] ?? 'BRCA';

if (empty($taskId)) {
    echo json_encode(['success' => false, 'message' => '缺少任务ID']);
    exit;
}

$resultDir = __DIR__ . '/../../results/' . $taskId;

// 创建结果目录
if (!is_dir($resultDir)) {
    mkdir($resultDir, 0777, true);
}

// 调用R脚本生成结果
$output = [];  // 存储命令输出
$returnCode = 0;  // 存储返回码

// 执行命令并捕获输出
$svgExists = file_exists($resultDir . '/heatmap.jpg');
$txtExists = file_exists($resultDir . '/exprs.txt');
    
if (!$svgExists || !$txtExists) {
	$command = "Rscript ../../R/heatmap.R {$taskId} {$cancerType} 2>&1";
	exec($command, $output, $returnCode);
}
if ($returnCode === 0) {
    // 检查生成的文件
    $svgExists = file_exists($resultDir . '/heatmap.jpg');
    $txtExists = file_exists($resultDir . '/exprs.txt');
    $errExists = file_exists($resultDir . '/errors.txt');
    if ($svgExists && $txtExists) {
		if($errExists){
			echo json_encode([
				'success' => true,
				'errors' => true
			]);
		}else{
			echo json_encode(['success' => true]);
		}
    } else {
        // 返回错误及R脚本输出
        echo json_encode([
            'success' => false,
            'message' => '结果文件生成不完整',
            'output' => implode("\n", $output),
            'jpg' => $svgExists,
            'txt' => $txtExists
        ]);
    }
} else {
    // 返回错误及R脚本输出
    echo json_encode([
        'success' => false,
        'message' => implode("\n", $output),
        'return_code' => $returnCode
    ]);
}
