<?php
require_once __DIR__ . '/../api/config.php';
class DB {
    private $pdo;
    
    public function __construct() {
        $this->pdo = new PDO('sqlite:' . DB_PATH);
        $this->pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $this->createTables();
    }
    
    private function createTables() {
        $this->pdo->exec("
            CREATE TABLE IF NOT EXISTS users (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				name TEXT NOT NULL,
				email TEXT UNIQUE NOT NULL,
				password TEXT NOT NULL,
				remember_token TEXT,
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				last_login DATETIME,
				last_ip TEXT NOT NULL,
				status INTEGER DEFAULT 1
			);
            
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                files_json TEXT NOT NULL,
                file_count INTEGER NOT NULL,
                ip_address TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            
            CREATE TABLE IF NOT EXISTS verification_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                code TEXT NOT NULL,
                ip TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
			CREATE TABLE IF NOT EXISTS uploads (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				taskuuid TEXT NOT NULL,
				original_name TEXT NOT NULL,
				stored_name TEXT NOT NULL,
				uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY(taskuuid) REFERENCES analysis_tasks(task_id) ON DELETE CASCADE
			);
			 CREATE TABLE IF NOT EXISTS analysis_tasks (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				task_id TEXT NOT NULL,
				user_id INTEGER NOT NULL,
				task_name TEXT NOT NULL,
				cancer_type TEXT NOT NULL, -- 新增癌种字段
				status TEXT DEFAULT 'queued',
				submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				started_at DATETIME,
				completed_at DATETIME,
				result_json TEXT,
				ip_address TEXT NOT NULL,
				FOREIGN KEY(user_id) REFERENCES users(id)
			);
        ");
    }
    
    public function getConnection() {
        return $this->pdo;
    }
}
