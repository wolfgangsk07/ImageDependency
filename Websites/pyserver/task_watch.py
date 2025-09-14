import sqlite3
import time
import logging
import traceback
import shutil
import os
from datetime import datetime
from svsToExpr import process_svs_to_expression
import json
# 配置参数
DATABASE_PATH = '/var/www/html/ImageDependency/database/database.db'
UPLOADS_BASE_DIR = '/var/www/html/ImageDependency/uploads/'
POLL_INTERVAL = 5  # 轮询间隔(秒)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_uploaded_files(task_id):
    """从数据库中获取任务的上传文件信息"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT stored_name FROM uploads WHERE taskuuid=?", (task_id,))
    files = [file[0] for file in cursor.fetchall()]
    conn.close()
    return files

def get_orgfilename(task_id,stored_name):
    """从数据库中获取任务的上传文件信息"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT original_name FROM uploads WHERE taskuuid=? AND stored_name=?", (task_id,stored_name))
    
    files = [file[0] for file in cursor.fetchall()]
    conn.close()
    return files[0]
    
def update_task_status(task_id, status, result=None, error=None):
    """更新任务状态和结果"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    update_data = (status, task_id)
    result_str = None
    
    if status == 'processing':
        # 更新为处理中状态
        cursor.execute("""
            UPDATE analysis_tasks 
            SET status=?, started_at=datetime('now')
            WHERE task_id=? AND status IN ('queued', 'processing')
        """, update_data)
    elif status == 'completed' and result:
        # 更新为已完成状态并保存结果
        result_str = json.dumps(result) if result else None
        cursor.execute("""
            UPDATE analysis_tasks 
            SET status=?, completed_at=datetime('now'), result_json=?
            WHERE task_id=?
        """, (status, result_str, task_id))
    elif status == 'failed' and error:
        # 更新为失败状态
        error_info = {"error": error}
        error_str = json.dumps(error_info)
        cursor.execute("""
            UPDATE analysis_tasks 
            SET status=?, completed_at=datetime('now'), result_json=?
            WHERE task_id=?
        """, (status, error_str, task_id))
    
    conn.commit()
    conn.close()


def process_task(task_data):
    """处理单个任务"""
    task_id = task_data['task_id']
    task_name = task_data['task_name']
    cancer_type = task_data['cancer_type']
    
    logging.info(f"开始处理任务: {task_id} ({task_name})")
    task_dir = os.path.join(UPLOADS_BASE_DIR, task_id)
    
    try:
        # 获取任务文件
        files = get_uploaded_files(task_id)
        if not files:
            raise RuntimeError("Can't find any SVS file. Transfer failed？")
        
        # 构建文件路径

        # 检查文件是否存在
        for f in files:
            if not os.path.exists(os.path.join(task_dir,f)):
                raise FileNotFoundError(f"File not exists: {f}")
        
        # 处理文件
        logging.info(f"处理 {len(files)} 个文件...")
        results = []
        OKcount=0
        for f in files:
            org_filename=get_orgfilename(task_id,f)
            try:
                result = process_svs_to_expression(task_dir,f, cancer_type)
                OKcount+=1
            except Exception as e:
                result={"error":str(e)}
                #raise RuntimeError(f"{org_filename}: {str(e)}")
            results.append({"filename":org_filename,"result":result})
        if OKcount==0:
            error_details = "<br><br>".join([
                f"{res['filename']}:<br> {res['result']['error']}" 
                for res in results
            ])
            raise RuntimeError(f"No file completed:<br><br>{error_details}")
        
        # 更新任务状态为"completed"
        update_task_status(task_id, 'completed', result=results)
        logging.info(f"任务完成: {task_id}")
        
        return True
    except Exception as e:
        error_msg = str(e)
        logging.error(error_msg)
        update_task_status(task_id, 'failed', error=error_msg)
        return False
    finally:
        # 清理上传文件
        if os.path.exists(task_dir):
            try:
                shutil.rmtree(task_dir)
                logging.info(f"删除任务目录: {task_dir}")
            except Exception as e:
                logging.error(f"删除目录失败: {str(e)}")

def get_pending_tasks():
    """获取待处理的任务：排队中或进行中"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # 获取最早提交的排队中任务
    cursor.execute("""
        SELECT task_id, user_id, task_name, cancer_type
        FROM analysis_tasks
        WHERE status IN ('queued', 'processing')
        ORDER BY submitted_at ASC
        LIMIT 1
    """)
    
    task = cursor.fetchone()
    conn.close()
    
    if task:
        return {
            'task_id': task[0],
            'user_id': task[1],
            'task_name': task[2],
            'cancer_type': task[3]
        }
    return None

def main():
    """主轮询循环"""
    logging.info("启动任务处理器...")
    
    while True:
        try:
            # 获取待处理任务
            task_data = get_pending_tasks()
            
            if task_data:
                # 标记任务为处理中
                update_task_status(task_data['task_id'], 'processing')
                
                # 处理任务
                process_task(task_data)
            else:
                # 无任务时休眠
                time.sleep(POLL_INTERVAL)
                
        except Exception as e:
            logging.error(f"主循环错误: {str(e)}")
            time.sleep(POLL_INTERVAL)  # 防止错误循环

if __name__ == '__main__':
    main()
