#!/usr/bin/env python3
"""
PCB焊点检测系统 - Flask Web GUI
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
import sys
import subprocess
import threading
import time
import json
import shutil
import re
from datetime import datetime
import logging
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 修正到项目根目录
sys.path.append(os.path.dirname(current_dir))  # 添加src目录到路径

# 导入数据库管理器
from database_manager import PCBReportDatabase

# 导入分析数据解析器
from analysis_parser import parse_all_analysis_data, parse_analysis_file

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pcb_detection_system_2025'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# 配置路径
PROJECT_ROOT = project_root
DATA_USERS_DIR = os.path.join(PROJECT_ROOT, 'data', 'users')
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp')
TEMP1_DIR = os.path.join(PROJECT_ROOT, 'temp1')
TEMP1_PLUS_DIR = os.path.join(PROJECT_ROOT, 'temp1', 'plus')  # UNet++分析结果目录
MY_TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'temp', 'my_test_results')  # YOLO结果目录
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, 'temp', 'visualization')
UPLOAD_FOLDER = DATA_USERS_DIR

# 确保目录存在
os.makedirs(DATA_USERS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TEMP1_DIR, exist_ok=True)
os.makedirs(TEMP1_PLUS_DIR, exist_ok=True)
os.makedirs(MY_TEST_RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# 全局变量存储日志和任务状态
system_logs = []
current_tasks = {}

# 初始化数据库
db_path = os.path.join(PROJECT_ROOT, 'reports.db')
report_db = PCBReportDatabase(db_path)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_message(message, level="INFO"):
    """添加日志消息"""
    global system_logs
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] [{level}] {message}"
    system_logs.append(log_entry)
    # 保持最新的100条日志
    if len(system_logs) > 100:
        system_logs = system_logs[-100:]
    print(log_entry)

def run_script_with_logging(script_path, script_name, task_id):
    """在后台运行脚本并记录日志"""
    global current_tasks
    
    try:
        log_message(f"=== 执行脚本开始 ===", "INFO")
        log_message(f"脚本名称: {script_name}", "INFO")
        log_message(f"脚本路径: {script_path}", "INFO")
        log_message(f"任务ID: {task_id}", "INFO")
        log_message(f"当前工作目录: {os.getcwd()}", "INFO")
        
        current_tasks[task_id] = {"status": "running", "start_time": time.time()}
        
        # 改变到脚本所在目录
        script_dir = os.path.dirname(script_path)
        log_message(f"切换到脚本目录: {script_dir}", "INFO")
        
        # 设置环境变量以支持UTF-8编码
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
        
        log_message(f"启动进程: {sys.executable} {script_path}", "INFO")
        
        # 运行脚本
        process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        log_message(f"进程启动成功，PID: {process.pid}", "INFO")
        
        # 实时读取输出
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_line = output.strip()
                output_lines.append(output_line)
                log_message(f"[{script_name}] {output_line}", "SCRIPT")
        
        return_code = process.poll()
        execution_time = time.time() - current_tasks[task_id]["start_time"]
        
        log_message(f"=== 脚本执行完成 ===", "INFO")
        log_message(f"返回码: {return_code}", "INFO")
        log_message(f"执行时间: {execution_time:.2f}秒", "INFO")
        log_message(f"输出行数: {len(output_lines)}", "INFO")
        
        if return_code == 0:
            log_message(f"{script_name} 执行成功!", "INFO")
            current_tasks[task_id] = {"status": "completed", "end_time": time.time(), "execution_time": execution_time}
        else:
            log_message(f"{script_name} 执行失败，返回码: {return_code}", "ERROR")
            current_tasks[task_id] = {"status": "failed", "end_time": time.time(), "execution_time": execution_time}
            
    except Exception as e:
        execution_time = time.time() - current_tasks.get(task_id, {}).get("start_time", time.time())
        log_message(f"{script_name} 执行出错: {str(e)}", "ERROR")
        current_tasks[task_id] = {"status": "error", "end_time": time.time(), "execution_time": execution_time}

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/logs', methods=['POST'])
def add_log():
    """添加日志条目"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        level = data.get('level', 'INFO')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        # 格式化日志消息
        formatted_log = f"[{timestamp}] [{level}] [前端] {message}"
        system_logs.append(formatted_log)
        
        # 保持日志列表不超过1000条
        if len(system_logs) > 1000:
            system_logs.pop(0)
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs')
def get_logs():
    """获取系统日志"""
    return jsonify({"logs": system_logs[-50:]})  # 返回最新的50条日志

@app.route('/api/config/preprocessing', methods=['GET', 'POST'])
def preprocessing_config():
    """预处理配置"""
    preprocessor_path = os.path.join(PROJECT_ROOT, 'src', '1-preprocessing', 'preprocessor.py')
    
    if request.method == 'GET':
        # 读取当前配置
        try:
            with open(preprocessor_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 查找new_size参数
                import re
                match = re.search(r'new_size=\((\d+),\s*(\d+)\)', content)
                if match:
                    width, height = match.groups()
                    return jsonify({"width": int(width), "height": int(height)})
                else:
                    return jsonify({"width": 512, "height": 512})
        except Exception as e:
            log_message(f"读取预处理配置失败: {str(e)}", "ERROR")
            return jsonify({"error": str(e)}), 500
    
    elif request.method == 'POST':
        # 更新配置
        try:
            data = request.get_json()
            width = int(data.get('width', 512))
            height = int(data.get('height', 512))
            
            # 读取文件
            with open(preprocessor_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换new_size参数
            import re
            new_content = re.sub(
                r'new_size=\(\d+,\s*\d+\)',
                f'new_size=({width}, {height})',
                content
            )
            
            # 写回文件
            with open(preprocessor_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            log_message(f"预处理配置已更新: new_size=({width}, {height})")
            return jsonify({"success": True})
            
        except Exception as e:
            log_message(f"更新预处理配置失败: {str(e)}", "ERROR")
            return jsonify({"error": str(e)}), 500

@app.route('/api/run/preprocessing', methods=['POST'])
def run_preprocessing():
    """运行预处理"""
    try:
        script_path = os.path.join(PROJECT_ROOT, 'src', '1-preprocessing', 'preprocessor.py')
        task_id = f"preprocessing_{int(time.time())}"
        
        # 在后台运行
        thread = threading.Thread(
            target=run_script_with_logging,
            args=(script_path, "图像预处理", task_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"success": True, "task_id": task_id})
        
    except Exception as e:
        log_message(f"启动预处理失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/config/training', methods=['GET', 'POST'])
def training_config():
    """训练配置"""
    train_path = os.path.join(PROJECT_ROOT, 'src', '2-train', 'train.py')
    
    if request.method == 'GET':
        # 读取当前配置
        try:
            with open(train_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 提取配置信息
            import re
            img_dir_match = re.search(r"IMG_DIR = ['\"]([^'\"]+)['\"]", content)
            json_dir_match = re.search(r"JSON_DIR = ['\"]([^'\"]+)['\"]", content)
            model_save_match = re.search(r"MODEL_SAVE_PATH = ['\"]([^'\"]+)['\"]", content)
            
            config = {
                "img_dir": img_dir_match.group(1) if img_dir_match else './data/dataset_0706/processed',
                "json_dir": json_dir_match.group(1) if json_dir_match else './data/dataset_0706/json',
                "model_save_path": model_save_match.group(1) if model_save_match else './models/trained/unet_solder_model.pth'
            }
            
            return jsonify(config)
            
        except Exception as e:
            log_message(f"读取训练配置失败: {str(e)}", "ERROR")
            return jsonify({"error": str(e)}), 500
    
    elif request.method == 'POST':
        # 更新配置
        try:
            data = request.get_json()
            img_dir = data.get('img_dir', './data/dataset_0706/processed')
            json_dir = data.get('json_dir', './data/dataset_0706/json')
            model_save_path = data.get('model_save_path', './models/trained/unet_solder_model.pth')
            
            # 读取文件
            with open(train_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换配置
            import re
            content = re.sub(r"IMG_DIR = ['\"][^'\"]+['\"]", f"IMG_DIR = '{img_dir}'", content)
            content = re.sub(r"JSON_DIR = ['\"][^'\"]+['\"]", f"JSON_DIR = '{json_dir}'", content)
            content = re.sub(r"MODEL_SAVE_PATH = ['\"][^'\"]+['\"]", f"MODEL_SAVE_PATH = '{model_save_path}'", content)
            
            # 写回文件
            with open(train_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            log_message(f"训练配置已更新")
            return jsonify({"success": True})
            
        except Exception as e:
            log_message(f"更新训练配置失败: {str(e)}", "ERROR")
            return jsonify({"error": str(e)}), 500

@app.route('/api/run/training', methods=['POST'])
def run_training():
    """运行训练"""
    try:
        script_path = os.path.join(PROJECT_ROOT, 'src', '2-train', 'train.py')
        task_id = f"training_{int(time.time())}"
        
        # 在后台运行
        thread = threading.Thread(
            target=run_script_with_logging,
            args=(script_path, "模型训练", task_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"success": True, "task_id": task_id})
        
    except Exception as e:
        log_message(f"启动训练失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload/image', methods=['POST'])
def upload_image():
    """上传图片"""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "没有选择文件"}), 400
        
        files = request.files.getlist('files')
        uploaded_files = []
        
        for file in files:
            if file.filename == '':
                continue
                
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # 避免文件名冲突
                if os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
                    name, ext = os.path.splitext(filename)
                    timestamp = int(time.time())
                    filename = f"{name}_{timestamp}{ext}"
                
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                uploaded_files.append(filename)
                log_message(f"上传图片: {filename}")
        
        return jsonify({"success": True, "files": uploaded_files})
        
    except Exception as e:
        log_message(f"上传图片失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload/folder', methods=['POST'])
def upload_folder():
    """上传文件夹"""
    try:
        data = request.get_json()
        folder_type = data.get('folder_type')  # 'yolo' 或 'unet'
        
        # 这里只是示例，实际的文件夹上传需要客户端支持
        log_message(f"准备上传 {folder_type} 数据集文件夹")
        
        # 返回数据集结构指导
        if folder_type == 'yolo':
            structure = {
                "type": "YOLO数据集",
                "structure": {
                    "dataset/": {
                        "images/": {
                            "train/": "训练图片",
                            "val/": "验证图片"
                        },
                        "labels/": {
                            "train/": "训练标签(.txt文件)",
                            "val/": "验证标签(.txt文件)"
                        },
                        "data.yaml": "数据集配置文件"
                    }
                },
                "description": "YOLO格式的目标检测数据集，每个图片对应一个同名的.txt标签文件"
            }
        else:  # unet
            structure = {
                "type": "LabelMe数据集",
                "structure": {
                    "dataset/": {
                        "jpg/": "原始图片文件(.jpg)",
                        "json/": "LabelMe标注文件(.json)",
                        "processed/": "预处理后的图片(可选)"
                    }
                },
                "description": "LabelMe格式的语义分割数据集，每个图片对应一个同名的.json标注文件"
            }
        
        return jsonify({"success": True, "structure": structure})
        
    except Exception as e:
        log_message(f"处理文件夹上传失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/run/detection', methods=['POST'])
def run_detection():
    """运行检测"""
    try:
        data = request.get_json()
        model_type = data.get('model_type')  # 'yolo' 或 'unet'
        
        log_message(f"=== 开始运行检测 ===", "INFO")
        log_message(f"检测类型: {model_type}", "INFO")
        
        if model_type == 'yolo':
            script_path = os.path.join(PROJECT_ROOT, 'src', '3-test', 'predict_yolo.py')
            task_name = "YOLO检测"
        else:  # unet
            script_path = os.path.join(PROJECT_ROOT, 'src', '3-test', 'inference_with_visualization.py')
            task_name = "UNet++检测"
        
        log_message(f"脚本路径: {script_path}", "INFO")
        log_message(f"检查脚本文件是否存在: {os.path.exists(script_path)}", "INFO")
        
        if not os.path.exists(script_path):
            error_msg = f"检测脚本不存在: {script_path}"
            log_message(error_msg, "ERROR")
            return jsonify({"error": error_msg}), 400
        
        task_id = f"detection_{model_type}_{int(time.time())}"
        log_message(f"生成任务ID: {task_id}", "INFO")
        
        # 检查必要目录
        log_message(f"检查结果目录:", "INFO")
        if model_type == 'yolo':
            log_message(f"  YOLO结果目录: {MY_TEST_RESULTS_DIR} (存在: {os.path.exists(MY_TEST_RESULTS_DIR)})", "INFO")
        else:
            log_message(f"  UNet结果目录: {TEMP1_PLUS_DIR} (存在: {os.path.exists(TEMP1_PLUS_DIR)})", "INFO")
        
        # 在后台运行
        thread = threading.Thread(
            target=run_script_with_logging,
            args=(script_path, task_name, task_id)
        )
        thread.daemon = True
        thread.start()
        
        log_message(f"检测任务已启动，任务ID: {task_id}", "INFO")
        return jsonify({"success": True, "task_id": task_id, "model_type": model_type})
        
    except Exception as e:
        log_message(f"启动检测失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/run/visualization', methods=['POST'])
def run_visualization():
    """运行可视化检测"""
    try:
        log_message(f"=== 开始运行可视化检测 ===", "INFO")
        
        script_path = os.path.join(PROJECT_ROOT, 'src', '3-test', 'visualization.py')
        task_id = f"visualization_{int(time.time())}"
        
        log_message(f"可视化脚本路径: {script_path}", "INFO")
        log_message(f"检查脚本文件是否存在: {os.path.exists(script_path)}", "INFO")
        log_message(f"可视化结果目录: {VISUALIZATION_DIR} (存在: {os.path.exists(VISUALIZATION_DIR)})", "INFO")
        
        if not os.path.exists(script_path):
            error_msg = f"可视化脚本不存在: {script_path}"
            log_message(error_msg, "ERROR")
            return jsonify({"error": error_msg}), 400
        
        task_id = f"visualization_{int(time.time())}"
        log_message(f"生成任务ID: {task_id}", "INFO")
        
        # 在后台运行
        thread = threading.Thread(
            target=run_script_with_logging,
            args=(script_path, "可视化检测", task_id)
        )
        thread.daemon = True
        thread.start()
        
        log_message(f"可视化检测任务已启动，任务ID: {task_id}", "INFO")
        return jsonify({"success": True, "task_id": task_id})
        
    except Exception as e:
        log_message(f"启动可视化检测失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/results/<model_type>')
def get_results(model_type):
    """获取检测结果"""
    try:
        log_message(f"获取 {model_type} 检测结果", "INFO")
        
        if model_type == 'yolo':
            results_dir = MY_TEST_RESULTS_DIR
        elif model_type == 'unet':
            results_dir = TEMP1_PLUS_DIR  # 修正UNet++结果目录
        elif model_type == 'visualization':
            results_dir = VISUALIZATION_DIR
        else:
            log_message(f"不支持的模型类型: {model_type}", "ERROR")
            return jsonify({"error": "不支持的模型类型"}), 400
        
        log_message(f"结果目录: {results_dir}", "INFO")
        log_message(f"目录是否存在: {os.path.exists(results_dir)}", "INFO")
        
        if not os.path.exists(results_dir):
            log_message(f"结果目录不存在: {results_dir}", "WARNING")
            return jsonify({"files": []})
        
        # 获取结果文件列表
        files = []
        all_files = os.listdir(results_dir)
        log_message(f"目录中总文件数: {len(all_files)}", "INFO")
        
        for filename in all_files:
            if model_type == 'visualization':
                # 可视化结果只显示overlay图片
                if filename.lower().endswith('_overlay.jpg'):
                    file_path = os.path.join(results_dir, filename)
                    files.append({
                        "name": filename,
                        "size": os.path.getsize(file_path),
                        "modified": os.path.getmtime(file_path)
                    })
            else:
                # 其他模型显示所有图片和文本文件
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.txt')):
                    file_path = os.path.join(results_dir, filename)
                    files.append({
                        "name": filename,
                        "size": os.path.getsize(file_path),
                        "modified": os.path.getmtime(file_path)
                    })
        
        # 按修改时间排序
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        log_message(f"找到 {len(files)} 个结果文件", "INFO")
        if files:
            log_message(f"最新文件: {files[0]['name']}", "INFO")
        
        return jsonify({"files": files})
        
    except Exception as e:
        log_message(f"获取检测结果失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/results/<model_type>/<filename>')
def serve_result_file(model_type, filename):
    """提供结果文件"""
    try:
        if model_type == 'yolo':
            results_dir = MY_TEST_RESULTS_DIR
        elif model_type == 'unet':
            results_dir = TEMP1_PLUS_DIR  # 修正UNet++结果目录
        elif model_type == 'visualization':
            results_dir = VISUALIZATION_DIR
        else:
            return jsonify({"error": "不支持的模型类型"}), 400
        
        return send_from_directory(results_dir, filename)
        
    except Exception as e:
        log_message(f"提供结果文件失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status/<task_id>')
def get_task_status(task_id):
    """获取任务状态"""
    global current_tasks
    try:
        status = current_tasks.get(task_id, {"status": "unknown"})
        
        # 只在状态改变时记录日志，避免过多日志
        if status.get("status") in ["completed", "failed", "error"] and not status.get("logged", False):
            log_message(f"任务 {task_id} 状态: {status.get('status')}", "INFO")
            if "execution_time" in status:
                log_message(f"任务 {task_id} 执行时间: {status['execution_time']:.2f}秒", "INFO")
            current_tasks[task_id]["logged"] = True
        
        return jsonify(status)
    except Exception as e:
        log_message(f"获取任务状态失败: {str(e)}", "ERROR")
        return jsonify({"status": "error", "error": str(e)})

# ==================== 报告相关API ====================

@app.route('/api/analysis/<image_name>')
def get_analysis_data(image_name):
    """获取单个图片的分析数据"""
    try:
        log_message(f"获取图片分析数据: {image_name}", "INFO")
        
        # 移除扩展名，获取基本文件名
        base_name = os.path.splitext(image_name)[0]
        if base_name.endswith('_overlay'):
            base_name = base_name[:-8]  # 移除'_overlay'后缀
        
        log_message(f"解析后的基本文件名: {base_name}", "INFO")
        
        # 解析分析数据（包含连通域统计）
        from analysis_parser import parse_analysis_file_with_mask
        analysis_file = os.path.join(TEMP1_PLUS_DIR, f"{base_name}_analysis.txt")
        
        log_message(f"分析文件路径: {analysis_file}", "INFO")
        log_message(f"分析文件是否存在: {os.path.exists(analysis_file)}", "INFO")
        
        if not os.path.exists(analysis_file):
            error_msg = f"分析文件不存在: {analysis_file}"
            log_message(error_msg, "ERROR")
            return jsonify({"error": error_msg}), 404
        
        log_message(f"开始解析分析文件...", "INFO")
        parsed_image_name, analysis_data, component_counts = parse_analysis_file_with_mask(analysis_file)
        
        if not analysis_data:
            error_msg = "分析数据解析失败"
            log_message(error_msg, "ERROR")
            return jsonify({"error": error_msg}), 500
        
        log_message(f"分析数据解析成功", "INFO")
        log_message(f"连通域统计: {component_counts}", "INFO")
        
        # 计算像素统计
        good_pixels = analysis_data.get('good', {}).get('pixels', 0)
        background_pixels = analysis_data.get('background', {}).get('pixels', 0)
        insufficient_pixels = analysis_data.get('insufficient', {}).get('pixels', 0)
        excess_pixels = analysis_data.get('excess', {}).get('pixels', 0)
        shift_pixels = analysis_data.get('shift', {}).get('pixels', 0)
        miss_pixels = analysis_data.get('miss', {}).get('pixels', 0)
        
        total_pixels = good_pixels + background_pixels + insufficient_pixels + excess_pixels + shift_pixels + miss_pixels
        effective_pixels = total_pixels - background_pixels  # 有效像素（非背景）
        defect_pixels = insufficient_pixels + excess_pixels + shift_pixels + miss_pixels
        
        # 使用连通域统计真实焊点数量
        if component_counts:
            good_solder_points = component_counts.get('good', 0)
            insufficient_solder_points = component_counts.get('insufficient', 0)
            excess_solder_points = component_counts.get('excess', 0)
            shift_solder_points = component_counts.get('shift', 0)
            miss_solder_points = component_counts.get('miss', 0)
            
            total_solder_points = good_solder_points + insufficient_solder_points + excess_solder_points + shift_solder_points + miss_solder_points
            defect_solder_points = insufficient_solder_points + excess_solder_points + shift_solder_points + miss_solder_points
            
            # 基于真实焊点数计算良品率
            quality_rate = (good_solder_points / total_solder_points) if total_solder_points > 0 else 0
        else:
            # 如果无法读取掩码，回退到像素估算
            estimated_solder_points = max(5, min(50, effective_pixels // 10000))  # 更合理的估算
            quality_rate = (good_pixels / effective_pixels) if effective_pixels > 0 else 0
            good_solder_points = int(estimated_solder_points * quality_rate)
            total_solder_points = estimated_solder_points
            defect_solder_points = total_solder_points - good_solder_points
            insufficient_solder_points = defect_solder_points // 4
            excess_solder_points = defect_solder_points // 4
            shift_solder_points = defect_solder_points // 4
            miss_solder_points = defect_solder_points - insufficient_solder_points - excess_solder_points - shift_solder_points
        
        result = {
            'image_name': image_name,
            'base_name': base_name,
            'parsed_image_name': parsed_image_name,
            'total_pixels': total_pixels,
            'effective_pixels': effective_pixels,
            'good_pixels': good_pixels,
            'defect_pixels': defect_pixels,
            'background_pixels': background_pixels,
            'insufficient_pixels': insufficient_pixels,
            'excess_pixels': excess_pixels,
            'shift_pixels': shift_pixels,
            'miss_pixels': miss_pixels,
            'quality_rate': quality_rate,
            'total_solder_points': total_solder_points,
            'good_solder_points': good_solder_points,
            'defect_solder_points': defect_solder_points,
            'insufficient_solder_points': insufficient_solder_points,
            'excess_solder_points': excess_solder_points,
            'shift_solder_points': shift_solder_points,
            'miss_solder_points': miss_solder_points,
            'component_counts': component_counts,
            'analysis_data': analysis_data
        }
        
        return jsonify(result)
        
    except Exception as e:
        log_message(f"获取分析数据失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """获取报告列表"""
    try:
        limit = request.args.get('limit', 50, type=int)
        batch = request.args.get('batch', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        
        if batch:
            reports = report_db.get_reports_by_batch(batch)
        elif start_date and end_date:
            reports = report_db.get_reports_by_date_range(start_date, end_date)
        else:
            reports = report_db.get_all_reports(limit)
        
        # 转换为字典格式
        reports_list = []
        for report in reports:
            # 计算良品率和缺陷数
            total_solder_points = report[5] if report[5] else 0
            good_count = report[6] if report[6] else 0
            pass_rate = (good_count / total_solder_points * 100) if total_solder_points > 0 else 0
            defect_count = total_solder_points - good_count
            
            reports_list.append({
                'id': report[0],
                'image_name': report[1],
                'pcb_model': report[2],
                'batch_number': report[3],
                'detection_time': report[4],
                'total_solder_points': total_solder_points,
                'good_count': good_count,
                'defect_count': defect_count,
                'pass_rate': round(pass_rate, 1),
                'conclusion': report[7]
            })
        
        return jsonify({'reports': reports_list})
        
    except Exception as e:
        log_message(f"获取报告列表失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/<int:report_id>', methods=['GET'])
def get_report_detail(report_id):
    """获取报告详情"""
    try:
        detail = report_db.get_report_detail(report_id)
        
        if not detail:
            return jsonify({"error": "报告未找到"}), 404
        
        report = detail['report']
        defects = detail['defects']
        
        result = {
            'id': report[0],
            'image_name': report[1],
            'pcb_model': report[2],
            'batch_number': report[3],
            'detection_time': report[4],
            'detection_device': report[5],
            'total_solder_points': report[6],
            'good_count': report[7],
            'excess_count': report[8],
            'insufficient_count': report[9],
            'shift_count': report[10],
            'miss_count': report[11],
            'defect_description': report[12],
            'conclusion': report[13],
            'remarks': report[14],
            'defects': []
        }
        
        for defect in defects:
            result['defects'].append({
                'type': defect[0],
                'pixel_count': defect[1],
                'percentage': defect[2]
            })
        
        return jsonify(result)
        
    except Exception as e:
        log_message(f"获取报告详情失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/statistics', methods=['GET'])
def get_statistics():
    """获取统计数据"""
    try:
        stats = report_db.get_statistics_summary()
        return jsonify(stats)
        
    except Exception as e:
        log_message(f"获取统计数据失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/export', methods=['POST'])
def export_reports():
    """导出报告"""
    try:
        data = request.get_json()
        date_filter = data.get('date_filter', '')
        filename = data.get('filename', f'pcb_reports_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        # 确保文件名安全
        filename = secure_filename(filename)
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        export_path = os.path.join(TEMP_DIR, filename)
        report_db.export_to_csv(export_path, date_filter)
        
        log_message(f"报告已导出: {filename}")
        return jsonify({
            "success": True,
            "filename": filename,
            "download_url": f"/api/reports/download/{filename}"
        })
        
    except Exception as e:
        log_message(f"导出报告失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/download/<filename>')
def download_report(filename):
    """下载导出的报告"""
    try:
        return send_from_directory(TEMP_DIR, filename, as_attachment=True)
    except Exception as e:
        log_message(f"下载报告失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/visualizations/<filename>')
def serve_visualization(filename):
    """提供可视化图片"""
    try:
        return send_from_directory(VISUALIZATION_DIR, filename)
    except Exception as e:
        log_message(f"提供可视化图片失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/generate', methods=['POST'])
def generate_report():
    """生成检测报告"""
    try:
        data = request.get_json()
        
        log_message(f"=== 开始生成检测报告 ===", "INFO")
        log_message(f"接收到的数据: {data}", "INFO")
        
        # 构建报告数据
        report_data = {
            'image_name': data.get('image_name', ''),
            'pcb_model': data.get('pcb_model', 'Unknown'),
            'batch_number': data.get('batch_number', 'BATCH001'),
            'detection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'detection_device': 'PCB检测系统v1.0',
            'total_solder_points': data.get('total_solder_points', 0),
            'good_count': data.get('good_count', 0),
            'excess_count': data.get('excess_count', 0),
            'insufficient_count': data.get('insufficient_count', 0),
            'shift_count': data.get('shift_count', 0),
            'miss_count': data.get('miss_count', 0),
            'defect_description': data.get('defect_description', ''),
            'conclusion': data.get('conclusion', 'PASS'),
            'remarks': data.get('remarks', '')
        }
        
        log_message(f"报告数据构建完成:", "INFO")
        log_message(f"  图片名称: {report_data['image_name']}", "INFO")
        log_message(f"  PCB型号: {report_data['pcb_model']}", "INFO")
        log_message(f"  批次号: {report_data['batch_number']}", "INFO")
        log_message(f"  总焊点数: {report_data['total_solder_points']}", "INFO")
        log_message(f"  良品数: {report_data['good_count']}", "INFO")
        log_message(f"  检测结论: {report_data['conclusion']}", "INFO")
        
        analysis_data = data.get('analysis_data', {})
        log_message(f"分析数据: {len(analysis_data)} 个条目", "INFO")
        
        # 插入数据库
        log_message(f"开始插入数据库...", "INFO")
        report_id = report_db.insert_report(report_data, analysis_data)
        log_message(f"数据库插入成功，报告ID: {report_id}", "INFO")
        
        # 计算良品率
        pass_rate = (report_data['good_count'] / report_data['total_solder_points'] * 100) if report_data['total_solder_points'] > 0 else 0
        
        log_message(f"=== 检测报告生成成功 ===", "INFO")
        log_message(f"报告ID: {report_id}", "INFO")
        log_message(f"图片: {report_data['image_name']}", "INFO")
        log_message(f"良品率: {pass_rate:.1f}%", "INFO")
        
        return jsonify({
            "success": True,
            "report_id": report_id,
            "message": "检测报告已生成",
            "pass_rate": f"{pass_rate:.1f}%"
        })
        
    except Exception as e:
        log_message(f"生成报告失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/auto-generate', methods=['POST'])
def auto_generate_reports():
    """自动生成检测报告 - 使用真实分析数据"""
    try:
        log_message(f"=== 开始自动生成检测报告 ===", "INFO")
        log_message(f"分析数据目录: {TEMP1_PLUS_DIR}", "INFO")
        log_message(f"目录是否存在: {os.path.exists(TEMP1_PLUS_DIR)}", "INFO")
        
        # 获取前端传来的已存在报告图片列表
        data = request.get_json()
        existing_report_images = data.get('existingReportImages', []) if data else []
        skip_existing = data.get('skipExisting', True) if data else True
        
        if existing_report_images:
            log_message(f"已存在报告图片数量: {len(existing_report_images)}", "INFO")
        
        if os.path.exists(TEMP1_PLUS_DIR):
            files = os.listdir(TEMP1_PLUS_DIR)
            analysis_files = [f for f in files if f.endswith('_analysis.txt')]
            log_message(f"发现分析文件: {len(analysis_files)} 个", "INFO")
            for f in analysis_files:
                log_message(f"  - {f}", "INFO")
        
        # 获取真实的分析数据
        log_message(f"开始解析分析数据...", "INFO")
        analysis_results = parse_all_analysis_data(TEMP1_PLUS_DIR)
        
        if not analysis_results:
            error_msg = "没有找到分析数据文件"
            log_message(error_msg, "ERROR")
            return jsonify({"error": error_msg}), 400
        
        log_message(f"成功解析 {len(analysis_results)} 个分析结果", "INFO")
        
        generated_reports = []
        skipped_reports = []
        
        for i, (report_data, analysis_result) in enumerate(analysis_results, 1):
            try:
                log_message(f"处理第 {i}/{len(analysis_results)} 个报告...", "INFO")
                log_message(f"  图片: {report_data['image_name']}", "INFO")
                
                # 检查是否已经存在该图片的报告
                if skip_existing and report_data['image_name'] in existing_report_images:
                    log_message(f"  跳过已存在报告的图片: {report_data['image_name']}", "INFO")
                    skipped_reports.append(report_data['image_name'])
                    continue
                
                log_message(f"  总焊点: {report_data['total_solder_points']}", "INFO")
                log_message(f"  良品数: {report_data['good_count']}", "INFO")
                log_message(f"  结论: {report_data['conclusion']}", "INFO")
                
                # 插入数据库
                report_id = report_db.insert_report(report_data, analysis_result)
                
                pass_rate = (report_data['good_count'] / report_data['total_solder_points'] * 100) if report_data['total_solder_points'] > 0 else 0
                
                generated_reports.append({
                    'report_id': report_id,
                    'image_name': report_data['image_name'],
                    'conclusion': report_data['conclusion'],
                    'pass_rate': f"{pass_rate:.1f}%"
                })
                
                log_message(f"报告生成成功: ID={report_id}, 良品率={pass_rate:.1f}%", "INFO")
                
            except Exception as e:
                log_message(f"处理分析数据时出错: {str(e)}", "ERROR")
                continue
        
        log_message(f"=== 自动报告生成完成 ===", "INFO")
        log_message(f"成功生成 {len(generated_reports)} 个报告", "INFO")
        log_message(f"跳过 {len(skipped_reports)} 个已存在报告", "INFO")
        log_message(f"基于真实检测数据", "INFO")
        
        return jsonify({
            "success": True,
            "generated": len(generated_reports),
            "skipped": len(skipped_reports),
            "reports": generated_reports,
            "message": f"成功自动生成 {len(generated_reports)} 个报告，跳过 {len(skipped_reports)} 个已存在报告 (基于真实检测数据)"
        })
        
    except Exception as e:
        log_message(f"自动生成报告失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/clear-all', methods=['POST'])
def clear_all_reports():
    """清空所有报告数据"""
    try:
        count = report_db.clear_all_reports()
        log_message(f"已清空所有报告数据，共删除 {count} 条记录")
        return jsonify({
            "success": True,
            "deleted_count": count,
            "message": f"已清空所有报告数据，共删除 {count} 条记录"
        })
        
    except Exception as e:
        log_message(f"清空报告数据失败: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/test_statistics.html')
def test_statistics():
    """统计功能测试页面"""
    return send_from_directory(os.path.dirname(__file__), 'test_statistics.html')

@app.route('/test_quick_stats.html')
def test_quick_stats():
    """快速统计功能测试页面"""
    return send_from_directory(os.path.dirname(__file__), 'test_quick_stats.html')

if __name__ == '__main__':
    log_message("=== PCB焊点检测系统启动 ===", "INFO")
    log_message(f"项目根目录: {PROJECT_ROOT}", "INFO")
    log_message(f"数据库路径: {db_path}", "INFO")
    log_message(f"检测结果目录:", "INFO")
    log_message(f"  YOLO结果: {MY_TEST_RESULTS_DIR} (存在: {os.path.exists(MY_TEST_RESULTS_DIR)})", "INFO")
    log_message(f"  UNet结果: {TEMP1_PLUS_DIR} (存在: {os.path.exists(TEMP1_PLUS_DIR)})", "INFO")
    log_message(f"  可视化结果: {VISUALIZATION_DIR} (存在: {os.path.exists(VISUALIZATION_DIR)})", "INFO")
    log_message(f"服务器配置: host=0.0.0.0, port=5000, debug=True", "INFO")
    log_message("系统准备就绪，等待用户操作...", "INFO")
    app.run(debug=True, host='0.0.0.0', port=5000)
