# PCB焊点检测系统 GUI 启动脚本

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        'flask',
        'werkzeug'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_project_structure():
    """检查项目结构"""
    print("\n🔍 检查项目结构...")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    required_dirs = [
        'src/1-preprocessing',
        'src/2-train',
        'src/2-train_YOLO',
        'src/3-test',
        'data',
        'models',
        'temp'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (缺失)")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n⚠️ 缺少以下目录: {', '.join(missing_dirs)}")
        print("某些功能可能无法正常工作")
    else:
        print("✅ 项目结构完整")
    
    return len(missing_dirs) == 0

def create_missing_dirs():
    """创建缺失的目录"""
    print("\n🔧 创建必要的目录...")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    dirs_to_create = [
        'data/users',
        'temp',
        'temp/my_test_results',
        'temp1'
    ]
    
    for dir_path in dirs_to_create:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")

def start_gui():
    """启动GUI"""
    print("\n🚀 启动PCB焊点检测系统...")
    
    try:
        # 启动Flask应用
        app_path = Path(__file__).parent / 'app.py'
        
        print("正在启动Web服务器...")
        process = subprocess.Popen([
            sys.executable, str(app_path)
        ], cwd=Path(__file__).parent)
        
        # 等待服务器启动
        print("等待服务器启动...")
        time.sleep(3)
        
        # 打开浏览器
        url = "http://localhost:5000"
        print(f"正在打开浏览器: {url}")
        webbrowser.open(url)
        
        print("\n✅ PCB焊点检测系统已启动!")
        print("📱 GUI界面已在浏览器中打开")
        print("⚠️ 请不要关闭此终端窗口")
        print("🛑 按 Ctrl+C 停止服务器")
        
        # 等待用户中断
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n\n🛑 正在停止服务器...")
            process.terminate()
            process.wait()
            print("✅ 服务器已停止")
            
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("🔬 PCB焊点检测系统 - GUI启动器")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        input("\n按回车键退出...")
        return
    
    # 检查项目结构
    check_project_structure()
    
    # 创建缺失目录
    create_missing_dirs()
    
    # 启动GUI
    print("\n" + "=" * 60)
    if start_gui():
        print("✅ 系统正常退出")
    else:
        print("❌ 系统异常退出")
        input("按回车键退出...")

if __name__ == "__main__":
    main()
