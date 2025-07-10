import os
import sys
import shutil
from datetime import datetime
from database_manager import PCBReportDatabase
from report_generation import generate_structured_report, save_csv_report

class ReportUploader:
    """报告上传管理器"""
    
    def __init__(self, upload_dir="uploads", db_path="pcb_reports.db"):
        self.upload_dir = upload_dir
        self.db = PCBReportDatabase(db_path)
        os.makedirs(upload_dir, exist_ok=True)
    
    def upload_detection_results(self, results_dir, pcb_info=None):
        """上传检测结果到数据库和云端"""
        print("🚀 开始上传检测结果...")
        
        if pcb_info is None:
            pcb_info = {
                'model': input("请输入PCB型号: ") or "Unknown",
                'batch': input("请输入批次号: ") or datetime.now().strftime("%Y%m%d"),
                'device': "AI Detection System"
            }
        
        # 查找分析文件
        analysis_files = []
        for file in os.listdir(results_dir):
            if file.endswith('_analysis.txt'):
                analysis_files.append(file)
        
        if not analysis_files:
            print("❌ 未找到分析文件")
            return
        
        report_data_list = []
        uploaded_count = 0
        
        for analysis_file in analysis_files:
            try:
                # 解析分析文件
                analysis_data = self._parse_analysis_file(
                    os.path.join(results_dir, analysis_file)
                )
                
                image_name = analysis_file.replace('_analysis.txt', '')
                
                # 生成结构化报告
                report_data = generate_structured_report(
                    image_name, analysis_data, self.upload_dir, pcb_info
                )
                
                # 插入数据库
                report_id = self.db.insert_report(report_data, analysis_data)
                
                # 复制相关文件到上传目录
                self._copy_result_files(results_dir, image_name, report_id)
                
                report_data_list.append(report_data)
                uploaded_count += 1
                
            except Exception as e:
                print(f"⚠️ 处理文件 {analysis_file} 时出错: {e}")
                continue
        
        # 生成批次报告
        if report_data_list:
            batch_report_path = save_csv_report(report_data_list, self.upload_dir)
            print(f"✅ 成功上传 {uploaded_count} 个检测结果")
            print(f"📊 批次报告: {batch_report_path}")
            
            # 上传到云端（模拟）
            self._upload_to_cloud(batch_report_path, pcb_info)
        
        return uploaded_count
    
    def _parse_analysis_file(self, file_path):
        """解析分析文件"""
        analysis_data = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            if ':' in line and '像素' in line:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    class_name = parts[0].strip()
                    stats_part = parts[1].strip()
                    
                    # 提取像素数和百分比
                    if '像素' in stats_part and '(' in stats_part:
                        pixel_part = stats_part.split('像素')[0].strip()
                        percent_part = stats_part.split('(')[1].split('%')[0]
                        
                        analysis_data[class_name] = {
                            'pixels': int(pixel_part),
                            'percentage': float(percent_part)
                        }
        
        return analysis_data
    
    def _copy_result_files(self, source_dir, image_name, report_id):
        """复制检测结果文件"""
        target_dir = os.path.join(self.upload_dir, f"report_{report_id}")
        os.makedirs(target_dir, exist_ok=True)
        
        # 复制相关文件
        file_patterns = [
            f"{image_name}_original.jpg",
            f"{image_name}_mask.jpg",
            f"{image_name}_overlay.jpg",
            f"{image_name}_comparison.jpg",
            f"{image_name}_analysis.txt"
        ]
        
        for pattern in file_patterns:
            source_path = os.path.join(source_dir, pattern)
            if os.path.exists(source_path):
                target_path = os.path.join(target_dir, pattern)
                shutil.copy2(source_path, target_path)
    
    def _upload_to_cloud(self, report_path, pcb_info):
        """上传到云端（模拟）"""
        print(f"☁️ 正在上传到云端...")
        print(f"   批次: {pcb_info['batch']}")
        print(f"   型号: {pcb_info['model']}")
        print(f"   文件: {report_path}")
        
        # 这里可以集成实际的云存储API
        # 比如AWS S3, 阿里云OSS, 腾讯云COS等
        
        # 模拟上传延时
        import time
        time.sleep(2)
        
        print("✅ 云端上传完成")
        
        # 生成分享链接（模拟）
        share_link = f"https://cloud.example.com/reports/{pcb_info['batch']}/{datetime.now().strftime('%Y%m%d%H%M%S')}"
        print(f"🔗 分享链接: {share_link}")
        
        return share_link

def main():
    """主程序"""
    if len(sys.argv) < 2:
        print("使用方法: python upload_manager.py <检测结果目录>")
        return
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"❌ 目录不存在: {results_dir}")
        return
    
    uploader = ReportUploader()
    uploader.upload_detection_results(results_dir)

if __name__ == "__main__":
    main()
