#!/usr/bin/env python3
"""
生成示例报告数据
"""

import os
import sys
from datetime import datetime, timedelta
import random

# 添加GUI目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
gui_dir = os.path.join(current_dir, '..', 'GUI')
sys.path.append(gui_dir)

from database_manager import PCBReportDatabase

def generate_sample_reports():
    """生成示例报告数据"""
    
    # 初始化数据库
    db_path = os.path.join(gui_dir, 'reports.db')
    db = PCBReportDatabase(db_path)
    
    # 示例数据
    pcb_models = ['PCB-2024-A01', 'PCB-2024-B02', 'PCB-2024-C03']
    batches = ['BATCH001', 'BATCH002', 'BATCH003']
    conclusions = ['PASS', 'FAIL']
    
    print("正在生成示例报告数据...")
    
    for i in range(50):  # 生成50个示例报告
        # 随机生成时间（最近30天内）
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
        detection_time = base_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 随机生成检测数据
        total_points = random.randint(100, 500)
        good_count = random.randint(int(total_points * 0.7), total_points)
        
        # 剩余的分配给各种缺陷
        remaining = total_points - good_count
        excess = random.randint(0, remaining // 3) if remaining > 0 else 0
        insufficient = random.randint(0, (remaining - excess) // 2) if remaining > excess else 0
        shift = random.randint(0, remaining - excess - insufficient) if remaining > excess + insufficient else 0
        miss = remaining - excess - insufficient - shift
        
        # 构建报告数据
        report_data = {
            'image_name': f'test_pcb_{i+1:03d}.jpg',
            'pcb_model': random.choice(pcb_models),
            'batch_number': random.choice(batches),
            'detection_time': detection_time,
            'detection_device': 'PCB检测系统v1.0',
            'total_solder_points': total_points,
            'good_count': good_count,
            'excess_count': excess,
            'insufficient_count': insufficient,
            'shift_count': shift,
            'miss_count': miss,
            'defect_description': f'检测到 {remaining} 个缺陷焊点' if remaining > 0 else '未发现缺陷',
            'conclusion': 'PASS' if good_count / total_points >= 0.95 else 'FAIL',
            'remarks': f'批量检测 - 第{i+1}个样本'
        }
        
        # 生成分析数据
        analysis_data = {}
        if excess > 0:
            analysis_data['excess'] = {
                'pixels': excess * random.randint(50, 200),
                'percentage': (excess / total_points) * 100
            }
        if insufficient > 0:
            analysis_data['insufficient'] = {
                'pixels': insufficient * random.randint(30, 150),
                'percentage': (insufficient / total_points) * 100
            }
        if shift > 0:
            analysis_data['shift'] = {
                'pixels': shift * random.randint(40, 180),
                'percentage': (shift / total_points) * 100
            }
        if miss > 0:
            analysis_data['miss'] = {
                'pixels': miss * random.randint(20, 100),
                'percentage': (miss / total_points) * 100
            }
        
        # 插入数据库
        report_id = db.insert_report(report_data, analysis_data)
        print(f"已生成报告 {report_id}: {report_data['image_name']} - {report_data['conclusion']}")
    
    print(f"\n✅ 成功生成 50 个示例报告!")
    print(f"数据库路径: {db_path}")

if __name__ == "__main__":
    generate_sample_reports()
