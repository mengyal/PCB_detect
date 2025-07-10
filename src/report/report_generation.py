import os
import numpy as np
from datetime import datetime
import csv
import sqlite3
import json
from pathlib import Path

CLASS_MAPPING = {
    'background': 0,
    'good': 1,
    'insufficient': 2,
    'excess': 3,
    'shift': 4,
    'miss': 5,
}

ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

def analyze_segmentation(prediction):
    """分析分割结果"""
    unique, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size
    analysis = {}
    for class_id, count in zip(unique, counts):
        class_name = ID_TO_CLASS.get(class_id, f"unknown_{class_id}")
        percentage = (count / total_pixels) * 100
        analysis[class_name] = {
            'pixels': int(count),
            'percentage': percentage
        }
    return analysis

def save_analysis_report(image_name, analysis, output_dir):
    """保存单张图片的分析报告"""
    report_path = os.path.join(output_dir, f"{image_name}_analysis.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"PCB焊点检测分析报告\n")
        f.write(f"图片: {image_name}\n")
        f.write("=" * 50 + "\n\n")
        important_classes = ['good', 'insufficient', 'excess', 'shift', 'background']
        for class_name in important_classes:
            if class_name in analysis:
                stats = analysis[class_name]
                f.write(f"{class_name:12s}: {stats['pixels']:8d} 像素 ({stats['percentage']:6.2f}%)\n")
        for class_name, stats in analysis.items():
            if class_name not in important_classes:
                f.write(f"{class_name:12s}: {stats['pixels']:8d} 像素 ({stats['percentage']:6.2f}%)\n")
        f.write(f"\n总像素数: {sum(stats['pixels'] for stats in analysis.values())}\n")

def generate_summary_report(output_dir, successful, failed, total, config):
    """生成总结报告"""
    report_path = os.path.join(output_dir, "inference_summary.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PCB焊点检测推理总结报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"推理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型配置: {config['model_name']}\n")
        f.write(f"模型架构: {config['model_name']}\n")
        f.write(f"编码器: {config['encoder_name']}\n")
        f.write(f"描述: {config['description']}\n\n")
        f.write(f"处理统计:\n")
        f.write(f"  总图片数: {total}\n")
        f.write(f"  成功处理: {successful}\n")
        f.write(f"  处理失败: {failed}\n")
        f.write(f"  成功率: {successful/total*100:.1f}%\n\n")
        f.write(f"输入目录: {output_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write("类别说明:\n")
        for class_name, class_id in CLASS_MAPPING.items():
            f.write(f"  {class_name:12s} (ID: {class_id})\n")
        f.write("\n生成的文件类型:\n")
        f.write("  *_original.jpg     - 原始图片\n")
        f.write("  *_mask.jpg         - 彩色分割掩码\n")
        f.write("  *_overlay.jpg      - 叠加可视化\n")
        f.write("  *_comparison.jpg   - 三联对比图\n")
        f.write("  *_raw_mask.png     - 原始预测掩码\n")
        f.write("  *_analysis.txt     - 像素统计分析\n")
        f.write("  inference_summary.txt - 本总结报告\n")

def generate_structured_report(image_name, analysis, output_dir, pcb_info=None):
    """生成结构化报告（CSV格式）"""
    if pcb_info is None:
        pcb_info = {
            'model': 'Unknown',
            'batch': 'Unknown',
            'device': 'AI Detection System'
        }
    
    # 计算焊点统计
    total_solder_points = sum(stats['pixels'] for name, stats in analysis.items() if name != 'background')
    good_count = analysis.get('good', {}).get('pixels', 0)
    excess_count = analysis.get('excess', {}).get('pixels', 0)
    insufficient_count = analysis.get('insufficient', {}).get('pixels', 0)
    shift_count = analysis.get('shift', {}).get('pixels', 0)
    miss_count = analysis.get('miss', {}).get('pixels', 0)
    
    # 生成缺陷位置描述
    defect_description = []
    if excess_count > 0:
        defect_description.append(f"锡料过多区域: {excess_count}个像素点")
    if insufficient_count > 0:
        defect_description.append(f"锡料不足区域: {insufficient_count}个像素点")
    if shift_count > 0:
        defect_description.append(f"元件偏移区域: {shift_count}个像素点")
    if miss_count > 0:
        defect_description.append(f"缺失焊点区域: {miss_count}个像素点")
    
    defect_desc = "; ".join(defect_description) if defect_description else "无缺陷"
    
    # 检测结论
    defect_total = excess_count + insufficient_count + shift_count + miss_count
    if defect_total == 0:
        conclusion = "合格"
    elif defect_total / total_solder_points < 0.1:
        conclusion = "轻微缺陷"
    elif defect_total / total_solder_points < 0.3:
        conclusion = "中等缺陷"
    else:
        conclusion = "严重缺陷"
    
    # 生成报告数据
    report_data = {
        'image_name': image_name,
        'pcb_model': pcb_info['model'],
        'batch_number': pcb_info['batch'],
        'detection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detection_device': pcb_info['device'],
        'total_solder_points': total_solder_points,
        'good_count': good_count,
        'excess_count': excess_count,
        'insufficient_count': insufficient_count,
        'shift_count': shift_count,
        'miss_count': miss_count,
        'defect_description': defect_desc,
        'conclusion': conclusion,
        'remarks': f"良品率: {(good_count/total_solder_points*100):.2f}%" if total_solder_points > 0 else "无焊点检测到"
    }
    
    return report_data

def save_csv_report(report_data_list, output_dir):
    """保存CSV格式报告"""
    csv_path = os.path.join(output_dir, "pcb_detection_report.csv")
    os.makedirs(output_dir, exist_ok=True)
    
    fieldnames = [
        '序号', 'PCB板信息（型号/批次）', '检测时间', '检测设备', '焊点总数',
        '良品（good）', '锡料过多（excess）', '锡料不足（insufficient）',
        '元件偏移（shift）', '缺失焊点（miss）', '缺陷位置描述',
        '检测结论', '备注'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, data in enumerate(report_data_list, 1):
            writer.writerow({
                '序号': idx,
                'PCB板信息（型号/批次）': f"{data['pcb_model']}/{data['batch_number']}",
                '检测时间': data['detection_time'],
                '检测设备': data['detection_device'],
                '焊点总数': data['total_solder_points'],
                '良品（good）': data['good_count'],
                '锡料过多（excess）': data['excess_count'],
                '锡料不足（insufficient）': data['insufficient_count'],
                '元件偏移（shift）': data['shift_count'],
                '缺失焊点（miss）': data['miss_count'],
                '缺陷位置描述': data['defect_description'],
                '检测结论': data['conclusion'],
                '备注': data['remarks']
            })
    
    print(f"CSV报告已保存到: {csv_path}")
    return csv_path
