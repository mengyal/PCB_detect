#!/usr/bin/env python3
"""
分析结果解析器 - 解析真实的检测分析数据
"""

import os
import re
import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage

CLASS_MAPPING = {
    'background': 0,
    'good': 1,
    'insufficient': 2,
    'excess': 3,
    'shift': 4,
    'miss': 5,
}

def count_connected_components_from_mask(mask_path, min_area=50):
    """从掩码图像统计连通域数量（焊点数量），过滤掉面积太小的连通域
    
    Args:
        mask_path: 掩码图像路径
        min_area: 最小连通域面积，小于此面积的连通域将被忽略（默认50像素）
    """
    try:
        if not os.path.exists(mask_path):
            print(f"掩码文件不存在: {mask_path}")
            return None
        
        # 读取掩码图像
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取掩码图像: {mask_path}")
            return None
        
        # 统计各类别的连通域
        component_counts = {}
        
        for class_name, class_id in CLASS_MAPPING.items():
            if class_name == 'background':
                continue  # 跳过背景
            
            # 创建当前类别的二值掩码
            class_mask = (mask == class_id).astype(np.uint8)
            
            # 统计连通域数量（带面积过滤）
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            
            # 过滤掉面积太小的连通域
            valid_components = 0
            for i in range(1, num_labels):  # 从1开始，跳过背景标签0
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area:
                    valid_components += 1
                else:
                    print(f"忽略面积过小的连通域: 类别{class_name}, 面积={area}像素")
            
            component_counts[class_name] = valid_components
        
        return component_counts
        
    except Exception as e:
        print(f"统计连通域失败 {mask_path}: {e}")
        return None

def parse_analysis_file(analysis_file_path):
    """解析分析文件，提取真实检测数据"""
    try:
        with open(analysis_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取图片名称
        image_match = re.search(r'图片: (.+)', content)
        image_name = image_match.group(1) if image_match else "unknown"
        
        # 解析各类别的像素数和百分比
        analysis_data = {}
        
        # 匹配格式: "good        :   233377 像素 (  8.44%)"
        pattern = r'(\w+)\s*:\s*(\d+)\s*像素\s*\(\s*([\d.]+)%\)'
        matches = re.findall(pattern, content)
        
        for class_name, pixels, percentage in matches:
            analysis_data[class_name] = {
                'pixels': int(pixels),
                'percentage': float(percentage)
            }
        
        return image_name, analysis_data
        
    except Exception as e:
        print(f"解析分析文件失败 {analysis_file_path}: {e}")
        return None, None

def convert_to_report_data_with_mask(image_name, analysis_data, mask_file_path, pcb_model="PCB-REAL", batch_number="BATCH-REAL"):
    """将分析数据转换为报告格式 - 使用连通域分析"""
    
    # 提取像素数据
    good_pixels = analysis_data.get('good', {}).get('pixels', 0)
    insufficient_pixels = analysis_data.get('insufficient', {}).get('pixels', 0)
    excess_pixels = analysis_data.get('excess', {}).get('pixels', 0)
    shift_pixels = analysis_data.get('shift', {}).get('pixels', 0)
    miss_pixels = analysis_data.get('miss', {}).get('pixels', 0)
    background_pixels = analysis_data.get('background', {}).get('pixels', 0)
    
    # 使用连通域分析统计真实焊点数
    solder_point_data = count_solder_points_from_mask(mask_file_path)
    
    if solder_point_data:
        # 使用真实的连通域统计
        solder_stats = solder_point_data['solder_point_stats']
        good_count = solder_stats.get('good', 0)
        insufficient_count = solder_stats.get('insufficient', 0)
        excess_count = solder_stats.get('excess', 0)
        shift_count = solder_stats.get('shift', 0)
        miss_count = solder_stats.get('miss', 0)
        total_solder_points = solder_point_data['total_solder_points']
    else:
        # 回退到像素估算
        PIXELS_PER_SOLDER_POINT = 10000  # 更保守的估算
        total_solder_pixels = good_pixels + insufficient_pixels + excess_pixels + shift_pixels + miss_pixels
        total_solder_points = max(1, total_solder_pixels // PIXELS_PER_SOLDER_POINT)
        
        good_count = good_pixels // PIXELS_PER_SOLDER_POINT
        insufficient_count = insufficient_pixels // PIXELS_PER_SOLDER_POINT
        excess_count = excess_pixels // PIXELS_PER_SOLDER_POINT
        shift_count = shift_pixels // PIXELS_PER_SOLDER_POINT
        miss_count = miss_pixels // PIXELS_PER_SOLDER_POINT
        
        # 确保总数一致
        defect_count = insufficient_count + excess_count + shift_count + miss_count
        if good_count + defect_count < total_solder_points:
            good_count = total_solder_points - defect_count
    
    # 计算良品率
    pass_rate = (good_count / total_solder_points * 100) if total_solder_points > 0 else 0
    
    # 生成缺陷描述
    defect_descriptions = []
    if excess_count > 0:
        defect_descriptions.append(f"锡料过多:{excess_count}")
    if insufficient_count > 0:
        defect_descriptions.append(f"锡料不足:{insufficient_count}")
    if shift_count > 0:
        defect_descriptions.append(f"元件偏移:{shift_count}")
    if miss_count > 0:
        defect_descriptions.append(f"缺失焊点:{miss_count}")
    
    defect_description = f"良品{good_count}, " + ", ".join(defect_descriptions) if defect_descriptions else f"良品{good_count}, 无缺陷"
    
    # 判断检测结论
    conclusion = "PASS" if pass_rate >= 95.0 else "FAIL"
    
    from datetime import datetime
    
    report_data = {
        'image_name': f"{image_name}.jpg",
        'pcb_model': pcb_model,
        'batch_number': batch_number,
        'detection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detection_device': 'PCB检测系统v1.0',
        'total_solder_points': total_solder_points,
        'good_count': good_count,
        'excess_count': excess_count,
        'insufficient_count': insufficient_count,
        'shift_count': shift_count,
        'miss_count': miss_count,
        'defect_description': defect_description,
        'conclusion': conclusion,
        'remarks': f"连通域分析, 良品率: {pass_rate:.1f}%, 焊点总数: {total_solder_points}"
    }
    
    # 构建分析数据
    analysis_result = {}
    if excess_pixels > 0:
        analysis_result['excess'] = {'pixels': excess_pixels, 'percentage': analysis_data.get('excess', {}).get('percentage', 0)}
    if insufficient_pixels > 0:
        analysis_result['insufficient'] = {'pixels': insufficient_pixels, 'percentage': analysis_data.get('insufficient', {}).get('percentage', 0)}
    if shift_pixels > 0:
        analysis_result['shift'] = {'pixels': shift_pixels, 'percentage': analysis_data.get('shift', {}).get('percentage', 0)}
    if miss_pixels > 0:
        analysis_result['miss'] = {'pixels': miss_pixels, 'percentage': analysis_data.get('miss', {}).get('percentage', 0)}
    
    return report_data, analysis_result

def get_all_analysis_files(analysis_dir):
    """获取所有分析文件"""
    analysis_files = []
    if os.path.exists(analysis_dir):
        for filename in os.listdir(analysis_dir):
            if filename.endswith('_analysis.txt'):
                analysis_files.append(os.path.join(analysis_dir, filename))
    return analysis_files

def parse_all_analysis_data(analysis_dir):
    """解析所有分析数据 - 使用连通域分析"""
    analysis_files = get_all_analysis_files(analysis_dir)
    results = []
    
    for analysis_file in analysis_files:
        try:
            image_name, analysis_data = parse_analysis_file(analysis_file)
            if not (image_name and analysis_data):
                continue
            
            # 查找对应的掩码文件
            base_name = os.path.basename(analysis_file).replace('_analysis.txt', '')
            base_path = os.path.dirname(analysis_file)
            
            # 优先使用raw_mask.png
            possible_mask_files = [
                os.path.join(base_path, f"{base_name}_raw_mask.png"),
                os.path.join(base_path, f"{base_name}_mask.png"),
                os.path.join(base_path, f"{base_name}_mask.jpg"),
            ]
            
            mask_file = None
            for possible_path in possible_mask_files:
                if os.path.exists(possible_path):
                    mask_file = possible_path
                    break
            
            if mask_file:
                report_data, analysis_result = convert_to_report_data_with_mask(image_name, analysis_data, mask_file)
                results.append((report_data, analysis_result))
                print(f"处理完成: {image_name}, 焊点数: {report_data['total_solder_points']}")
            else:
                print(f"警告: 找不到掩码文件 {base_name}")
                
        except Exception as e:
            print(f"处理分析文件失败 {analysis_file}: {e}")
            continue
    
    return results

def parse_analysis_file_with_mask(analysis_file_path):
    """解析分析文件并通过掩码统计真实焊点数量"""
    try:
        # 解析文本文件
        image_name, analysis_data = parse_analysis_file(analysis_file_path)
        if not analysis_data:
            return None, None, None
        
        # 查找对应的掩码文件
        base_path = os.path.dirname(analysis_file_path)
        base_name = os.path.basename(analysis_file_path).replace('_analysis.txt', '')
        
        # 尝试找到掩码文件（优先使用raw_mask.png，因为它包含准确的类别值）
        possible_mask_files = [
            os.path.join(base_path, f"{base_name}_raw_mask.png"),
            os.path.join(base_path, f"{base_name}_mask.png"),
            os.path.join(base_path, f"{base_name}_mask.jpg"),
        ]
        
        mask_path = None
        for possible_path in possible_mask_files:
            if os.path.exists(possible_path):
                mask_path = possible_path
                break
        
        # 统计连通域（焊点数量）
        component_counts = None
        if mask_path:
            component_counts = count_connected_components_from_mask(mask_path)
        
        return image_name, analysis_data, component_counts
        
    except Exception as e:
        print(f"解析分析文件失败 {analysis_file_path}: {e}")
        return None, None, None

def count_solder_points_from_mask(mask_image_path, min_area=50):
    """通过连通域分析统计焊点数量，过滤掉面积太小的连通域
    
    Args:
        mask_image_path: 掩码图像路径
        min_area: 最小连通域面积，小于此面积的连通域将被忽略（默认50像素）
    """
    try:
        # 读取掩码图像
        mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取掩码图像: {mask_image_path}")
            return None
        
        # 统计各类别的连通域数量
        solder_point_stats = {}
        total_solder_points = 0
        
        # 获取所有非背景的像素值
        unique_values = np.unique(mask)
        background_value = 0  # 假设背景是0
        
        for class_value in unique_values:
            if class_value == background_value:
                continue
                
            # 创建当前类别的二值掩码
            binary_mask = (mask == class_value).astype(np.uint8)
            
            # 使用连通域分析统计该类别的焊点数量
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            # 过滤掉面积太小的连通域
            valid_solder_points = 0
            for i in range(1, num_labels):  # 从1开始，跳过背景标签0
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area:
                    valid_solder_points += 1
                else:
                    print(f"忽略面积过小的连通域: 类别{class_value}, 面积={area}像素")
            
            if valid_solder_points > 0:
                # 根据类别值确定类别名称
                class_name = get_class_name_from_value(class_value)
                solder_point_stats[class_name] = valid_solder_points
                total_solder_points += valid_solder_points
                
                print(f"类别 {class_name} (值={class_value}): {valid_solder_points} 个有效焊点 (最小面积>{min_area}像素)")
        
        print(f"总有效焊点数: {total_solder_points} (过滤面积<{min_area}像素的连通域)")
        
        return {
            'total_solder_points': total_solder_points,
            'solder_point_stats': solder_point_stats
        }
        
    except Exception as e:
        print(f"连通域分析失败 {mask_image_path}: {e}")
        return None

def get_class_name_from_value(pixel_value):
    """根据像素值获取类别名称"""
    # 根据实际观察到的raw_mask.png像素值映射
    if pixel_value == 0:
        return 'background'
    elif pixel_value == 1:
        return 'good'
    elif pixel_value == 2:
        return 'insufficient'
    elif pixel_value == 3:
        return 'excess'
    elif pixel_value == 4:
        return 'shift'
    elif pixel_value == 5:
        return 'miss'
    else:
        # 对于压缩图像，使用范围映射
        if 1 <= pixel_value <= 50:
            return 'good'
        elif 51 <= pixel_value <= 100:
            return 'insufficient'
        elif 101 <= pixel_value <= 150:
            return 'excess'
        elif 151 <= pixel_value <= 200:
            return 'shift'
        elif 201 <= pixel_value <= 255:
            return 'miss'
        else:
            return 'unknown'

if __name__ == "__main__":
    # 测试代码
    analysis_dir = "../../temp1/plus"
    results = parse_all_analysis_data(analysis_dir)
    
    print(f"解析了 {len(results)} 个分析文件:")
    for report_data, analysis_result in results:
        print(f"- {report_data['image_name']}: {report_data['conclusion']}, 良品率: {report_data['good_count']}/{report_data['total_solder_points']}")
