import sqlite3
import os
from datetime import datetime
import json

class PCBReportDatabase:
    """PCB检测报告数据库管理类"""
    
    def __init__(self, db_path="pcb_reports.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建报告表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pcb_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                pcb_model TEXT,
                batch_number TEXT,
                detection_time TEXT,
                detection_device TEXT,
                total_solder_points INTEGER,
                good_count INTEGER,
                excess_count INTEGER,
                insufficient_count INTEGER,
                shift_count INTEGER,
                miss_count INTEGER,
                defect_description TEXT,
                conclusion TEXT,
                remarks TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建缺陷详情表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS defect_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER,
                defect_type TEXT,
                pixel_count INTEGER,
                percentage REAL,
                FOREIGN KEY (report_id) REFERENCES pcb_reports (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_report(self, report_data, analysis_data=None):
        """插入检测报告"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 插入主报告
        cursor.execute('''
            INSERT INTO pcb_reports (
                image_name, pcb_model, batch_number, detection_time,
                detection_device, total_solder_points, good_count,
                excess_count, insufficient_count, shift_count,
                miss_count, defect_description, conclusion, remarks
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report_data['image_name'],
            report_data['pcb_model'],
            report_data['batch_number'],
            report_data['detection_time'],
            report_data['detection_device'],
            report_data['total_solder_points'],
            report_data['good_count'],
            report_data['excess_count'],
            report_data['insufficient_count'],
            report_data['shift_count'],
            report_data['miss_count'],
            report_data['defect_description'],
            report_data['conclusion'],
            report_data['remarks']
        ))
        
        report_id = cursor.lastrowid
        
        # 插入缺陷详情
        if analysis_data:
            for class_name, stats in analysis_data.items():
                if class_name != 'background':
                    cursor.execute('''
                        INSERT INTO defect_details (report_id, defect_type, pixel_count, percentage)
                        VALUES (?, ?, ?, ?)
                    ''', (report_id, class_name, stats['pixels'], stats['percentage']))
        
        conn.commit()
        conn.close()
        return report_id
    
    def get_all_reports(self, limit=50):
        """获取所有报告"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, image_name, pcb_model, batch_number, detection_time, 
                   total_solder_points, good_count, conclusion
            FROM pcb_reports 
            ORDER BY detection_time DESC 
            LIMIT ?
        ''', (limit,))
        
        reports = cursor.fetchall()
        conn.close()
        return reports
    
    def get_report_detail(self, report_id):
        """获取报告详情"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取主报告
        cursor.execute('SELECT * FROM pcb_reports WHERE id = ?', (report_id,))
        report = cursor.fetchone()
        
        if not report:
            conn.close()
            return None
        
        # 获取缺陷详情
        cursor.execute('''
            SELECT defect_type, pixel_count, percentage 
            FROM defect_details 
            WHERE report_id = ?
        ''', (report_id,))
        
        defects = cursor.fetchall()
        conn.close()
        
        return {
            'report': report,
            'defects': defects
        }
    
    def get_reports_by_batch(self, batch_number):
        """根据批次查询报告"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, image_name, pcb_model, batch_number, detection_time, 
                   total_solder_points, good_count, conclusion
            FROM pcb_reports WHERE batch_number = ?
            ORDER BY detection_time DESC
        ''', (batch_number,))
        
        reports = cursor.fetchall()
        conn.close()
        return reports
    
    def get_reports_by_date_range(self, start_date, end_date):
        """根据日期范围查询报告"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, image_name, pcb_model, batch_number, detection_time, 
                   total_solder_points, good_count, conclusion
            FROM pcb_reports 
            WHERE DATE(detection_time) BETWEEN ? AND ?
            ORDER BY detection_time DESC
        ''', (start_date, end_date))
        
        reports = cursor.fetchall()
        conn.close()
        return reports
    
    def get_statistics_summary(self):
        """获取统计摘要"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总体统计
        cursor.execute('''
            SELECT 
                COUNT(*) as total_reports,
                SUM(total_solder_points) as total_solder_points,
                SUM(good_count) as total_good,
                SUM(excess_count + insufficient_count + shift_count + miss_count) as total_defects,
                AVG(CASE WHEN total_solder_points > 0 THEN 
                    CAST(good_count AS FLOAT) / total_solder_points * 100 
                    ELSE 0 END) as avg_pass_rate
            FROM pcb_reports
        ''')
        
        summary = cursor.fetchone()
        
        # 按结论分类统计
        cursor.execute('''
            SELECT conclusion, COUNT(*) as count
            FROM pcb_reports
            GROUP BY conclusion
        ''')
        
        conclusion_stats = cursor.fetchall()
        
        # 缺陷趋势（最近7天）
        cursor.execute('''
            SELECT 
                DATE(detection_time) as date,
                COUNT(*) as total_reports,
                AVG(CASE WHEN total_solder_points > 0 THEN 
                    CAST(good_count AS FLOAT) / total_solder_points * 100 
                    ELSE 0 END) as avg_pass_rate,
                SUM(excess_count + insufficient_count + shift_count + miss_count) as total_defects
            FROM pcb_reports 
            WHERE detection_time >= date('now', '-7 days')
            GROUP BY DATE(detection_time)
            ORDER BY date DESC
        ''')
        
        trends = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_reports': summary[0],
            'total_solder_points': summary[1] or 0,
            'total_good': summary[2] or 0,
            'total_defects': summary[3] or 0,
            'avg_pass_rate': summary[4] or 0,
            'conclusion_distribution': dict(conclusion_stats),
            'trends': trends
        }
    
    def export_to_csv(self, output_path, date_filter=None):
        """导出数据到CSV"""
        import csv
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT 
                id, image_name, pcb_model, batch_number, detection_time,
                detection_device, total_solder_points, good_count,
                excess_count, insufficient_count, shift_count, miss_count,
                defect_description, conclusion, remarks
            FROM pcb_reports
        '''
        
        if date_filter:
            query += f" WHERE DATE(detection_time) >= '{date_filter}'"
        
        query += " ORDER BY detection_time DESC"
        
        cursor.execute(query)
        reports = cursor.fetchall()
        
        # 获取列名
        columns = ['ID', '图片名称', 'PCB型号', '批次号', '检测时间', '检测设备', 
                  '焊点总数', '良品数', '锡料过多', '锡料不足', '元件偏移', '缺失焊点',
                  '缺陷描述', '检测结论', '备注']
        
        # 写入CSV
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            writer.writerows(reports)
        
        conn.close()
        return output_path
    
    def clear_all_reports(self):
        """清空所有报告数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 获取删除前的记录数
            cursor.execute("SELECT COUNT(*) FROM pcb_reports")
            count = cursor.fetchone()[0]
            
            # 删除所有缺陷详情
            cursor.execute("DELETE FROM defect_details")
            
            # 删除所有报告
            cursor.execute("DELETE FROM pcb_reports")
            
            # 重置自增ID
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='pcb_reports'")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='defect_details'")
            
            conn.commit()
            return count
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
