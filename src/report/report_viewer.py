import os
import sys
from datetime import datetime, timedelta
from database_manager import PCBReportDatabase
import json

class ReportViewer:
    """报告查看器"""
    
    def __init__(self, db_path="pcb_reports.db"):
        self.db = PCBReportDatabase(db_path)
    
    def show_menu(self):
        """显示主菜单"""
        while True:
            print("\n" + "="*60)
            print("📊 PCB检测报告查看系统")
            print("="*60)
            print("1. 查看所有报告")
            print("2. 按批次查询")
            print("3. 按日期范围查询")
            print("4. 查看统计摘要")
            print("5. 导出报告")
            print("6. 查看缺陷趋势")
            print("0. 退出")
            print("-"*60)
            
            choice = input("请选择操作 (0-6): ").strip()
            
            if choice == '0':
                print("👋 再见!")
                break
            elif choice == '1':
                self.show_all_reports()
            elif choice == '2':
                self.query_by_batch()
            elif choice == '3':
                self.query_by_date_range()
            elif choice == '4':
                self.show_statistics()
            elif choice == '5':
                self.export_reports()
            elif choice == '6':
                self.show_defect_trends()
            else:
                print("❌ 无效选择，请重试")
    
    def show_all_reports(self, limit=20):
        """显示所有报告"""
        import sqlite3
        
        conn = sqlite3.connect(self.db.db_path)
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
        
        if not reports:
            print("📭 暂无检测报告")
            return
        
        print(f"\n📋 最近 {len(reports)} 条检测报告:")
        print("-"*100)
        print(f"{'ID':<4} {'图片名称':<20} {'型号':<15} {'批次':<12} {'检测时间':<20} {'焊点':<6} {'良品':<6} {'结论':<8}")
        print("-"*100)
        
        for report in reports:
            print(f"{report[0]:<4} {report[1]:<20} {report[2]:<15} {report[3]:<12} "
                  f"{report[4]:<20} {report[5]:<6} {report[6]:<6} {report[7]:<8}")
        
        # 显示详细信息选项
        detail_id = input(f"\n输入报告ID查看详情 (1-{reports[-1][0]}，回车跳过): ").strip()
        if detail_id.isdigit():
            self.show_report_detail(int(detail_id))
    
    def show_report_detail(self, report_id):
        """显示报告详情"""
        import sqlite3
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # 获取主报告
        cursor.execute('SELECT * FROM pcb_reports WHERE id = ?', (report_id,))
        report = cursor.fetchone()
        
        if not report:
            print(f"❌ 未找到ID为 {report_id} 的报告")
            return
        
        # 获取缺陷详情
        cursor.execute('''
            SELECT defect_type, pixel_count, percentage 
            FROM defect_details 
            WHERE report_id = ?
        ''', (report_id,))
        
        defects = cursor.fetchall()
        conn.close()
        
        print(f"\n📄 报告详情 (ID: {report_id})")
        print("="*60)
        print(f"图片名称: {report[1]}")
        print(f"PCB型号: {report[2]}")
        print(f"批次号: {report[3]}")
        print(f"检测时间: {report[4]}")
        print(f"检测设备: {report[5]}")
        print(f"焊点总数: {report[6]}")
        print(f"良品数量: {report[7]}")
        print(f"锡料过多: {report[8]}")
        print(f"锡料不足: {report[9]}")
        print(f"元件偏移: {report[10]}")
        print(f"缺失焊点: {report[11]}")
        print(f"缺陷描述: {report[12]}")
        print(f"检测结论: {report[13]}")
        print(f"备注: {report[14]}")
        
        if defects:
            print(f"\n🔍 缺陷详细分布:")
            print("-"*40)
            for defect in defects:
                print(f"{defect[0]:<15}: {defect[1]:>6} 像素 ({defect[2]:>6.2f}%)")
    
    def query_by_batch(self):
        """按批次查询"""
        batch = input("请输入批次号: ").strip()
        if not batch:
            return
        
        reports = self.db.get_reports_by_batch(batch)
        
        if not reports:
            print(f"📭 批次 '{batch}' 没有检测报告")
            return
        
        print(f"\n📋 批次 '{batch}' 的检测报告 (共{len(reports)}条):")
        self._display_reports_table(reports)
    
    def query_by_date_range(self):
        """按日期范围查询"""
        print("请输入日期范围 (格式: YYYY-MM-DD)")
        start_date = input("开始日期: ").strip()
        end_date = input("结束日期 (回车默认今天): ").strip()
        
        if not start_date:
            return
        
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # 验证日期格式
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            print("❌ 日期格式错误，请使用 YYYY-MM-DD 格式")
            return
        
        reports = self.db.get_reports_by_date_range(start_date, end_date)
        
        if not reports:
            print(f"📭 日期范围 {start_date} 到 {end_date} 没有检测报告")
            return
        
        print(f"\n📋 {start_date} 到 {end_date} 的检测报告 (共{len(reports)}条):")
        self._display_reports_table(reports)
    
    def show_statistics(self):
        """显示统计摘要"""
        stats = self.db.get_statistics_summary()
        
        print("\n📊 检测统计摘要")
        print("="*50)
        print(f"总报告数: {stats['total_reports']}")
        print(f"总焊点数: {stats['total_solder_points']}")
        print(f"良品总数: {stats['total_good']}")
        print(f"缺陷总数: {stats['total_defects']}")
        print(f"平均良品率: {stats['avg_pass_rate']:.2f}%")
        
        print(f"\n📈 结论分布:")
        print("-"*30)
        for conclusion, count in stats['conclusion_distribution'].items():
            percentage = count / stats['total_reports'] * 100 if stats['total_reports'] > 0 else 0
            print(f"{conclusion:<10}: {count:>3} ({percentage:>5.1f}%)")
    
    def export_reports(self):
        """导出报告"""
        export_path = input("请输入导出文件路径 (默认: reports_export.csv): ").strip()
        if not export_path:
            export_path = "reports_export.csv"
        
        date_filter = input("请输入起始日期过滤 (YYYY-MM-DD，回车跳过): ").strip()
        
        try:
            self.db.export_to_csv(export_path, date_filter)
            print(f"✅ 报告已导出到: {export_path}")
        except Exception as e:
            print(f"❌ 导出失败: {e}")
    
    def show_defect_trends(self):
        """显示缺陷趋势"""
        import sqlite3
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # 按日期统计缺陷
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
        
        if not trends:
            print("📭 最近7天没有检测数据")
            return
        
        print(f"\n📈 最近7天缺陷趋势:")
        print("-"*60)
        print(f"{'日期':<12} {'报告数':<8} {'平均良品率':<12} {'缺陷总数':<8}")
        print("-"*60)
        
        for trend in trends:
            print(f"{trend[0]:<12} {trend[1]:<8} {trend[2]:<12.2f}% {trend[3]:<8}")
    
    def _display_reports_table(self, reports):
        """显示报告表格"""
        print("-"*120)
        print(f"{'ID':<4} {'图片名称':<25} {'型号':<15} {'批次':<12} {'检测时间':<20} {'焊点':<6} {'良品':<6} {'结论':<8}")
        print("-"*120)
        
        for report in reports:
            print(f"{report[0]:<4} {report[1]:<25} {report[2]:<15} {report[3]:<12} "
                  f"{report[4]:<20} {report[6]:<6} {report[7]:<6} {report[13]:<8}")

def main():
    """主程序"""
    viewer = ReportViewer()
    viewer.show_menu()

if __name__ == "__main__":
    main()
