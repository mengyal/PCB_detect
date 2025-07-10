# PCB检测报告系统

这个系统包含了完整的PCB检测报告生成、数据库管理和查看功能。

## 文件分工

### 1. `report_generation.py` - 报告生成模块
- **功能**: 生成结构化检测报告
- **主要函数**:
  - `analyze_segmentation()`: 分析分割结果
  - `generate_structured_report()`: 生成结构化报告数据
  - `save_csv_report()`: 保存CSV格式报告
  - `save_analysis_report()`: 保存单张图片分析报告

### 2. `database_manager.py` - 数据库管理模块
- **功能**: 管理SQLite数据库，存储和查询检测报告
- **主要功能**:
  - 创建和初始化数据库表
  - 插入检测报告数据
  - 按批次、日期查询报告
  - 生成统计摘要
  - 导出数据到CSV

### 3. `upload_manager.py` - 上传管理模块
- **功能**: 负责将检测结果上传到数据库和云端
- **主要功能**:
  - 解析检测结果文件
  - 批量上传检测报告
  - 复制相关文件到指定目录
  - 模拟云端上传功能

### 4. `report_viewer.py` - 报告查看模块
- **功能**: 提供交互式界面查看和管理检测报告
- **主要功能**:
  - 查看所有报告列表
  - 按批次和日期范围查询
  - 查看详细报告信息
  - 显示统计摘要和趋势分析
  - 导出报告数据

## 数据库结构

### pcb_reports 表
```sql
- id: 主键
- image_name: 图片名称
- pcb_model: PCB型号
- batch_number: 批次号
- detection_time: 检测时间
- detection_device: 检测设备
- total_solder_points: 焊点总数
- good_count: 良品数量
- excess_count: 锡料过多数量
- insufficient_count: 锡料不足数量
- shift_count: 元件偏移数量
- miss_count: 缺失焊点数量
- defect_description: 缺陷位置描述
- conclusion: 检测结论
- remarks: 备注
```

### defect_details 表
```sql
- id: 主键
- report_id: 关联报告ID
- defect_type: 缺陷类型
- pixel_count: 像素数量
- percentage: 百分比
```

## 使用方法

### 1. 上传检测结果
```bash
python upload_manager.py <检测结果目录>
```

### 2. 查看报告
```bash
python report_viewer.py
```

### 3. 在代码中使用
```python
from database_manager import PCBReportDatabase
from report_generation import generate_structured_report

# 初始化数据库
db = PCBReportDatabase()

# 生成报告
report_data = generate_structured_report(image_name, analysis, output_dir, pcb_info)

# 插入数据库
report_id = db.insert_report(report_data, analysis)
```

## 报告格式

生成的CSV报告包含以下列：
- 序号
- PCB板信息（型号/批次）
- 检测时间
- 检测设备
- 焊点总数
- 良品（good）
- 锡料过多（excess）
- 锡料不足（insufficient）
- 元件偏移（shift）
- 缺失焊点（miss）
- 缺陷位置描述
- 检测结论
- 备注

## 特性

1. **自动化报告生成**: 根据分割结果自动生成标准化报告
2. **数据库管理**: 使用SQLite存储所有检测数据，支持复杂查询
3. **批次管理**: 支持按批次组织和查询检测结果
4. **统计分析**: 提供详细的统计摘要和趋势分析
5. **数据导出**: 支持导出为CSV格式，便于进一步分析
6. **云端集成**: 预留云端上传接口，可集成各种云存储服务
7. **交互式查看**: 提供命令行界面，方便查看和管理报告

## 扩展功能

- 可以集成实际的云存储API（AWS S3、阿里云OSS等）
- 可以添加Web界面进行报告查看和管理
- 可以集成邮件通知功能
- 可以添加报告模板自定义功能
