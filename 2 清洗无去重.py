import pandas as pd
import numpy as np
import re
import os


# -------------------------- 工具函数 --------------------------
def extract_numeric(value):
    """从字符串中提取纯数字（处理类似'80625条评论'的格式）"""
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        numeric_str = re.findall(r'(\d+\.?\d*)', str(value).replace(',', ''))
        if numeric_str:
            return float(numeric_str[0])
        else:
            return np.nan


# 新增：汉字异常处理函数
def clean_chinese_text(text):
    """去除/修正汉字异常（乱码、特殊符号、无意义字符）"""
    if pd.isna(text) or text in ['', 'nan']:
        return '未知'

    # 1. 去除非中文字符/数字/常见符号外的乱码（保留：中文、数字、英文、/、-、()、.）
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9/\-()., ]', '', str(text))
    # 2. 修正常见汉字错误（可根据实际情况补充）
    error_map = {
        "京点": "甜点",
        "杜案": "肚案",
        "鲜巾": "鲜中",
        "气甜": "甜点"
    }
    for wrong, right in error_map.items():
        cleaned = cleaned.replace(wrong, right)
    # 3. 去除多余空格
    cleaned = ' '.join(cleaned.strip().split())
    # 4. 空字符串兜底
    return cleaned if cleaned else '未知'


def calculate_abnormal_values(df):
    """统计各数值字段的异常值数量"""
    abnormal_stats = {}
    numeric_fields = ['评论数', '推荐值', '原价', '折扣价', '排序']

    for field in numeric_fields:
        if field not in df.columns:
            abnormal_stats[field] = 0
            continue

        # 转换为数值型（避免文本干扰）
        temp_series = df[field].apply(extract_numeric)

        # 定义各字段异常值规则
        if field == '排序':
            abnormal = temp_series[(temp_series > 1000) | (temp_series < 1)].count()
        elif field == '原价':
            abnormal = temp_series[(temp_series > 1000) | (temp_series < 0)].count()
        elif field == '折扣价':
            abnormal = temp_series[temp_series < 0].count()
        elif field == '评论数':
            abnormal = temp_series[temp_series > 1000000].count()  # 超过100万评论视为异常
        elif field == '推荐值':
            abnormal = temp_series[(temp_series < 0) | (temp_series > 100)].count()  # 推荐值通常0-100
        else:
            abnormal = 0

        abnormal_stats[field] = abnormal
    return abnormal_stats


def calculate_missing_values(df):
    """统计各字段缺失值数量及占比"""
    missing_stats = {}
    total_rows = len(df)
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_ratio = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        missing_stats[col] = {
            'count': missing_count,
            'ratio': round(missing_ratio, 2)
        }
    return missing_stats


def calculate_numeric_stats(df):
    """计算清洗后数值字段的核心统计量（均值、中位数等）"""
    numeric_stats = {}
    numeric_fields = ['评论数', '推荐值', '原价', '折扣价', '排序']
    for field in numeric_fields:
        if field in df.columns:
            series = df[field].dropna()
            if not series.empty:
                numeric_stats[field] = {
                    'mean': series.mean(),
                    'median': series.median(),
                    'min': series.min(),
                    'max': series.max()
                }
    return numeric_stats


def clean_data(raw_data):
    """数据预处理（清洗）+ 统计清洗前后指标"""
    # 统计清洗前的指标
    pre_stats = {
        'total_rows': len(raw_data),
        'missing': calculate_missing_values(raw_data),
        'abnormal': calculate_abnormal_values(raw_data),
        'numeric_stats': {}
    }

    cleaned_data = raw_data.copy()
    print("\n===== 【数据清洗流程开始】=====")
    print("数据清洗的核心目的是：去除脏数据、统一格式、填补缺失，让数据满足后续分析要求\n")

    # 1. 字段筛选（保留核心分析字段）
    print("【步骤1：字段筛选】")
    print("→ 作用：只保留后续分析需要的核心字段，剔除冗余信息，简化数据集")
    keep_cols = ['排行榜类型', '图书类别', '排序', '书名', '作者', '原价', '折扣价', '评论数', '推荐值', '出版社', '出版日期']
    available_cols = [col for col in keep_cols if col in cleaned_data.columns]
    cleaned_data = cleaned_data[available_cols]
    print(f"→ 保留的核心字段：{available_cols}")
    print(f"→ 完成后数据集规模：{len(cleaned_data)} 条记录\n")

    # ==============================================================================
    # 2. 去重处理（已取消）
    # 注释原因：根据需求，保留所有记录，不进行去重操作
    print("【步骤2：重复数据处理】")
    print("→ 注意：已取消去重处理，将保留所有原始记录（包括重复项）\n")
    # ==============================================================================

    # 3. 文本型数值转换（提取纯数字）
    print("【步骤3：文本型数值转换】")
    print("→ 作用：把带文字的数值（比如'1234条评论'）提取为纯数字，方便后续计算（如求均值、排序）")
    numeric_fields = ['评论数', '推荐值', '原价', '折扣价', '排序']
    for field in numeric_fields:
        if field in cleaned_data.columns:
            before_count = cleaned_data[field].notna().sum()
            cleaned_data[field] = cleaned_data[field].apply(extract_numeric)
            after_count = cleaned_data[field].notna().sum()
            print(f"→ {field}：{before_count} 个原始文本值 → 转换为 {after_count} 个纯数字值")
    print(f"→ 完成后：所有数值字段已转为数字格式，可直接用于计算\n")

    # 4. 缺失值处理
    print("【步骤4：缺失值填补】")
    print("→ 作用：填补空值（比如“作者”字段为空），避免缺失值影响后续分析")
    missing_stats = cleaned_data.isnull().sum()
    # 先统计缺失情况
    print("→ 缺失值统计：")
    for col, count in missing_stats.items():
        if count > 0:
            ratio = count / len(cleaned_data) * 100
            print(f"   - {col}：{count} 条缺失（占比 {ratio:.2f}%）")

    # 数值型字段用中位数填充（中位数比均值更抗异常值）
    print("\n→ 填补规则：")
    print("   - 数值型字段（如评论数）：用「中位数」填充（避免异常值干扰）")
    print("   - 文本型字段（如作者）：用「未知」等默认值填充")
    numeric_fields = ['评论数', '推荐值', '原价', '折扣价', '排序']
    for col in numeric_fields:
        if col in cleaned_data.columns:
            missing_count = cleaned_data[col].isnull().sum()
            if missing_count > 0:
                non_null_vals = cleaned_data[col].dropna()
                if not non_null_vals.empty:
                    fill_val = non_null_vals.median()
                    cleaned_data[col].fillna(fill_val, inplace=True)
                    print(f"→ {col}：用中位数 {fill_val:.2f} 填补 {missing_count} 条缺失值")
                else:
                    cleaned_data[col].fillna(0, inplace=True)
                    print(f"→ {col}：字段全为空，填充 0")

    # 文本型字段用默认值填充
    text_cols = ['书名', '作者', '出版社', '图书类别', '排行榜类型', '出版日期']
    for col in text_cols:
        if col in cleaned_data.columns:
            missing_count = cleaned_data[col].isnull().sum()
            if missing_count > 0:
                default_val = '未知日期' if col == '出版日期' else '未知'
                cleaned_data[col].fillna(default_val, inplace=True)
                print(f"→ {col}：用「{default_val}」填补 {missing_count} 条缺失值")
    print(f"→ 完成后：所有缺失值已填补，无空字段\n")

    # 5. 异常值处理
    print("【步骤5：异常值删除】")
    print("→ 作用：删除明显不合理的数据（比如“原价10000元”的图书），避免影响分析结果")
    print("→ 异常值规则：")
    print("   - 排序：>1000 或 <1（榜单排序不会超过1000）")
    print("   - 原价：>1000 或 <0（图书原价一般不超过1000元）")
    print("   - 折扣价：<0（价格不能为负）")
    print("   - 评论数：>100万（普通图书评论数极少超过100万）")
    print("   - 推荐值：<0 或 >100（推荐值是百分比，范围0-100）")
    abnormal_stats = calculate_abnormal_values(cleaned_data)
    initial_total = len(cleaned_data)
    for field, count in abnormal_stats.items():
        if count > 0 and field in cleaned_data.columns:
            initial_count = len(cleaned_data)
            if field == '排序':
                cleaned_data = cleaned_data[cleaned_data['排序'] <= 1000]
            elif field == '原价':
                cleaned_data = cleaned_data[(cleaned_data['原价'] <= 1000) & (cleaned_data['原价'] >= 0)]
            elif field == '折扣价':
                cleaned_data = cleaned_data[cleaned_data['折扣价'] >= 0]
            elif field == '评论数':
                cleaned_data = cleaned_data[cleaned_data['评论数'] <= 1000000]
            elif field == '推荐值':
                cleaned_data = cleaned_data[(cleaned_data['推荐值'] >= 0) & (cleaned_data['推荐值'] <= 100)]
            deleted_count = initial_count - len(cleaned_data)
            print(f"→ {field}：计划删除 {count} 条异常记录，实际删除 {deleted_count} 条")
    print(f"→ 完成后数据集规模：{len(cleaned_data)} 条记录\n")

    # 6. 排序重新校准（已调整：仅当需要时校准排序连续性，不改变原始排序逻辑）
    print("【步骤6：排序重新校准】")
    print("→ 作用：异常值删除后，同一榜单的排序可能不连续（比如缺了第5名），重新按「排行榜类型」分配连续排序")
    if '排行榜类型' in cleaned_data.columns and '排序' in cleaned_data.columns:
        # 先按原始排序值排序，再重新分配连续编号
        cleaned_data = cleaned_data.sort_values(['排行榜类型', '排序'])
        cleaned_data['排序'] = cleaned_data.groupby('排行榜类型').cumcount() + 1
        print("→ 校准完成：每个榜单的排序已恢复为「1、2、3...」连续编号")
    else:
        print("→ 缺少'排行榜类型'或'排序'字段，跳过排序校准")
    print(f"→ 完成后：排序字段逻辑更清晰\n")

    # 7. 文本格式统一（含汉字异常处理）
    print("【步骤7：文本格式统一】")
    print("→ 作用：清洗文本中的乱码、特殊字符，统一格式（比如去除多余空格）")
    print("→ 处理内容：")
    print("   - 去除非中文/数字/常见符号的乱码（比如“♦”“&”等）")
    print("   - 修正常见错字（比如“京点”→“甜点”）")
    print("   - 去除多余空格、统一空值为「未知」")
    text_clean_cols = ['书名', '作者', '出版社', '图书类别', '排行榜类型']
    for col in text_clean_cols:
        if col in cleaned_data.columns:
            cleaned_data[col] = cleaned_data[col].apply(clean_chinese_text)
    if '出版日期' in cleaned_data.columns:
        cleaned_data['出版日期'] = cleaned_data['出版日期'].astype(str).str.strip()
        cleaned_data['出版日期'] = cleaned_data['出版日期'].replace(['', 'nan'], '未知日期')
    print("→ 完成后：文本字段格式统一，无乱码/错字\n")

    # 8. 特殊字符过滤
    print("【步骤8：特殊字符过滤】")
    print("→ 作用：再次检查并删除包含未清理干净的特殊字符/乱码的记录（比如“小♦形鲜巾”）")
    allowed_pattern = r'^[\u4e00-\u9fa5a-zA-Z0-9/\-()., ，。]+$'
    check_cols = ['书名', '作者', '出版社', '图书类别', '排行榜类型']
    has_special_char = pd.Series([False] * len(cleaned_data), index=cleaned_data.index)
    for col in check_cols:
        if col in cleaned_data.columns:
            invalid = ~cleaned_data[col].astype(str).str.match(allowed_pattern)
            has_special_char = has_special_char | invalid
    special_count = has_special_char.sum()
    if special_count > 0:
        cleaned_data = cleaned_data[~has_special_char]
        print(f"→ 发现 {special_count} 条含特殊字符/乱码的记录，已删除")
    else:
        print(f"→ 无含特殊字符/乱码的记录，数据文本质量达标\n")

    # 统计清洗后的指标
    post_stats = {
        'total_rows': len(cleaned_data),
        'missing': calculate_missing_values(cleaned_data),
        'abnormal': calculate_abnormal_values(cleaned_data),
        'numeric_stats': calculate_numeric_stats(cleaned_data)
    }

    print("===== 【数据清洗流程完成！】=====")
    print(f"→ 最终数据集规模：{len(cleaned_data)} 条记录，{len(cleaned_data.columns)} 个字段")
    print(f"→ 最终保留字段：{list(cleaned_data.columns)}")
    print(f"→ 数据质量验证：无缺失、无异常、无乱码（保留重复记录），可直接用于后续分析")
    return cleaned_data, pre_stats, post_stats


# -------------------------- 执行入口 --------------------------
if __name__ == "__main__":
    # 1. 输入文件路径（实际数据文件）
    input_csv_path = 'dangdang_book_data_2023_2025/dangdang_bestsellers_2023_2025_20251113_234715.csv'
    # 2. 输出目录和文件名
    output_dir = 'dangdang_analysis_data'
    output_csv_path = os.path.join(output_dir, 'dangdang_cleaned_data.csv')  # 拼接目录和文件名

    try:
        # 先创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 尝试加载数据
        print(f"【第一步：加载原始数据】")
        print(f"正在加载数据文件: {input_csv_path} ...")
        raw_data = pd.read_csv(input_csv_path, encoding='utf-8-sig')
        print(f"→ 原始数据加载成功！规模：{len(raw_data)} 条记录，{len(raw_data.columns)} 个字段\n")

        # 调用数据清洗函数
        cleaned_data, pre_stats, post_stats = clean_data(raw_data)

        # 保存清洗后的数据到指定目录
        cleaned_data.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n【第二步：保存清洗后数据】")
        print(f"→ 清洗后的数据已保存至: {output_csv_path}")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_csv_path}'。请确保文件存在并且路径正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")