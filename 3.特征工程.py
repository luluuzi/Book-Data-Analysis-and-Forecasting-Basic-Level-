# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class DangDangFeatureEngineer:
    """
    当当网畅销书数据特征工程类
    """

    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = []

    def engineer_features(self, cleaned_data):
        """
        执行完整的特征工程流程
        """
        print("开始特征工程...")
        df = cleaned_data.copy()

        # 1. 基础特征工程
        df = self._create_basic_features(df)

        # 2. 时间特征工程
        df = self._create_time_features(df)

        # 3. 价格特征工程
        df = self._create_price_features(df)

        # 4. 文本特征工程
        df = self._create_text_features(df)

        # 5. 作者和出版社特征
        df = self._create_author_publisher_features(df)

        # 6. 时间序列特征
        df = self._create_time_series_features(df)

        # 7. 目标变量创建
        df = self._create_target_variables(df)

        # 8. 特征编码和清理
        df = self._encode_and_clean_features(df)

        print(f"特征工程完成，最终特征数量: {len(self.feature_columns)}")
        print(f"特征列表: {self.feature_columns}")

        return df

    def _create_basic_features(self, df):
        """创建基础特征"""
        print("创建基础特征...")

        # 确保数值字段是数值类型
        numeric_columns = ['排序', '评论数', '推荐值', '原价', '折扣价']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _create_time_features(self, df):
        """创建时间相关特征"""
        print("创建时间特征...")

        # 提取年份和月份
        df['年份'] = df['排行榜类型'].str.extract(r'(\d{4})年').astype(float)
        df['月份'] = df['排行榜类型'].str.extract(r'(\d{1,2})月').fillna(12).astype(int)

        # 创建时间序列标识
        df['时间序号'] = (df['年份'] - 2023) * 12 + df['月份']

        # 季节性特征
        df['季度'] = ((df['月份'] - 1) // 3 + 1).astype(int)
        df['是否年底'] = df['月份'].isin([11, 12]).astype(int)
        df['是否年初'] = df['月份'].isin([1, 2]).astype(int)

        # 榜单类型
        df['是否月度榜单'] = df['排行榜类型'].str.contains('月').astype(int)

        return df

    def _create_price_features(self, df):
        """创建价格相关特征"""
        print("创建价格特征...")

        # 折扣相关特征
        df['折扣率'] = (df['折扣价'] / df['原价']).fillna(1)
        df['折扣率'] = df['折扣率'].clip(0, 1)
        df['价格差'] = df['原价'] - df['折扣价']
        df['是否打折'] = (df['折扣率'] < 0.99).astype(int)
        df['折扣力度'] = (1 - df['折扣率']) * 100  # 百分比折扣

        # 价格区间
        price_bins = [0, 20, 40, 60, 100, 200, 1000]
        price_labels = ['超低价', '低价', '中价', '高价', '超高价', '奢侈价']
        df['价格区间'] = pd.cut(df['原价'], bins=price_bins, labels=price_labels)

        return df

    def _create_text_features(self, df):
        """创建文本相关特征"""
        print("创建文本特征...")

        # 书名特征
        df['书名长度'] = df['书名'].str.len()
        df['书名关键词数'] = df['书名'].str.split().str.len()

        # 作者特征
        df['作者数量'] = df['作者'].str.split('、').str.len().fillna(1)
        df['是否有多个作者'] = (df['作者数量'] > 1).astype(int)

        return df

    def _create_author_publisher_features(self, df):
        """创建作者和出版社影响力特征"""
        print("创建作者和出版社特征...")

        # 作者统计特征
        author_stats = df.groupby('作者').agg({
            '评论数': ['mean', 'max', 'count'],
            '排序': 'mean',
            '推荐值': 'mean'
        })
        author_stats.columns = [
            '作者平均评论数', '作者最高评论数', '作者作品数',
            '作者平均排名', '作者平均推荐值'
        ]
        df = df.merge(author_stats, on='作者', how='left')

        # 出版社统计特征
        publisher_stats = df.groupby('出版社').agg({
            '评论数': ['mean', 'count'],
            '排序': 'mean'
        })
        publisher_stats.columns = [
            '出版社平均评论数', '出版社图书数', '出版社平均排名'
        ]
        df = df.merge(publisher_stats, on='出版社', how='left')

        return df

    def _create_time_series_features(self, df):
        """创建时间序列特征"""
        print("创建时间序列特征...")

        # 按书名和时间排序
        df = df.sort_values(['书名', '时间序号'])

        # 滞后特征
        df['上期排名'] = df.groupby('书名')['排序'].shift(1)
        df['上期评论数'] = df.groupby('书名')['评论数'].shift(1)
        df['排名变化'] = df.groupby('书名')['排序'].diff()

        # 滚动统计特征
        df['近期排名均值'] = df.groupby('书名')['排序'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        df['近期排名标准差'] = df.groupby('书名')['排序'].rolling(3, min_periods=1).std().reset_index(level=0, drop=True)
        df['近期评论均值'] = df.groupby('书名')['评论数'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

        # 历史统计特征
        df['历史排名均值'] = df.groupby('书名')['排序'].expanding().mean().reset_index(level=0, drop=True)
        df['历史评论均值'] = df.groupby('书名')['评论数'].expanding().mean().reset_index(level=0, drop=True)

        # 趋势特征
        df['排名改善趋势'] = df.groupby('书名')['排序'].transform(
            lambda x: -x.diff().rolling(3, min_periods=1).mean()  # 负号表示排名数字越小越好
        )

        return df

    def _create_target_variables(self, df):
        """创建目标变量"""
        print("创建目标变量...")

        # 回归目标
        df['排名分数'] = 1 / df['排序']  # 排名越靠前分数越高
        df['标准化排名'] = (df['排序'].max() - df['排序']) / (df['排序'].max() - df['排序'].min())

        # 分类目标
        df['是否畅销书'] = (df['排序'] <= 50).astype(int)
        df['是否爆款书'] = (df['排序'] <= 10).astype(int)

        # 多分类目标
        conditions = [
            df['排序'] <= 10,
            df['排序'] <= 30,
            df['排序'] <= 100,
            df['排序'] <= 300
        ]
        choices = ['爆款', '畅销', '一般', '长尾']
        df['销量等级'] = np.select(conditions, choices, default='其他')

        print("目标变量统计:")
        print(f"是否畅销书分布: {df['是否畅销书'].value_counts().to_dict()}")
        print(f"销量等级分布: {df['销量等级'].value_counts().to_dict()}")

        return df

    def _encode_and_clean_features(self, df):
        """特征编码和最终清理"""
        print("特征编码和清理...")

        # 类别特征编码
        categorical_columns = ['图书类别', '价格区间', '销量等级']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_编码'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # 定义最终特征列
        self.feature_columns = [
            # 基础特征
            '排序', '评论数', '推荐值', '原价', '折扣价',
            # 时间特征
            '年份', '月份', '时间序号', '季度', '是否年底', '是否年初', '是否月度榜单',
            # 价格特征
            '折扣率', '价格差', '是否打折', '折扣力度',
            # 文本特征
            '书名长度', '书名关键词数', '作者数量', '是否有多个作者',
            # 作者出版社特征
            '作者平均评论数', '作者最高评论数', '作者作品数', '作者平均排名', '作者平均推荐值',
            '出版社平均评论数', '出版社图书数', '出版社平均排名',
            # 时间序列特征
            '上期排名', '上期评论数', '排名变化', '近期排名均值', '近期排名标准差',
            '近期评论均值', '历史排名均值', '历史评论均值', '排名改善趋势',
            # 编码特征
            '图书类别_编码', '价格区间_编码', '销量等级_编码'
        ]

        # 只保留存在的特征列
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]

        # 填充缺失值
        for col in self.feature_columns:
            if df[col].isnull().any():
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(0, inplace=True)

        return df

    def get_feature_summary(self, df):
        """获取特征摘要"""
        summary = {
            '总特征数': len(self.feature_columns),
            '特征列表': self.feature_columns,
            '数据形状': df.shape,
            '数值特征统计': df[self.feature_columns].describe().to_dict()
        }
        return summary


# 使用示例
if __name__ == "__main__":
    # 示例用法
    feature_engineer = DangDangFeatureEngineer()

    # 加载清洗后的数据
    try:
        cleaned_data = pd.read_csv('dangdang_analysis_data/cleaned_dangdang_data.csv')
        featured_data = feature_engineer.engineer_features(cleaned_data)

        # 保存特征工程后的数据
        featured_data.to_csv('dangdang_analysis_data/featured_dangdang_data.csv', index=False)
        print("特征工程完成，数据已保存")

        # 输出特征摘要
        summary = feature_engineer.get_feature_summary(featured_data)
        print(f"特征工程摘要: {summary}")

    except FileNotFoundError:
        print("请先运行数据预处理脚本生成清洗后的数据")