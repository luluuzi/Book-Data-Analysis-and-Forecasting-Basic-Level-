import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


# ==================== 参数化配置类 ====================
class RFConfig:
    """随机森林动态预测配置类（仅保留预测功能）"""
    # 带时间维度的重复数据集（动态预测核心）
    DATA_PATH = 'dangdang_analysis_data/featured_dangdang_data_with_duplicates.csv'
    MODEL_SAVE_DIR = 'saved_models'
    PREDICTION_SAVE_PATH = 'prediction_results/rf_predictions_with_details.csv'

    # 特征配置：保留所有动态特征
    EXCLUDE_COLUMNS = ['书名', '作者', '出版社', '出版日期']
    CAT_FEATURES = ['图书类别', '价格区间', '排行榜类型', '销量等级']
    # 缺失值填充策略
    NUM_IMPUTE_STRATEGY = 'median'
    CAT_IMPUTE_STRATEGY = 'most_frequent'

    # 模型配置（使用经验最优参数，无需评估调优）
    REG_TARGET = '排序'
    REG_PARAMS = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 8,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }

    CLF_TARGET = '是否畅销书'
    CLF_PARAMS = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 8,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    # 预测配置（PROGRESS_STEP改为10，简洁进度显示）
    RANDOM_STATE = 42
    FUTURE_TIME_INDEX = None  # 动态获取
    PROGRESS_STEP = 10  # 每处理10本书打印一次进度（可根据需求调整）


# ==================== 核心预测类 ====================
class RandomForestDynamicPredictor:
    """随机森林动态排名预测器（简洁进度显示）"""

    def __init__(self, config=RFConfig()):
        self.config = config
        self.reg_pipeline = None  # 回归流水线（预处理+模型）
        self.clf_pipeline = None  # 分类流水线（预处理+模型）
        self.reg_feature_columns = []
        self.clf_feature_columns = []

        # 创建必要目录
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.PREDICTION_SAVE_PATH), exist_ok=True)

    def _get_feature_columns(self, df, task_type):
        """获取任务对应的特征列（保留动态特征）"""
        exclude = self.config.EXCLUDE_COLUMNS.copy()
        if task_type == 'regression':
            exclude.extend([self.config.REG_TARGET, '标准化排名', '排名分数', self.config.CLF_TARGET, '是否爆款书', '销量等级'])
        else:
            exclude.extend([self.config.CLF_TARGET, '是否爆款书', '销量等级', self.config.REG_TARGET, '标准化排名', '排名分数'])

        # 数值特征（含动态特征）
        num_features = [col for col in df.columns if col not in exclude and df[col].dtype in ['int64', 'float64']]
        # 分类特征
        cat_features = [col for col in self.config.CAT_FEATURES if col in df.columns and col not in exclude]
        return num_features, cat_features

    def _build_preprocessor(self, num_features, cat_features):
        """构建特征预处理流水线"""
        # 数值特征：填充缺失值→标准化
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config.NUM_IMPUTE_STRATEGY)),
            ('scaler', StandardScaler())
        ])

        # 分类特征：填充缺失值→独热编码
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config.CAT_IMPUTE_STRATEGY)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # 合并处理
        preprocessor = ColumnTransformer(
            transformers=[('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)],
            remainder='drop'
        )
        return preprocessor

    def train_models(self, df):
        """训练模型（无评估，全量数据训练）"""
        print("[步骤1/3] 开始训练模型（全量数据，无评估）...")

        # 数据验证（确保时间维度存在）
        if '时间序号' not in df.columns:
            raise ValueError("数据集缺少'时间序号'字段，无法进行动态预测")
        if self.config.REG_TARGET not in df.columns or self.config.CLF_TARGET not in df.columns:
            raise ValueError("数据集缺少目标变量（排序/是否畅销书）")

        # 1. 回归模型训练（预测排名）
        reg_num_feats, reg_cat_feats = self._get_feature_columns(df, 'regression')
        self.reg_feature_columns = reg_num_feats + reg_cat_feats
        X_reg = df[self.reg_feature_columns]
        y_reg = df[self.config.REG_TARGET]

        reg_preprocessor = self._build_preprocessor(reg_num_feats, reg_cat_feats)
        self.reg_pipeline = Pipeline(steps=[
            ('preprocessor', reg_preprocessor),
            ('regressor', RandomForestRegressor(**self.config.REG_PARAMS))
        ])
        self.reg_pipeline.fit(X_reg, y_reg)  # 全量数据训练
        print(f"回归模型训练完成（特征数：{len(self.reg_feature_columns)}）")

        # 2. 分类模型训练（预测是否畅销书）
        clf_num_feats, clf_cat_feats = self._get_feature_columns(df, 'classification')
        self.clf_feature_columns = clf_num_feats + clf_cat_feats
        X_clf = df[self.clf_feature_columns]
        y_clf = df[self.config.CLF_TARGET]

        clf_preprocessor = self._build_preprocessor(clf_num_feats, clf_cat_feats)
        self.clf_pipeline = Pipeline(steps=[
            ('preprocessor', clf_preprocessor),
            ('classifier', RandomForestClassifier(**self.config.CLF_PARAMS))
        ])
        self.clf_pipeline.fit(X_clf, y_clf)  # 全量数据训练
        print(f"分类模型训练完成（特征数：{len(self.clf_feature_columns)}）")

        # 保存模型
        self.save_models()

    def _get_future_time_index(self, df):
        """动态获取未来时间序号（最大时间序号+1）"""
        self.config.FUTURE_TIME_INDEX = df['时间序号'].max() + 1
        print(f"\n预测目标时间序号：{self.config.FUTURE_TIME_INDEX}")

    def predict_future_ranking(self, df):
        """核心功能：预测未来时间点的排名（简洁进度显示）"""
        print(f"\n[步骤2/3] 基于书籍最新动态数据预测未来排名...")

        # 获取所有独特书籍名称
        unique_books = df['书名'].unique()
        total_books = len(unique_books)
        print(f"共需处理 {total_books} 本独特书籍，每{self.config.PROGRESS_STEP}本打印一次进度...")

        # 初始化存储结果的列表
        book_results = []

        # 遍历每本书，筛选最新数据并预测（简洁进度显示）
        for i, book_name in enumerate(unique_books):
            # 每处理PROGRESS_STEP本书打印一次进度（仅显示处理数量）
            if i % self.config.PROGRESS_STEP == 0:
                print(f"  进度：{i}/{total_books} 本")

            # 筛选当前书籍的所有时间点数据，取最新一条（时间序号最大）
            book_data = df[df['书名'] == book_name]
            latest_data = book_data.sort_values('时间序号').iloc[-1]  # 最新数据（Series格式）

            # 关键修复：将Series转为DataFrame（保持列名，适配模型输入）
            latest_df = latest_data.to_frame().T  # T转置后列名就是原Series的索引（特征名）

            # 准备当前书籍的预测特征（直接用训练时确定的特征列）
            reg_required = self.reg_feature_columns  # 回归模型需要的特征列
            clf_required = self.clf_feature_columns  # 分类模型需要的特征列

            # 单本书预测（传入DataFrame，而非numpy数组）
            # 排名预测（修正为≥1的整数）
            rank_raw = self.reg_pipeline.predict(latest_df[reg_required])[0]
            rank_raw = max(1, round(rank_raw))  # 确保≥1

            # 畅销书预测（结果+概率）
            is_best = self.clf_pipeline.predict(latest_df[clf_required])[0]
            is_best_prob = self.clf_pipeline.predict_proba(latest_df[clf_required])[0, 1].round(4)

            # 收集当前书籍的结果
            book_results.append({
                '书名': book_name,
                '作者': latest_data.get('作者', ''),
                '出版社': latest_data.get('出版社', ''),
                '最后出现时间序号': latest_data['时间序号'],
                '预测时间序号': self.config.FUTURE_TIME_INDEX,
                '预测排名（原始）': rank_raw,
                '上期评论数': latest_data.get('上期评论数', 0),
                '上期推荐值': latest_data.get('推荐值', 0),
                '是否畅销书_预测': is_best,
                '是否畅销书_概率': is_best_prob
            })

        # 所有书籍处理完成后，打印完成提示
        print(f"  进度：{total_books}/{total_books} 本（处理完成！）")
        future_results = pd.DataFrame(book_results)

        # 排序并分配唯一排名（原始排名→评论数→推荐值）
        future_results_sorted = future_results.sort_values(
            by=['预测排名（原始）', '上期评论数', '上期推荐值'],
            ascending=[True, False, False]
        ).reset_index(drop=True)
        future_results_sorted['最终预测排名'] = range(1, len(future_results_sorted) + 1)
        future_results_sorted['实际排名'] = np.nan  # 未来数据暂为NaN

        return future_results_sorted

    def save_models(self):
        """保存训练好的模型流水线"""
        reg_path = os.path.join(self.config.MODEL_SAVE_DIR, 'rf_regression_pipeline.pkl')
        clf_path = os.path.join(self.config.MODEL_SAVE_DIR, 'rf_classification_pipeline.pkl')
        joblib.dump(self.reg_pipeline, reg_path)
        joblib.dump(self.clf_pipeline, clf_path)
        print(f"\n模型保存完成：")
        print(f"- 回归模型：{reg_path}")
        print(f"- 分类模型：{clf_path}")

    def save_predictions(self, future_results):
        """保存预测结果（对齐ARIMA格式）"""
        print(f"\n[步骤3/3] 保存预测结果...")
        # 输出字段顺序（与ARIMA完全一致）
        output_columns = [
            '最终预测排名', '书名', '作者', '出版社',
            '最后出现时间序号', '预测时间序号', '预测排名（原始）', '实际排名',
            '是否畅销书_预测', '是否畅销书_概率'
        ]

        # 保存CSV
        future_results[output_columns].to_csv(
            self.config.PREDICTION_SAVE_PATH,
            index=False,
            encoding='utf-8-sig'
        )

        # 预览结果
        print(f"预测结果已保存至：{os.path.abspath(self.config.PREDICTION_SAVE_PATH)}")
        print(f"共预测 {len(future_results)} 本书的排名")
        print("\n前10条预测结果预览：")
        print(future_results[output_columns].head(10).to_string(index=False))

    @classmethod
    def load_and_predict(cls, new_data, model_type='regression', config=RFConfig()):
        """加载已保存的模型进行预测（可选）"""
        model = cls(config=config)
        # 加载模型
        model_path = os.path.join(config.MODEL_SAVE_DIR, f'rf_{model_type}_pipeline.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        pipeline = joblib.load(model_path)
        preprocessor = pipeline.named_steps['preprocessor']
        num_feats = preprocessor.transformers_[0][2]
        cat_feats = preprocessor.transformers_[1][2]
        required_feats = num_feats + cat_feats

        # 验证特征（确保new_data是DataFrame）
        if not isinstance(new_data, pd.DataFrame):
            raise TypeError("new_data必须是pandas DataFrame格式")
        missing_feats = set(required_feats) - set(new_data.columns)
        if missing_feats:
            raise ValueError(f"缺少必要特征：{missing_feats}")

        # 预测
        if model_type == 'regression':
            predictions = pipeline.predict(new_data[required_feats])
            return np.clip(predictions, 1, None).round().astype(int)
        else:
            pred_label = pipeline.predict(new_data[required_feats])
            pred_prob = pipeline.predict_proba(new_data[required_feats])[:, 1].round(4)
            return pred_label, pred_prob


# ==================== 运行入口 ====================
def run_rf_dynamic_prediction():
    """运行随机森林动态预测流程（简洁进度显示）"""
    print("=" * 60)
    print("随机森林 - 书籍动态排名预测（简洁进度显示）")
    print("=" * 60)

    try:
        # 1. 加载数据
        print(f"[加载数据] {RFConfig.DATA_PATH}")
        df = pd.read_csv(RFConfig.DATA_PATH, encoding='utf-8-sig')
        print(f"数据加载成功：形状={df.shape}，独特书籍数={df['书名'].nunique()}，时间序号范围={df['时间序号'].min()}~{df['时间序号'].max()}")

        # 2. 初始化预测器并训练模型
        predictor = RandomForestDynamicPredictor()
        predictor._get_future_time_index(df)  # 获取未来时间点
        predictor.train_models(df)  # 全量数据训练（无评估）

        # 3. 预测未来排名并保存（简洁进度显示）
        future_results = predictor.predict_future_ranking(df)
        predictor.save_predictions(future_results)

        print("\n" + "=" * 60)
        print("✅ 随机森林动态预测流程完成！")
        print(f"📁 预测结果文件：{os.path.abspath(RFConfig.PREDICTION_SAVE_PATH)}")
        print("=" * 60)

        return future_results

    except Exception as e:
        print(f"\n❌ 预测出错：{str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ==================== 执行预测 ====================
if __name__ == "__main__":
    run_rf_dynamic_prediction()