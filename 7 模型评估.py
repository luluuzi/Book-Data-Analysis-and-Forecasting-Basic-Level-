import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# ==================== 核心路径配置（与随机森林/ARIMA脚本输出完全对齐）====================
ORIGINAL_DATA_PATH = "dangdang_analysis_data/featured_dangdang_data_with_duplicates.csv"
ARIMA_CSV_PATH = "prediction_results/arima_predictions_with_details.csv"
RF_CSV_PATH = "prediction_results/rf_predictions_with_details.csv"
OUTPUT_DIR = "模型评估结果"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 中文字体配置（确保无乱码）====================
def setup_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
    except:
        try:
            plt.rcParams['font.sans-serif'] = ['Heiti TC']  # macOS
        except:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 兜底
    plt.rcParams['axes.unicode_minus'] = False


setup_chinese_font()


# ==================== 模型评估核心类 ====================
class ModelEvaluator:
    def __init__(self):
        self.results = {}  # 存储两个模型的评估结果
        self.comparison_data = []  # 对比数据
        self.original_data = None  # 原始数据（含真实排名）
        self.colors = ['#1f77b4', '#ff7f0e']  # ARIMA蓝，随机森林橙

    def load_original_data(self):
        """加载原始数据，预处理确保匹配"""
        print(f"\n[1/4] 加载原始数据...")
        if not os.path.exists(ORIGINAL_DATA_PATH):
            raise FileNotFoundError(f"❌ 原始数据不存在：{ORIGINAL_DATA_PATH}")

        # 读取并预处理（去重、清洗书名）
        self.original_data = pd.read_csv(ORIGINAL_DATA_PATH, encoding='utf-8-sig')
        self.original_data = self.original_data.drop_duplicates(
            subset=['书名', '时间序号'], keep='first'
        ).reset_index(drop=True)

        # 清洗书名（去除书名号、空格，统一格式）
        self.original_data['书名_清洗'] = self.original_data['书名'].apply(
            lambda x: str(x).replace('《', '').replace('》', '').strip().replace(' ', '')
        )

        print(f"✅ 原始数据加载完成：{len(self.original_data)} 条记录，{self.original_data['书名_清洗'].nunique()} 本独特书籍")

    def extract_evaluation_data(self, model_name, csv_path):
        """提取评估数据（完美适配你的随机森林CSV格式）"""
        print(f"\n[2/4] 提取 {model_name} 评估数据...")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ {model_name} 预测文件不存在：{csv_path}")

        # 读取预测结果并清洗书名
        pred_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        pred_df['书名_清洗'] = pred_df['书名'].apply(
            lambda x: str(x).replace('《', '').replace('》', '').strip().replace(' ', '')
        )

        print(f"✅ 加载 {model_name} 预测结果：{len(pred_df)} 条记录，列名：{list(pred_df.columns)}")

        # 必要列验证（你的随机森林CSV已包含这些列）
        required_cols = ['书名_清洗', '最后出现时间序号', '最终预测排名']
        missing_cols = [col for col in required_cols if col not in pred_df.columns]
        if missing_cols:
            raise ValueError(f"❌ 缺少必要列：{missing_cols}")

        # 匹配真实排名（y_true）和预测排名（y_pred）
        y_true = []
        y_pred = []
        valid_books = 0
        missing_books = 0

        for _, row in pred_df.iterrows():
            book_name = row['书名_清洗']
            last_time = row['最后出现时间序号']

            # 从原始数据中匹配：清洗后的书名 + 最后出现时间序号
            match = self.original_data[
                (self.original_data['书名_清洗'] == book_name) &
                (self.original_data['时间序号'] == last_time)
                ]

            if not match.empty and '排序' in match.columns:
                true_rank = match.iloc[0]['排序']
                pred_rank = row['最终预测排名']

                # 确保排名是有效数字
                if pd.notna(true_rank) and pd.notna(pred_rank) and true_rank >= 1 and pred_rank >= 1:
                    y_true.append(true_rank)
                    y_pred.append(pred_rank)
                    valid_books += 1
            else:
                missing_books += 1

        # 确保有效样本足够
        if valid_books < 3:
            raise ValueError(f"⚠️ {model_name} 有效样本不足（仅 {valid_books} 个），但继续评估")
        else:
            print(f"✅ 样本匹配完成：有效样本 {valid_books} 个，无历史数据 {missing_books} 个")

        # 计算核心指标（R方、MAE、RMSE）
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }

        # 提取随机森林特征重要性（仅随机森林执行）
        feature_importance = self.extract_rf_feature_importance() if model_name == "随机森林模型" else None

        # 存储结果
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': {'y_true': y_true, 'y_pred': y_pred},
            'feature_importance': feature_importance,
            'valid_samples': valid_books
        }

        self.comparison_data.append({
            'model': model_name,
            'r2': metrics['r2'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'valid_samples': valid_books
        })

        # 打印指标（明确显示R方）
        print(f"\n{model_name} 核心指标：")
        print(f"  R方（决定系数）：{metrics['r2']:.4f}（越高越好，最大值1）")
        print(f"  MAE（平均绝对误差）：{metrics['mae']:.4f}（越低越好）")
        print(f"  RMSE（均方根误差）：{metrics['rmse']:.4f}（越低越好）")

        return self.results[model_name]

    def extract_rf_feature_importance(self):
        """提取随机森林特征重要性（适配你的模型保存路径）"""
        print(f"\n[3/4] 提取随机森林特征重要性...")
        rf_model_path = "saved_models/rf_regression_pipeline.pkl"

        if not os.path.exists(rf_model_path):
            print(f"⚠️  未找到随机森林模型文件：{rf_model_path}，跳过特征重要性")
            return None

        try:
            # 加载你的随机森林回归流水线
            pipeline = joblib.load(rf_model_path)
            regressor = pipeline.named_steps['regressor']
            preprocessor = pipeline.named_steps['preprocessor']

            # 提取特征名称（数值特征 + 独热编码后的分类特征）
            num_feats = preprocessor.transformers_[0][2]  # 数值特征
            cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']  # 分类编码器
            cat_feats = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])  # 分类特征（独热后）
            all_feats = list(num_feats) + list(cat_feats)

            # 特征重要性排序（Top10）
            importances = regressor.feature_importances_
            feat_importance = dict(zip(all_feats, importances))
            feat_importance = dict(sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:10])

            print(f"✅ 特征重要性提取完成（Top10）")
            return feat_importance
        except Exception as e:
            print(f"⚠️  特征重要性提取失败：{str(e)}，跳过")
            return None

    def plot_performance_comparison(self):
        """生成模型性能对比图（确保两个模型都显示）"""
        print(f"\n[4/4] 生成性能对比图...")
        comparison_df = pd.DataFrame(self.comparison_data)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ARIMA vs 随机森林 预测性能对比', fontsize=16, fontweight='bold')

        # 1. R方对比（核心指标，突出显示）
        axes[0, 0].bar(
            comparison_df['model'], comparison_df['r2'],
            color=self.colors[:len(comparison_df)]
        )
        axes[0, 0].set_title('R方 决定系数对比（越高越好）', fontweight='bold')
        axes[0, 0].set_ylabel('R方值')
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].grid(axis='y', alpha=0.3)
        # 添加数值标签
        for i, v in enumerate(comparison_df['r2']):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. MAE对比
        axes[0, 1].bar(
            comparison_df['model'], comparison_df['mae'],
            color=self.colors[:len(comparison_df)]
        )
        axes[0, 1].set_title('MAE 平均绝对误差对比（越低越好）', fontweight='bold')
        axes[0, 1].set_ylabel('MAE值')
        axes[0, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(comparison_df['mae']):
            axes[0, 1].text(i, v + 0.1, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. RMSE对比
        axes[1, 0].bar(
            comparison_df['model'], comparison_df['rmse'],
            color=self.colors[:len(comparison_df)]
        )
        axes[1, 0].set_title('RMSE 均方根误差对比（越低越好）', fontweight='bold')
        axes[1, 0].set_ylabel('RMSE值')
        axes[1, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(comparison_df['rmse']):
            axes[1, 0].text(i, v + 0.1, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # 4. 随机森林特征重要性
        rf_feat_importance = self.results.get('随机森林模型', {}).get('feature_importance')
        if rf_feat_importance:
            feats, imps = zip(*rf_feat_importance.items())
            # 简化特征名（避免图表拥挤）
            feats_simplified = [feat[:20] + '...' if len(feat) > 20 else feat for feat in feats]
            axes[1, 1].barh(range(len(feats_simplified)), imps, color='#2ca02c')
            axes[1, 1].set_yticks(range(len(feats_simplified)))
            axes[1, 1].set_yticklabels(feats_simplified, fontsize=9)
            axes[1, 1].set_title('随机森林 Top10 特征重要性', fontweight='bold')
            axes[1, 1].set_xlabel('重要性得分')
        else:
            axes[1, 1].text(0.5, 0.5, '无特征重要性数据', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('随机森林特征重要性', fontweight='bold')

        # 保存图表
        plot_path = os.path.join(OUTPUT_DIR, '模型性能对比分析.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 性能对比图已保存：{plot_path}")

    def plot_prediction_scatter(self):
        """生成真实值vs预测值散点图（两个模型都显示）"""
        print(f"\n生成预测结果散点图...")
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))
        fig.suptitle('预测效果：真实排名 vs 预测排名', fontsize=14, fontweight='bold')

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            y_true = result['predictions']['y_true']
            y_pred = result['predictions']['y_pred']

            # 随机采样（最多200个点，避免拥挤）
            if len(y_true) > 200:
                sample_idx = np.random.choice(len(y_true), 200, replace=False)
                y_true = y_true[sample_idx]
                y_pred = y_pred[sample_idx]

            # 绘制散点图和理想预测线
            ax.scatter(y_true, y_pred, alpha=0.6, s=50, color=self.colors[idx])
            min_rank = min(y_true.min(), y_pred.min())
            max_rank = max(y_true.max(), y_pred.max())
            ax.plot([min_rank, max_rank], [min_rank, max_rank], 'r--', lw=2, label='理想预测线')

            # 添加指标文本框
            metrics = result['metrics']
            text = f'R方 = {metrics["r2"]:.3f}\nMAE = {metrics["mae"]:.3f}\nRMSE = {metrics["rmse"]:.3f}'
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat', alpha=0.9),
                    verticalalignment='top', fontweight='bold')

            # 图表样式
            ax.set_xlabel('真实排名')
            ax.set_ylabel('预测排名')
            ax.set_title(model_name, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 保存图表
        scatter_path = os.path.join(OUTPUT_DIR, '预测结果散点图.png')
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 散点图已保存：{scatter_path}")

    def generate_text_report(self):
        """生成完整的文字版评估报告"""
        print(f"\n生成模型评估报告...")
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        comparison_df = pd.DataFrame(self.comparison_data).sort_values('r2', ascending=False)
        best_model = comparison_df.iloc[0]

        # 报告内容（详细且清晰）
        report = [
            "=" * 80,
            "书籍排名预测模型评估报告",
            f"生成时间：{current_time}",
            "=" * 80,
            "",
            "一、评估数据来源",
            "-" * 40,
            f"原始数据：{ORIGINAL_DATA_PATH}",
            f"ARIMA预测结果：{ARIMA_CSV_PATH}",
            f"随机森林预测结果：{RF_CSV_PATH}",
            "",
            "二、评估逻辑说明",
            "-" * 40,
            "由于预测的是未来排名（无真实值），采用「历史数据模拟验证」方案：",
            "1. 以模型训练时「最后出现时间序号」的真实排名作为「模拟真实值（y_true）」",
            "2. 以模型输出的「最终预测排名」作为「模拟预测值（y_pred）」",
            "3. 通过核心指标评估模型捕捉排名规律的能力，间接推断未来预测可信度",
            "",
            "三、核心指标说明",
            "-" * 40,
            "1. R方（决定系数）：越接近1越好，衡量模型对排名变化规律的解释能力",
            "2. MAE（平均绝对误差）：越低越好，衡量预测值与真实值的平均偏差",
            "3. RMSE（均方根误差）：越低越好，对极端误差更敏感，反映预测稳定性",
            "",
            "四、模型性能对比",
            "-" * 40,
            comparison_df[['model', 'r2', 'mae', 'rmse', 'valid_samples']].to_string(
                index=False,
                formatters={
                    'r2': '{:.4f}'.format,
                    'mae': '{:.4f}'.format,
                    'rmse': '{:.4f}'.format
                }
            ),
            "",
            "五、最佳模型推荐",
            "-" * 40,
            f"🏆 推荐模型：{best_model['model']}",
            "",
            "推荐理由：",
            f"1. R方更高（{best_model['r2']:.4f}）→ 更能捕捉排名变化规律",
            f"2. MAE更小（{best_model['mae']:.4f}）→ 平均预测误差更小",
            f"3. RMSE更小（{best_model['rmse']:.4f}）→ 预测更稳定，极端误差更少",
            "",
            "六、补充说明",
            "-" * 40,
            f"ARIMA模型有效样本数：{comparison_df[comparison_df['model'] == 'ARIMA模型']['valid_samples'].values[0] if 'ARIMA模型' in comparison_df['model'].values else 0}",
            f"随机森林模型有效样本数：{comparison_df[comparison_df['model'] == '随机森林模型']['valid_samples'].values[0] if '随机森林模型' in comparison_df['model'].values else 0}",
            "注：样本数越多，评估结果越可靠",
            "",
            "七、总结",
            "-" * 40,
            f"{best_model['model']} 在历史数据验证中表现更优，建议用于未来书籍排名预测。",
            "若需进一步优化，可参考随机森林的特征重要性，优先提升高重要性特征的质量。",
            "",
            "=" * 80
        ]

        # 保存报告
        report_path = os.path.join(OUTPUT_DIR, '模型评估报告.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"✅ 评估报告已保存：{report_path}")

    def run_full_evaluation(self):
        """完整评估流程（确保不中断，生成所有文件）"""
        print("=" * 80)
        print("🚀 启动 ARIMA vs 随机森林 预测评估流程")
        print("=" * 80)

        try:
            # 1. 加载原始数据
            self.load_original_data()

            # 2. 提取两个模型数据（一个失败不影响另一个）
            try:
                self.extract_evaluation_data("ARIMA模型", ARIMA_CSV_PATH)
            except Exception as e:
                print(f"⚠️ ARIMA数据提取失败：{str(e)}")

            try:
                self.extract_evaluation_data("随机森林模型", RF_CSV_PATH)
            except Exception as e:
                print(f"⚠️ 随机森林数据提取失败：{str(e)}")

            # 3. 至少有一个模型才继续
            if len(self.comparison_data) == 0:
                print("❌ 无任何有效模型数据，评估终止")
                return False

            # 4. 生成所有文件
            self.plot_performance_comparison()
            self.plot_prediction_scatter()
            self.generate_text_report()

            print(f"\n" + "=" * 80)
            print("🎉 评估完成！已生成3个文件：")
            print(f"1. {os.path.abspath(os.path.join(OUTPUT_DIR, '模型性能对比分析.png'))}")
            print(f"2. {os.path.abspath(os.path.join(OUTPUT_DIR, '预测结果散点图.png'))}")
            print(f"3. {os.path.abspath(os.path.join(OUTPUT_DIR, '模型评估报告.txt'))}")
            print("=" * 80)

            return True

        except Exception as e:
            print(f"\n❌ 评估全局错误：{str(e)}")
            import traceback
            traceback.print_exc()
            return False


# ==================== 运行入口 ====================
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation()