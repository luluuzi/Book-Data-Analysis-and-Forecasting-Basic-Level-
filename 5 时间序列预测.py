import pandas as pd
import numpy as np
import warnings
import os
import traceback
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

class TimeSeriesPredictor:
    """
    时间序列预测模型 - 使用ARIMA算法
    """

    def __init__(self):
        self.models = {}  # 存储为每本书训练的ARIMA模型
        self.is_trained = False
        self.model_name = "ARIMA模型"

    def train(self, df):
        """
        为每本书训练一个ARIMA模型
        """
        print("\n" + "=" * 50)
        print("开始使用ARIMA算法训练时间序列模型")
        print("=" * 50)

        print("步骤 1/3: 对数据进行去重 (按 '书名' 和 '时间序号' 组合) ...")
        initial_count = len(df)
        df_unique = df.drop_duplicates(subset=['书名', '时间序号'], keep='first')
        deduplicated_count = len(df_unique)
        print(f"去重完成：从 {initial_count} 条记录减少到 {deduplicated_count} 条记录。")

        df_unique = df_unique.sort_values(['书名', '时间序号'])
        all_predictions = []

        unique_books = df_unique['书名'].unique()
        print(f"步骤 2/3: 找到 {len(unique_books)} 本独特的书，将为每本书训练一个ARIMA模型...")

        for i, book_name in enumerate(unique_books):
            if i % 50 == 0:
                print(f"已处理 {i}/{len(unique_books)} 本书...")

            book_data = df_unique[df_unique['书名'] == book_name].sort_values('时间序号')

            min_data_points = 5
            if len(book_data) < min_data_points:
                continue

            ts_data = book_data.set_index('时间序号')['排序']

            p, d, q = 1, 1, 1

            try:
                model = ARIMA(ts_data, order=(p, d, q))
                results = model.fit()

                self.models[book_name] = results

                last_record = book_data.iloc[-1]
                last_time_index = last_record['时间序号']
                next_time_index = last_time_index + 1

                forecast = results.get_forecast(steps=1)
                predicted_rank = forecast.predicted_mean.iloc[0]

                # ========================== 核心修改点 ==========================
                # 收集更多书籍信息用于输出
                author = last_record.get('作者', '未知作者')
                publisher = last_record.get('出版社', '未知出版社')
                # =================================================================

                all_predictions.append({
                    '书名': book_name,
                    '作者': author,
                    '出版社': publisher,
                    '最后出现时间序号': last_time_index,
                    '预测时间序号': next_time_index,
                    '预测排名': predicted_rank,
                    '实际排名': np.nan,  # 对于未来预测，实际排名未知
                    '上期评论数': last_record['评论数'],
                    '上期推荐值': last_record['推荐值']
                })

            except Exception as e:
                continue

        if not self.models:
            print(f"错误：未能为任何书籍训练出有效的ARIMA模型。")
            return None

        self.is_trained = True
        print(f"\n步骤 3/3: 训练完成！成功为 {len(self.models)} 本书训练了ARIMA模型。")

        if all_predictions:
            predictions_df = pd.DataFrame(all_predictions)
            # 修改输出文件名以反映新内容
            self._save_predictions_with_unique_ranking(predictions_df, "arima_predictions_with_details.csv")

            results_summary = {
                'model_name': self.model_name,
                'model_type': 'ARIMA',
                'trained_books_count': len(self.models),
                'predictions_count': len(predictions_df)
            }
            return results_summary
        else:
            print("未能生成任何预测结果。")
            return None

    def _save_predictions_with_unique_ranking(self, predictions_df, output_filename):
        """
        对预测结果进行后处理，确保排名唯一，并保存到CSV文件。
        """
        print(f"\n正在对ARIMA预测结果进行后处理，确保排名唯一...")

        predictions_df['预测排名'] = predictions_df['预测排名'].round().astype(int)

        predictions_df_sorted = predictions_df.sort_values(
            by=['预测排名', '上期评论数', '上期推荐值'],
            ascending=[True, False, False]
        ).reset_index(drop=True)

        predictions_df_sorted['最终预测排名'] = range(1, len(predictions_df_sorted) + 1)

        # ========================== 核心修改点 ==========================
        # 定义包含新字段的输出列顺序
        output_columns = [
            '最终预测排名', '书名', '作者', '出版社',
            '最后出现时间序号', '预测时间序号', '预测排名', '实际排名'
        ]
        # =================================================================

        output_dir = "prediction_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, output_filename)
        predictions_df_sorted[output_columns].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"ARIMA预测结果处理完成！已保存至: {output_path}")

def run_time_series_analysis(data_path, output_filename):
    """
    运行时间序列分析的独立函数
    """
    print(f"\n{'=' * 60}")
    print(f"正在使用ARIMA模型处理数据集: {os.path.basename(data_path)}")
    print(f"{'=' * 60}")

    try:
        featured_data = pd.read_csv(data_path, encoding='utf-8-sig')
        print(f"成功加载数据: {featured_data.shape}")

        ts_predictor = TimeSeriesPredictor()
        ts_results = ts_predictor.train(featured_data)

        if ts_results:
            print(f"\nARIMA时间序列分析完成!")
            print(f"成功训练了 {ts_results['trained_books_count']} 个模型。")
            return ts_results
        else:
            print(f"对数据集 {os.path.basename(data_path)} 的ARIMA分析失败。")
            return None

    except FileNotFoundError:
        print(f"错误: 数据文件未找到 - {data_path}")
        return None
    except Exception as e:
        print(f"处理数据集时发生严重错误: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    dataset = {
        'input_path': 'dangdang_analysis_data/featured_dangdang_data_with_duplicates.csv',
        'output_filename': 'arima_predictions_final.csv'
    }

    run_time_series_analysis(dataset['input_path'], dataset['output_filename'])

    print("\n所有ARIMA时间序列分析任务已全部完成！")