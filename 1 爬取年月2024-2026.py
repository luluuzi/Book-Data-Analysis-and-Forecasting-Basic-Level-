import os
import random
import time
import json
import pandas as pd
from lxml import etree
import requests
from datetime import datetime

# 忽略requests的SSL验证警告
requests.packages.urllib3.disable_warnings()

# -------------------------- 配置参数 --------------------------
cookies = {
    'ddscreen': '2',
    'dest_area': 'country_id%3D9000%26province_id%3D111%26city_id%20%3D0%26district_id%3D0%26town_id%3D0',
    '__permanent_id': '20240423210658530124490989268736883',
    'MDD_channelId': '70000',
    'MDD_fromPlatform': '307',
    '__visit_id': '20240530154038979262380281306734049',
    '__out_refer': '',
    '__rpm': '...1717054859559%7C...1717054899777',
    '__trace_id': '20240530154142377181404279783243769',
}

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
}

# 输出目录和进度文件路径
OUTPUT_DIR = 'dangdang_book_data_2024_onward'  # 修改目录名以区分
PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'crawl_progress.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 近三年年度榜单（2023-2025年）- 保留完整配置，后续在main中筛选
year_data = {
    "2023年": ("year", 2023, None),
    "2024年": ("year", 2024, None),
    "2025年": ("year", 2025, None)
}

# 2025年1-10月月度榜单
month_data = {
    "2025年1月": ("month", 2025, 1),
    "2025年2月": ("month", 2025, 2),
    "2025年3月": ("month", 2025, 3),
    "2025年4月": ("month", 2025, 4),
    "2025年5月": ("month", 2025, 5),
    "2025年6月": ("month", 2025, 6),
    "2025年7月": ("month", 2025, 7),
    "2025年8月": ("month", 2025, 8),
    "2025年9月": ("month", 2025, 9),
    "2025年10月": ("month", 2025, 10)
}

all_rank_data = {**year_data, **month_data}

# 目标分类列表（保持不变）
target_categories = [
    ('01.41.00.00.00.00', '童书'),
    ('01.43.00.00.00.00', '中小学用书'),
    ('01.03.00.00.00.00', '小说'),
    ('01.05.00.00.00.00', '文学'),
    ('01.45.00.00.00.00', '外语'),
    ('01.21.00.00.00.00', '成功/励志'),
    ('01.09.00.00.00.00', '动漫/幽默'),
    ('01.36.00.00.00.00', '历史'),
    ('01.28.00.00.00.00', '哲学/宗教'),
    ('01.31.00.00.00.00', '心理学'),
    ('01.22.00.00.00.00', '管理'),
    ('01.01.00.00.00.00', '青春文学'),
    ('01.07.00.00.00.00', '艺术'),
    ('01.15.00.00.00.00', '亲子/家教'),
    ('01.47.00.00.00.00', '考试'),
    ('01.18.00.00.00.00', '保健/养生'),
    ('01.27.00.00.00.00', '政治/军事'),
    ('01.49.00.00.00.00', '教材'),
    ('01.52.00.00.00.00', '科普读物'),
    ('01.30.00.00.00.00', '社会科学'),
    ('01.38.00.00.00.00', '传记'),
    ('01.32.00.00.00.00', '古籍'),
    ('01.24.00.00.00.00', '投资理财'),
    ('01.26.00.00.00.00', '法律'),
    ('01.25.00.00.00.00', '经济'),
    ('01.54.00.00.00.00', '计算机/网络'),
    ('01.56.00.00.00.00', '医学'),
    ('01.34.00.00.00.00', '文化'),
    ('01.12.00.00.00.00', '旅游/地图'),
    ('01.17.00.00.00.00', '育儿/早教'),
    ('01.63.00.00.00.00', '工业技术'),
    ('01.50.00.00.00.00', '工具书'),
    ('01.06.00.00.00.00', '孕产/胎教'),
    ('01.19.00.00.00.00', '体育/运动'),
    ('01.04.00.00.00.00', '休闲/爱好'),
    ('01.10.00.00.00.00', '烹饪/美食'),
    ('01.20.00.00.00.00', '手工/DIY'),
    ('01.62.00.00.00.00', '自然科学'),
    ('01.55.00.00.00.00', '建筑'),
    ('01.16.00.00.00.00', '两性关系'),
    ('01.14.00.00.00.00', '家庭/家居'),
    ('01.11.00.00.00.00', '时尚/美妆'),
    ('01.66.00.00.00.00', '农业/林业'),
]


# -------------------------- 核心函数 --------------------------
def load_progress():
    """加载已保存的爬取进度"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            print(f"成功加载爬取进度，已完成 {len(progress)} 个分类-榜单组合")
            return progress
        except Exception as e:
            print(f"加载进度失败：{e}，将重新开始爬取")
    return {}


def save_progress(progress):
    """保存当前爬取进度"""
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存进度失败：{e}")


def spider_data(url):
    """爬取网页内容"""
    response = requests.get(
        url,
        cookies=cookies,
        headers=headers,
        verify=False,
        timeout=15
    )
    response.encoding = response.apparent_encoding
    return response.text


def parse_data(page_text, rank_type, book_category):
    """解析网页数据并返回原始数据的DataFrame"""
    tree = etree.HTML(page_text)
    lis = tree.xpath('.//ul[@class="bang_list clearfix bang_list_mode"]/li')

    data = []
    for li in lis:
        # 提取原始文本
        rank = ''.join(li.xpath('.//div[@class="list_num red" or @class="list_num "]/text()')).replace('.', '')

        # 【核心修改点】优先获取a标签的title属性，这通常是完整的书名
        # 使用 @title 获取属性值
        name = ''.join(li.xpath('.//div[@class="name"]/a/@title'))

        # 如果title属性为空（极少数情况），则回退到获取标签内的文本
        if not name:
            name = ''.join(li.xpath('.//div[@class="name"]/a/text()'))

        comments = ''.join(li.xpath('.//div[@class="star"]/a/text()')).split('条')[0]
        recommends = ''.join(li.xpath('.//div[@class="star"]/span/text()')).split('推荐')[0]
        author = ''.join(li.xpath('.//div[@class="publisher_info"][1]/a[1]/text()'))
        publish_date = ''.join(li.xpath('.//div[@class="publisher_info"][2]/span/text()'))
        publish_house = ''.join(li.xpath('.//div[@class="publisher_info"][2]/a/text()'))
        original_price = ''.join(li.xpath('.//div[@class="price"]/p[1]/span[1]/text()'))
        discount_price = ''.join(li.xpath('.//span[@class="price_r"]/text()'))
        discount = ''.join(li.xpath('.//span[@class="price_s"]/text()'))
        ebook_price = ''.join(
            li.xpath('./div[@class="price"]/p[@class="price_e"]/span[@class="price_n"]/text()'))

        data.append([
            rank_type, book_category,
            rank, name, comments, recommends,
            author, publish_date, publish_house,
            original_price, discount_price, discount, ebook_price
        ])

    # 创建DataFrame
    df = pd.DataFrame(data, columns=[
        '排行榜类型', '图书类别', '排序', '书名', '评论数', '推荐值',
        '作者', '出版日期', '出版社', '原价', '折扣价',
        '折扣比例', '电子书价格'
    ])

    return df

def build_url(bang_type, year, period, page, category_code='01.00.00.00.00.00'):
    """构建当当网榜单URL"""
    if bang_type == "year":
        return f'http://bang.dangdang.com/books/bestsellers/{category_code}-year-{year}-0-1-{page}'
    elif bang_type == "month":
        return f'http://bang.dangdang.com/books/bestsellers/{category_code}-month-{year}-{period}-1-{page}'
    else:
        raise ValueError("未知的榜单类型")


def main():
    """主函数：仅爬取2024年至今的数据（带进度保存）"""
    # 初始化数据存储和进度记录
    columns = [
        '排行榜类型', '图书类别', '排序', '书名', '评论数', '推荐值',
        '作者', '出版日期', '出版社', '原价', '折扣价',
        '折扣比例', '电子书价格'
    ]
    all_data_df = pd.DataFrame(columns=columns)
    crawl_progress = load_progress()

    # 定义起始年份（2024年）
    start_year = 2024
    # 获取当前日期，用于过滤未来的月度榜单（避免爬取未到的月份）
    current_date = datetime.now()

    # 循环爬取每个分类
    for category_code, category_name in target_categories:
        print(f'\n==================================================')
        print(f'=====================开始爬{category_name}分类===================')
        print(f'==================================================')

        # 筛选：仅保留2024年及之后的榜单
        filtered_ranks = {}
        for rank_name, (bang_type, year, period) in all_rank_data.items():
            # 1. 年度榜单：仅保留 >= start_year（2024）
            if bang_type == "year" and year >= start_year:
                filtered_ranks[rank_name] = (bang_type, year, period)
            # 2. 月度榜单：仅保留 >= start_year（2024），且不超过当前年月
            elif bang_type == "month" and year >= start_year:
                # 构造榜单的年月日期（取当月1号）
                rank_date = datetime(year, period, 1)
                # 仅保留 <= 当前年月的榜单（避免爬取未发布的未来月份）
                if rank_date <= current_date:
                    filtered_ranks[rank_name] = (bang_type, year, period)

        # 遍历筛选后的榜单（2024年至今）
        for rank_name, (bang_type, year, period) in filtered_ranks.items():
            progress_key = f"{category_name}-{rank_name}"

            # 跳过已完成的
            if progress_key in crawl_progress:
                print(f'---------------------{rank_name}已爬取完成，跳过---------------------')
                continue

            print(f'\n---------------------开始爬{rank_name}数据---------------------')
            current_rank_data = []

            # 每个榜单爬取25页
            for page in range(25):
                print(f'*****************开始爬取第{page + 1}页数据*****************')
                url = build_url(bang_type, year, period, page + 1, category_code)

                try:
                    time.sleep(random.uniform(1.5, 3.5))  # 随机延时
                    page_text = spider_data(url)
                    page_df = parse_data(page_text, rank_name, category_name)

                    if not page_df.empty:
                        current_rank_data.append(page_df)
                        print(f'*********第{page + 1}页爬取成功，新增{len(page_df)}条数据**********')
                    else:
                        print(f'*********第{page + 1}页无有效数据**********')

                except Exception as e:
                    print(f'!!!!!!!!!第{page + 1}页爬取失败: {e}!!!!!!!!!')
                    continue

            # 合并当前榜单数据
            if current_rank_data:
                rank_df = pd.concat(current_rank_data, ignore_index=True)
                all_data_df = pd.concat([all_data_df, rank_df], ignore_index=True)
                print(f'------------------{rank_name}爬取完成，累计新增{len(rank_df)}条数据------------------')
            else:
                print(f'------------------{rank_name}未获取到有效数据------------------')

            # 标记进度并保存
            crawl_progress[progress_key] = "completed"
            save_progress(crawl_progress)

        print(f'==================================================')
        print(f'=================={category_name}分类爬取完成===================')
        print(f'==================================================')

    # 保存最终数据
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(OUTPUT_DIR, f'dangdang_bestsellers_2024_onward_{timestamp}.csv')
    all_data_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    # 爬取完成后删除进度文件
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("爬取全部完成，已删除进度文件")

    print(f'\n所有数据爬取完成！')
    print(f'爬取范围：2024年至今（年度榜单 + 已发布的月度榜单）')
    print(f'合并后总数据量：{len(all_data_df)}条')
    print(f'数据已保存至: {output_path}')


if __name__ == "__main__":
    main()