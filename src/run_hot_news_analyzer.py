#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行热点新闻资金流共振分析器
"""

import pandas as pd
import argparse
from pathlib import Path
from loguru import logger
import sys
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.analyzers.hot_news_fund_flow_analyzer import HotNewsFundFlowAnalyzer

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='热点新闻资金流共振分析器')
    parser.add_argument('--news-days', type=int, default=1, help='获取最近几天的新闻，默认1天')
    parser.add_argument('--fund-flow-days', type=str, default='1,3,5', help='获取的资金流向天数列表，默认"1,3,5"，用逗号分隔')
    parser.add_argument('--top-n', type=int, default=20, help='返回前N个结果，默认20')
    parser.add_argument('--output', type=str, default='hot_news_stocks.csv', help='输出文件名，默认hot_news_stocks.csv')
    
    args = parser.parse_args()
    
    # 解析资金流向天数列表
    fund_flow_days_list = [int(days) for days in args.fund_flow_days.split(',')]
    
    # 配置日志 - 只使用控制台日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info(f"开始运行热点新闻资金流共振分析器，新闻天数: {args.news_days}，资金流向天数: {fund_flow_days_list}")
    
    # 创建并运行分析器
    analyzer = HotNewsFundFlowAnalyzer()
    result = analyzer.analyze(news_days=args.news_days, fund_flow_days_list=fund_flow_days_list, top_n=args.top_n)
    
    if result.empty:
        logger.warning("未找到任何与新闻热点匹配且资金流入的股票")
        return
    
    # 保存结果
    output_path = project_root / "output"
    output_path.mkdir(exist_ok=True)
    result.to_csv(output_path / args.output, index=False, encoding='utf-8-sig')
    
    # 打印结果
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("\n热点新闻资金流共振股票:")
    print(result)
    
    logger.info(f"分析完成，共找到 {len(result)} 只热点股票，结果已保存至 {output_path / args.output}")

if __name__ == "__main__":
    main() 