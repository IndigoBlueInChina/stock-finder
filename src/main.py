#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主程序入口
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 加载环境变量
load_dotenv()

# 配置loguru
logger.remove()  # 移除默认的sink
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

from src.agents.agent_manager import AgentManager
from src.data_fetchers.data_manager import DataManager
from src.analyzers.analyzer_manager import AnalyzerManager

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="股票潜力分析系统")
    parser.add_argument("--max-stocks", type=int, default=20,
                        help="最大处理的股票数量，默认为20")
    parser.add_argument("--top-n", type=int, default=10,
                        help="推荐的股票数量，默认为10")
    parser.add_argument("--min-score", type=float, default=60,
                        help="最低推荐评分，默认为60")
    parser.add_argument("--no-reason", action="store_true",
                        help="不生成推荐理由，加快分析速度")
    return parser.parse_args()

def main():
    """主程序入口函数"""
    # 解析命令行参数
    args = parse_args()
    
    logger.info("启动股票潜力分析系统...")
    logger.info(f"参数: 最大处理股票数量={args.max_stocks}, 推荐数量={args.top_n}, 最低评分={args.min_score}")
    
    try:
        # 初始化数据管理器
        data_manager = DataManager()
        
        # 初始化分析管理器
        analyzer_manager = AnalyzerManager()
        
        # 初始化Agent管理器
        agent_manager = AgentManager(data_manager, analyzer_manager)
        
        # 运行分析流程
        results = agent_manager.run_analysis_pipeline(
            top_n=args.top_n,
            min_score=args.min_score,
            max_stocks_to_process=args.max_stocks
        )
        
        # 输出结果
        if results:
            logger.info(f"分析完成，找到 {len(results)} 只潜力股票:")
            for i, stock in enumerate(results):
                print(f"\n股票 {i+1}/{len(results)}")
                print(f"代码: {stock['code']}")
                print(f"名称: {stock['name']}")
                print(f"潜力评分: {stock['potential_score']:.2f}")
                print(f"资金流向评分: {stock['fund_flow_score']:.2f}")
                print(f"社交热度评分: {stock['social_score']:.2f}")
                print(f"基本面评分: {stock['fundamental_score']:.2f}")
                print(f"技术面评分: {stock['technical_score']:.2f}")
                print(f"行业评分: {stock['industry_score']:.2f}")
                
                if 'reason' in stock:
                    print(f"推荐理由: {stock['reason']}")
                print("-" * 50)
        else:
            logger.warning("未找到符合条件的潜力股票")
    
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    # 运行主程序
    sys.exit(main()) 