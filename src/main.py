#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主程序入口
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{project_root}/logs/app.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)

from src.agents.agent_manager import AgentManager
from src.data_fetchers.data_manager import DataManager
from src.analyzers.analyzer_manager import AnalyzerManager

def main():
    """主程序入口函数"""
    logger.info("启动股票潜力分析系统...")
    
    try:
        # 初始化数据管理器
        data_manager = DataManager()
        
        # 初始化分析管理器
        analyzer_manager = AnalyzerManager()
        
        # 初始化Agent管理器
        agent_manager = AgentManager(data_manager, analyzer_manager)
        
        # 运行分析流程
        results = agent_manager.run_analysis_pipeline()
        
        # 输出结果
        logger.info("分析完成，潜力股票列表:")
        for stock in results:
            print(f"股票代码: {stock['code']}, 名称: {stock['name']}, 潜力评分: {stock['potential_score']}")
            print(f"推荐理由: {stock['reason']}")
            print("-" * 50)
    
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    # 创建logs目录
    Path(f"{project_root}/logs").mkdir(exist_ok=True)
    
    # 运行主程序
    sys.exit(main()) 