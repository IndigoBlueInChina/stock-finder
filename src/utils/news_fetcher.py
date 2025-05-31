#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
新闻数据获取工具
"""

import pandas as pd
from loguru import logger
from pathlib import Path

def get_news_data(days=1, cache_dir=None):
    """
    获取热点新闻数据
    
    Args:
        days: 获取最近几天的新闻，默认1天
        cache_dir: 数据缓存目录，默认为None，使用默认缓存目录
        
    Returns:
        pandas.DataFrame: 包含新闻标题、内容、时间等信息的DataFrame
    """
    try:
        # 动态导入以避免循环导入问题
        from src.analyzers.hot_news_fund_flow_analyzer import HotNewsFundFlowAnalyzer
        
        # 创建分析器实例
        analyzer = HotNewsFundFlowAnalyzer(cache_dir=cache_dir)
        
        # 获取热点新闻
        news_df = analyzer.get_hot_news(days=days)
        
        return news_df
    
    except Exception as e:
        logger.error(f"获取新闻数据失败: {e}")
        # 返回空DataFrame
        return pd.DataFrame() 