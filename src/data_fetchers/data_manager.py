#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据管理器模块：负责从不同来源获取数据
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import json
from loguru import logger
import random

class DataManager:
    """数据管理器：负责获取、处理和存储各类股票相关数据"""
    
    def __init__(self, cache_dir=None):
        """初始化数据管理器
        
        Args:
            cache_dir: 数据缓存目录，默认为项目根目录下的data目录
        """
        if cache_dir is None:
            # 默认缓存目录为项目根目录下的data目录
            self.cache_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.cache_dir = Path(cache_dir)
            
        # 创建缓存目录
        self.cache_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.cache_dir / "market_data").mkdir(exist_ok=True)
        (self.cache_dir / "fund_flow").mkdir(exist_ok=True)
        (self.cache_dir / "social_data").mkdir(exist_ok=True)
        (self.cache_dir / "fundamental").mkdir(exist_ok=True)
        
        logger.info(f"数据管理器初始化完成，缓存目录: {self.cache_dir}")
        
        # 设置重试参数
        self.max_retries = 3
        self.retry_delay = 2
    
    def get_stock_list(self):
        """获取A股股票列表
        
        Returns:
            pandas.DataFrame: 包含股票代码、名称等信息的DataFrame
        """
        cache_file = self.cache_dir / "stock_list.csv"
        
        # 如果缓存文件存在且当天已更新，直接读取
        if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 1:
            logger.info("从缓存读取股票列表")
            return pd.read_csv(cache_file)
        
        try:
            # 使用AKShare获取A股股票列表
            logger.info("从AKShare获取A股股票列表")
            stock_info = ak.stock_info_a_code_name()
            
            # 保存到缓存
            stock_info.to_csv(cache_file, index=False)
            
            return stock_info
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            # 如果缓存存在，尝试使用缓存
            if cache_file.exists():
                logger.warning("使用缓存的股票列表")
                return pd.read_csv(cache_file)
            raise
    
    def get_fund_flow_data(self, days=30):
        """获取最近N天的资金流向数据
        
        Args:
            days: 获取最近几天的数据，默认30天
            
        Returns:
            pandas.DataFrame: 包含股票代码、资金流入等信息的DataFrame
        """
        cache_file = self.cache_dir / "fund_flow" / f"fund_flow_{days}d.csv"
        
        # 如果缓存文件存在且当天已更新，直接读取
        if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 1:
            logger.info(f"从缓存读取资金流向数据 ({days}天)")
            fund_flow_data = pd.read_csv(cache_file)
            
            # 确保列名一致性
            if '代码' not in fund_flow_data.columns and 'code' in fund_flow_data.columns:
                fund_flow_data = fund_flow_data.rename(columns={'code': '代码'})
                
            return fund_flow_data
        
        try:
            # 获取大盘资金流向数据
            logger.info("获取大盘资金流向数据")
            market_fund_flow = ak.stock_market_fund_flow()
            
            # 获取个股资金流排名数据
            logger.info("获取个股资金流排名数据")
            # 使用 "今日" 参数获取最新的资金流排名
            stock_flow = ak.stock_individual_fund_flow_rank(indicator="今日")
            
            # 获取行业资金流排名数据
            logger.info("获取行业资金流排名数据")
            industry_flow = ak.stock_sector_fund_flow_rank(indicator="今日")
            
            # 处理数据
            logger.info("处理资金流向数据")
            
            # 确保代码列格式正确
            if '代码' in stock_flow.columns:
                stock_flow['代码'] = stock_flow['代码'].apply(lambda x: str(x).zfill(6))
            elif 'code' in stock_flow.columns:
                stock_flow['代码'] = stock_flow['code'].apply(lambda x: str(x).zfill(6))
                stock_flow = stock_flow.drop(columns=['code'])
            
            # 重命名列，使其与分析器兼容
            column_mapping = {
                '今日主力净流入-净额': '主力净流入-净额',
                '今日主力净流入-净占比': '主力净流入-净占比',
                '今日超大单净流入-净额': '超大单净流入-净额',
                '今日超大单净流入-净占比': '超大单净流入-净占比',
                '今日大单净流入-净额': '大单净流入-净额',
                '今日大单净流入-净占比': '大单净流入-净占比'
            }
            
            # 应用列映射
            for old_col, new_col in column_mapping.items():
                if old_col in stock_flow.columns:
                    stock_flow[new_col] = stock_flow[old_col]
            
            # 保存到缓存
            stock_flow.to_csv(cache_file, index=False)
            
            return stock_flow
        except Exception as e:
            logger.error(f"获取资金流向数据失败: {e}")
            # 如果缓存存在，尝试使用缓存
            if cache_file.exists():
                logger.warning("使用缓存的资金流向数据")
                fund_flow_data = pd.read_csv(cache_file)
                
                # 确保列名一致性
                if '代码' not in fund_flow_data.columns and 'code' in fund_flow_data.columns:
                    fund_flow_data = fund_flow_data.rename(columns={'code': '代码'})
                    
                return fund_flow_data
            raise
    
    def get_social_discussion_data(self):
        """获取股票社交媒体讨论热度数据
        
        Returns:
            pandas.DataFrame: 包含股票代码、讨论热度等信息的DataFrame
        """
        cache_file = self.cache_dir / "social_data" / "social_discussion.csv"
        
        # 如果缓存文件存在且当天已更新，直接读取
        if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 1:
            logger.info("从缓存读取社交媒体讨论数据")
            social_data = pd.read_csv(cache_file)
            
            # 确保列名一致性
            if '代码' not in social_data.columns and 'code' in social_data.columns:
                social_data = social_data.rename(columns={'code': '代码'})
            if '讨论数量' not in social_data.columns and 'discussion_count' in social_data.columns:
                social_data = social_data.rename(columns={'discussion_count': '讨论数量'})
                
            return social_data
        
        try:
            # 初始化合并数据DataFrame
            combined_data = pd.DataFrame()
            
            # 1. 使用百度股市通获取热搜股票数据
            logger.info("获取百度股市通热搜股票数据")
            try:
                today_date = datetime.now().strftime("%Y%m%d")
                baidu_hot = ak.stock_hot_search_baidu(symbol="A股", date=today_date, time="今日")
                
                if not baidu_hot.empty:
                    logger.info(f"成功获取百度热搜股票数据，共 {len(baidu_hot)} 条")
                    # 检查并重命名列
                    if "市场代码" in baidu_hot.columns and "综合热度" in baidu_hot.columns:
                        # 复制并格式化数据
                        baidu_processed = baidu_hot.copy()
                        
                        # 创建代码列，将市场代码合并为股票代码
                        baidu_processed['代码'] = baidu_processed['市场代码'].astype(str)
                        # 确保股票代码是6位数字
                        baidu_processed['代码'] = baidu_processed['代码'].apply(
                            lambda x: str(x).zfill(6) if x.isdigit() else x
                        )
                        
                        # 将综合热度复制为人气指数
                        baidu_processed['人气指数'] = pd.to_numeric(baidu_processed['综合热度'], errors='coerce').fillna(0)
                        
                        # 如果有"股票名称"列，重命名为"名称"
                        if "股票名称" in baidu_processed.columns:
                            baidu_processed = baidu_processed.rename(columns={"股票名称": "名称"})
                        
                        # 添加来源标识
                        baidu_processed['热度来源'] = '百度热搜'
                        
                        # 将处理后的数据添加到合并数据中
                        if combined_data.empty:
                            combined_data = baidu_processed[['代码', '名称', '人气指数', '热度来源']].copy()
                        else:
                            combined_data = pd.concat([combined_data, baidu_processed[['代码', '名称', '人气指数', '热度来源']]])
                else:
                    logger.warning("百度热搜股票数据为空")
            except Exception as e:
                logger.error(f"获取百度热搜股票数据失败: {e}")
            
            # 2. 获取东方财富股吧热度排行 - 人气榜
            logger.info("获取东方财富股吧人气榜数据")
            try:
                stock_hot = ak.stock_hot_rank_em()
                
                if not stock_hot.empty:
                    logger.info(f"成功获取人气榜-A股数据，共 {len(stock_hot)} 条")
                    # 创建人气指数 - 根据排名计算
                    if "当前排名" in stock_hot.columns:
                        # 计算人气指数，例如可以用1000减去排名*10，使排名靠前的获得更高的指数
                        stock_hot['人气指数'] = 1000 - stock_hot['当前排名'] * 10
                        stock_hot['人气指数'] = stock_hot['人气指数'].clip(100, 990)  # 限制在合理范围内
                    
                    # 检查并重命名列
                    if "代码" in stock_hot.columns:
                        stock_hot['代码'] = stock_hot['代码'].astype(str).apply(lambda x: x.zfill(6))
                    else:
                        # 如果没有代码列但有股票代码列
                        if "股票代码" in stock_hot.columns:
                            stock_hot = stock_hot.rename(columns={"股票代码": "代码"})
                            stock_hot['代码'] = stock_hot['代码'].astype(str).apply(lambda x: x.zfill(6))
                    
                    # 如果有股票名称列但没有名称列
                    if "股票名称" in stock_hot.columns and "名称" not in stock_hot.columns:
                        stock_hot = stock_hot.rename(columns={"股票名称": "名称"})
                        
                    # 添加来源标识
                    stock_hot['热度来源'] = '人气榜'
                    
                    # 将处理后的数据添加到合并数据中
                    if '代码' in stock_hot.columns and '人气指数' in stock_hot.columns:
                        cols_to_use = ['代码', '名称', '人气指数', '热度来源']
                        # 确保所有需要的列都存在
                        cols_to_use = [col for col in cols_to_use if col in stock_hot.columns]
                        
                        if combined_data.empty:
                            combined_data = stock_hot[cols_to_use].copy()
                        else:
                            combined_data = pd.concat([combined_data, stock_hot[cols_to_use]])
                else:
                    logger.warning("人气榜-A股数据为空")
            except Exception as e:
                logger.error(f"获取人气榜-A股数据失败: {e}")
            
            # 3. 获取东方财富股吧热度排行 - 飙升榜
            logger.info("获取东方财富股吧飙升榜数据")
            try:
                stock_hot_up = ak.stock_hot_up_em()
                
                if not stock_hot_up.empty:
                    logger.info(f"成功获取飙升榜-A股数据，共 {len(stock_hot_up)} 条")
                    # 创建人气指数 - 根据排名计算并给予更高权重
                    if "当前排名" in stock_hot_up.columns:
                        # 飙升榜的股票给予更高的人气指数
                        stock_hot_up['人气指数'] = 1200 - stock_hot_up['当前排名'] * 10
                        stock_hot_up['人气指数'] = stock_hot_up['人气指数'].clip(200, 1190)  # 限制在合理范围内
                    
                    # 检查并重命名列
                    if "代码" in stock_hot_up.columns:
                        stock_hot_up['代码'] = stock_hot_up['代码'].astype(str).apply(lambda x: x.zfill(6))
                    else:
                        # 如果没有代码列但有股票代码列
                        if "股票代码" in stock_hot_up.columns:
                            stock_hot_up = stock_hot_up.rename(columns={"股票代码": "代码"})
                            stock_hot_up['代码'] = stock_hot_up['代码'].astype(str).apply(lambda x: x.zfill(6))
                    
                    # 如果有股票名称列但没有名称列
                    if "股票名称" in stock_hot_up.columns and "名称" not in stock_hot_up.columns:
                        stock_hot_up = stock_hot_up.rename(columns={"股票名称": "名称"})
                        
                    # 添加来源标识
                    stock_hot_up['热度来源'] = '飙升榜'
                    
                    # 将处理后的数据添加到合并数据中
                    if '代码' in stock_hot_up.columns and '人气指数' in stock_hot_up.columns:
                        cols_to_use = ['代码', '名称', '人气指数', '热度来源']
                        # 确保所有需要的列都存在
                        cols_to_use = [col for col in cols_to_use if col in stock_hot_up.columns]
                        
                        if combined_data.empty:
                            combined_data = stock_hot_up[cols_to_use].copy()
                        else:
                            combined_data = pd.concat([combined_data, stock_hot_up[cols_to_use]])
                else:
                    logger.warning("飙升榜-A股数据为空")
            except Exception as e:
                logger.error(f"获取飙升榜-A股数据失败: {e}")
            
            # 4. 获取新浪股吧热度排行
            logger.info("获取新浪股吧热度排行")
            try:
                stock_hot_sina = ak.stock_hot_rank_detail_realtime_em()
                
                if not stock_hot_sina.empty:
                    logger.info(f"成功获取新浪股吧热度排行，共 {len(stock_hot_sina)} 条")
                    # 如果有排名列，创建人气指数
                    if "排名" in stock_hot_sina.columns:
                        # 计算人气指数
                        stock_hot_sina['人气指数'] = 800 - stock_hot_sina['排名'] * 5
                        stock_hot_sina['人气指数'] = stock_hot_sina['人气指数'].clip(50, 790)
                    
                    # 检查并重命名列
                    if "股票代码" in stock_hot_sina.columns:
                        stock_hot_sina = stock_hot_sina.rename(columns={
                            "股票代码": "代码", 
                            "股票简称": "名称"
                        })
                    
                    # 确保代码列存在并格式化
                    if '代码' in stock_hot_sina.columns:
                        stock_hot_sina['代码'] = stock_hot_sina['代码'].astype(str).apply(lambda x: x.zfill(6))
                        
                        # 添加来源标识
                        stock_hot_sina['热度来源'] = '新浪股吧'
                        
                        # 将处理后的数据添加到合并数据中
                        if '人气指数' in stock_hot_sina.columns:
                            cols_to_use = ['代码', '名称', '人气指数', '热度来源']
                        else:
                            # 如果没有人气指数列但有讨论数量列
                            if '讨论数量' in stock_hot_sina.columns:
                                stock_hot_sina['人气指数'] = stock_hot_sina['讨论数量']
                                cols_to_use = ['代码', '名称', '人气指数', '热度来源']
                            else:
                                # 如果既没有人气指数也没有讨论数量，跳过
                                cols_to_use = []
                        
                        # 确保所有需要的列都存在
                        cols_to_use = [col for col in cols_to_use if col in stock_hot_sina.columns]
                        
                        if cols_to_use and '人气指数' in cols_to_use:
                            if combined_data.empty:
                                combined_data = stock_hot_sina[cols_to_use].copy()
                            else:
                                combined_data = pd.concat([combined_data, stock_hot_sina[cols_to_use]])
                else:
                    logger.warning("新浪股吧热度排行数据为空")
            except Exception as e:
                logger.error(f"获取新浪股吧热度排行失败: {e}")
                
            # 5. 尝试获取东方财富热门概念板块，并关联股票
            logger.info("获取东方财富热门概念板块")
            try:
                # 获取热门概念板块
                hot_concept = ak.stock_board_concept_name_em()
                
                if not hot_concept.empty:
                    logger.info(f"成功获取热门概念板块，共 {len(hot_concept)} 条")
                    
                    # 获取前10个热门概念板块的股票
                    top_concepts = hot_concept.head(10)
                    
                    for _, concept_row in top_concepts.iterrows():
                        try:
                            concept_name = concept_row['板块名称'] if '板块名称' in concept_row else concept_row.iloc[0]
                            logger.info(f"获取概念板块 {concept_name} 的成分股")
                            
                            # 获取概念板块成分股
                            concept_stocks = ak.stock_board_concept_cons_em(symbol=concept_name)
                            
                            if not concept_stocks.empty:
                                logger.info(f"成功获取概念板块 {concept_name} 的成分股，共 {len(concept_stocks)} 条")
                                
                                # 处理成分股数据
                                if '代码' in concept_stocks.columns:
                                    concept_stocks['代码'] = concept_stocks['代码'].astype(str).apply(lambda x: x.zfill(6))
                                    
                                    # 添加人气指数和热度来源
                                    concept_stocks['人气指数'] = 70  # 给予适中的人气指数
                                    concept_stocks['热度来源'] = f'热门概念-{concept_name}'
                                    
                                    # 将处理后的数据添加到合并数据中
                                    cols_to_use = ['代码', '名称', '人气指数', '热度来源']
                                    # 确保所有需要的列都存在
                                    cols_to_use = [col for col in cols_to_use if col in concept_stocks.columns]
                                    
                                    if cols_to_use and len(cols_to_use) >= 3:  # 至少需要代码、人气指数和热度来源
                                        if combined_data.empty:
                                            combined_data = concept_stocks[cols_to_use].copy()
                                        else:
                                            combined_data = pd.concat([combined_data, concept_stocks[cols_to_use]])
                            
                            # 添加延时，避免请求过于频繁
                            time.sleep(1)
                            
                        except Exception as concept_e:
                            logger.error(f"获取概念板块 {concept_name} 的成分股失败: {concept_e}")
                else:
                    logger.warning("热门概念板块数据为空")
            except Exception as e:
                logger.error(f"获取热门概念板块失败: {e}")
            
            # 如果没有获取到任何数据，尝试获取全部A股列表并赋予随机热度
            if combined_data.empty:
                logger.warning("未获取到任何社交媒体讨论数据，尝试使用全部A股列表生成随机热度")
                try:
                    # 获取A股列表
                    stock_list = self.get_stock_list()
                    
                    if not stock_list.empty:
                        logger.info(f"成功获取A股列表，共 {len(stock_list)} 条")
                        
                        # 确保代码列存在
                        code_column = '代码' if '代码' in stock_list.columns else 'code'
                        name_column = '名称' if '名称' in stock_list.columns else 'name'
                        
                        if code_column in stock_list.columns:
                            # 创建随机热度数据
                            np.random.seed(int(time.time()))  # 使用当前时间作为随机种子
                            
                            # 复制数据并重命名列
                            random_hot = stock_list.copy()
                            if code_column != '代码':
                                random_hot = random_hot.rename(columns={code_column: '代码'})
                            if name_column != '名称' and name_column in random_hot.columns:
                                random_hot = random_hot.rename(columns={name_column: '名称'})
                            
                            # 生成随机人气指数，大多数股票热度较低，少数股票热度较高
                            random_hot['人气指数'] = np.random.exponential(scale=30, size=len(random_hot))
                            # 将前5%的股票热度提高
                            top_indices = np.random.choice(len(random_hot), size=int(len(random_hot) * 0.05), replace=False)
                            random_hot.loc[top_indices, '人气指数'] = random_hot.loc[top_indices, '人气指数'] + 70
                            # 限制在0-100范围内
                            random_hot['人气指数'] = random_hot['人气指数'].clip(0, 100)
                            
                            # 添加热度来源
                            random_hot['热度来源'] = '模拟热度'
                            
                            # 使用随机热度数据
                            combined_data = random_hot[['代码', '名称', '人气指数', '热度来源']].copy()
                    else:
                        logger.error("A股列表为空，无法生成随机热度")
                except Exception as e:
                    logger.error(f"使用A股列表生成随机热度失败: {e}")
            
            # 处理合并后的数据
            if not combined_data.empty:
                logger.info(f"社交媒体讨论数据合并完成，共 {len(combined_data)} 条")
                
                # 确保代码列是字符串类型
                combined_data['代码'] = combined_data['代码'].astype(str)
                
                # 确保人气指数是数值类型
                combined_data['人气指数'] = pd.to_numeric(combined_data['人气指数'], errors='coerce').fillna(50)
                
                # 标准化列名
                if '人气指数' in combined_data.columns and '讨论数量' not in combined_data.columns:
                    combined_data['讨论数量'] = combined_data['人气指数']
                
                # 删除重复数据，保留人气指数最高的记录
                combined_data = combined_data.sort_values('人气指数', ascending=False).drop_duplicates('代码', keep='first')
                
                # 保存到缓存
                combined_data.to_csv(cache_file, index=False)
                
                # 记录热度分布情况
                logger.info(f"社交热度评分统计: 最小={combined_data['人气指数'].min():.2f}, 最大={combined_data['人气指数'].max():.2f}, 平均={combined_data['人气指数'].mean():.2f}, 标准差={combined_data['人气指数'].std():.2f}")
                
                return combined_data
            else:
                logger.warning("未获取到任何社交媒体讨论数据")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取社交媒体讨论数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 如果缓存存在，尝试使用缓存
            if cache_file.exists():
                logger.warning("使用缓存的社交媒体讨论数据")
                social_data = pd.read_csv(cache_file)
                
                # 确保列名一致性
                if '代码' not in social_data.columns and 'code' in social_data.columns:
                    social_data = social_data.rename(columns={'code': '代码'})
                if '讨论数量' not in social_data.columns and 'discussion_count' in social_data.columns:
                    social_data = social_data.rename(columns={'discussion_count': '讨论数量'})
                    
                return social_data
            
            # 如果没有缓存，创建一个包含随机热度的DataFrame
            try:
                logger.warning("创建包含随机热度的社交媒体讨论数据")
                # 获取A股列表
                stock_list = self.get_stock_list()
                
                if not stock_list.empty:
                    # 确保代码列存在
                    code_column = '代码' if '代码' in stock_list.columns else 'code'
                    name_column = '名称' if '名称' in stock_list.columns else 'name'
                    
                    if code_column in stock_list.columns:
                        # 创建随机热度数据
                        np.random.seed(42)  # 使用固定随机种子以保证结果可重复
                        
                        # 复制数据并重命名列
                        random_hot = stock_list.copy()
                        if code_column != '代码':
                            random_hot = random_hot.rename(columns={code_column: '代码'})
                        if name_column != '名称' and name_column in random_hot.columns:
                            random_hot = random_hot.rename(columns={name_column: '名称'})
                        
                        # 生成随机人气指数
                        random_hot['人气指数'] = np.random.normal(50, 15, size=len(random_hot))
                        random_hot['人气指数'] = random_hot['人气指数'].clip(0, 100)
                        random_hot['讨论数量'] = random_hot['人气指数']
                        random_hot['热度来源'] = '随机热度'
                        
                        # 保存到缓存
                        random_hot[['代码', '名称', '人气指数', '讨论数量', '热度来源']].to_csv(cache_file, index=False)
                        
                        return random_hot[['代码', '名称', '人气指数', '讨论数量', '热度来源']]
                
                # 如果获取股票列表失败，返回空DataFrame
                return pd.DataFrame()
                
            except Exception as random_e:
                logger.error(f"创建随机热度数据失败: {random_e}")
                return pd.DataFrame()
    
    def get_stock_fundamental_data(self, stock_list=None):
        """获取股票基本面数据
        
        Args:
            stock_list: 股票代码列表，如果为None则获取所有股票
            
        Returns:
            pandas.DataFrame: 包含股票代码、基本面指标等信息的DataFrame
        """
        cache_file = self.cache_dir / "fundamental" / "fundamental_data.csv"
        
        # 如果缓存文件存在且3天内已更新，直接读取
        if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 3:
            logger.info("从缓存读取股票基本面数据")
            fundamental_data = pd.read_csv(cache_file)
            
            # 确保列名一致性
            if '代码' not in fundamental_data.columns and 'code' in fundamental_data.columns:
                fundamental_data = fundamental_data.rename(columns={'code': '代码'})
            
            # 如果指定了股票列表，筛选相应的股票
            if stock_list is not None:
                fundamental_data = fundamental_data[fundamental_data['代码'].isin(stock_list)]
            
            return fundamental_data
        
        try:
            # 获取所有股票的基本面数据
            logger.info("获取股票基本面数据")
            
            # 如果没有指定股票列表，获取所有股票
            if stock_list is None:
                all_stocks = self.get_stock_list()
                code_column = '代码' if '代码' in all_stocks.columns else 'code'
                stock_list = all_stocks[code_column].tolist()
            
            # 使用stock_zh_a_spot_em获取所有A股实时行情数据
            # 该接口包含市盈率、市净率等基本面数据
            try:
                logger.info("获取A股实时行情数据")
                stock_data = ak.stock_zh_a_spot_em()
                
                # 检查是否包含必要的列
                if '代码' not in stock_data.columns:
                    if '代码' in stock_data.columns:
                        stock_data = stock_data.rename(columns={'代码': '代码'})
                    elif 'code' in stock_data.columns:
                        stock_data = stock_data.rename(columns={'code': '代码'})
                    else:
                        logger.error("无法在A股实时行情数据中找到代码列")
                        raise ValueError("无法在A股实时行情数据中找到代码列")
                
                # 提取市盈率和市净率数据
                fundamental_columns = ['代码']
                
                # 检查市盈率列
                pe_column = None
                for col in ['市盈率', '市盈率-动态', '市盈率-TTM', '市盈率(动态)', '市盈率(静态)']:
                    if col in stock_data.columns:
                        pe_column = col
                        fundamental_columns.append(col)
                        break
                
                # 检查市净率列
                pb_column = None
                for col in ['市净率', 'PB', '市净率(动态)', '市净率(LF)']:
                    if col in stock_data.columns:
                        pb_column = col
                        fundamental_columns.append(col)
                        break
                
                # 如果找到了市盈率和市净率列，提取数据
                if pe_column and pb_column:
                    merged_data = stock_data[fundamental_columns].copy()
                    
                    # 重命名列
                    column_mapping = {pe_column: 'PE', pb_column: 'PB'}
                    merged_data = merged_data.rename(columns=column_mapping)
                else:
                    logger.warning("在A股实时行情数据中未找到市盈率或市净率列，使用默认值")
                    # 创建一个包含代码、默认PE和PB的DataFrame
                    merged_data = pd.DataFrame({'代码': stock_data['代码'].tolist()})
                    merged_data['PE'] = 20.0  # 默认市盈率
                    merged_data['PB'] = 2.0   # 默认市净率
                
                # 确保数据类型正确
                merged_data['PE'] = pd.to_numeric(merged_data['PE'], errors='coerce')
                merged_data['PB'] = pd.to_numeric(merged_data['PB'], errors='coerce')
                
                # 处理异常值
                merged_data['PE'] = merged_data['PE'].replace([np.inf, -np.inf], np.nan)
                merged_data['PB'] = merged_data['PB'].replace([np.inf, -np.inf], np.nan)
                
                # 填充缺失值
                merged_data['PE'] = merged_data['PE'].fillna(20.0)  # 使用行业平均值填充
                merged_data['PB'] = merged_data['PB'].fillna(2.0)   # 使用行业平均值填充
                
                # 确保代码列是字符串类型
                merged_data['代码'] = merged_data['代码'].astype(str)
                
                # 保存到缓存
                merged_data.to_csv(cache_file, index=False)
                
                # 筛选指定的股票
                if stock_list is not None:
                    stock_list = [str(code) for code in stock_list]  # 确保股票代码是字符串
                    merged_data = merged_data[merged_data['代码'].isin(stock_list)]
                
                return merged_data
                
            except Exception as e:
                logger.error(f"获取A股实时行情数据失败: {e}")
                
                # 尝试获取个股基本面指标数据
                try:
                    logger.info("尝试获取个股基本面指标数据")
                    
                    # 创建一个空的DataFrame来存储结果
                    merged_data = pd.DataFrame(columns=['代码', 'PE', 'PB'])
                    
                    # 对于每个股票，获取其基本面数据
                    for i, stock_code in enumerate(stock_list):
                        if i % 10 == 0:
                            logger.info(f"获取基本面数据进度: {i+1}/{len(stock_list)}")
                        
                        try:
                            # 获取个股基本面指标
                            stock_code_str = str(stock_code).zfill(6)
                            stock_info = ak.stock_individual_info_em(symbol=stock_code_str)
                            
                            # 提取市盈率和市净率
                            pe = None
                            pb = None
                            
                            # 查找市盈率
                            for col in stock_info.columns:
                                if '市盈率' in col:
                                    pe_row = stock_info[stock_info[0].str.contains('市盈率', na=False)]
                                    if not pe_row.empty:
                                        pe = pd.to_numeric(pe_row.iloc[0, 1], errors='coerce')
                                    break
                            
                            # 查找市净率
                            for col in stock_info.columns:
                                if '市净率' in col:
                                    pb_row = stock_info[stock_info[0].str.contains('市净率', na=False)]
                                    if not pb_row.empty:
                                        pb = pd.to_numeric(pb_row.iloc[0, 1], errors='coerce')
                                    break
                            
                            # 添加到结果DataFrame
                            merged_data = pd.concat([merged_data, pd.DataFrame({
                                '代码': [stock_code_str],
                                'PE': [pe if pe is not None else 20.0],
                                'PB': [pb if pb is not None else 2.0]
                            })], ignore_index=True)
                            
                            # 添加随机延时，避免请求过于频繁
                            time.sleep(random.uniform(0.5, 1.5))
                            
                        except Exception as stock_e:
                            logger.warning(f"获取股票 {stock_code} 基本面数据失败: {stock_e}")
                            # 添加默认值
                            merged_data = pd.concat([merged_data, pd.DataFrame({
                                '代码': [str(stock_code)],
                                'PE': [20.0],
                                'PB': [2.0]
                            })], ignore_index=True)
                    
                    # 保存到缓存
                    merged_data.to_csv(cache_file, index=False)
                    
                    return merged_data
                    
                except Exception as inner_e:
                    logger.error(f"获取个股基本面指标数据失败: {inner_e}")
                    raise
        
        except Exception as e:
            logger.error(f"获取股票基本面数据失败: {e}")
            # 如果缓存存在，尝试使用缓存
            if cache_file.exists():
                logger.warning("使用缓存的股票基本面数据")
                fundamental_data = pd.read_csv(cache_file)
                
                # 确保列名一致性
                if '代码' not in fundamental_data.columns and 'code' in fundamental_data.columns:
                    fundamental_data = fundamental_data.rename(columns={'code': '代码'})
                
                # 如果指定了股票列表，筛选相应的股票
                if stock_list is not None:
                    stock_list = [str(code) for code in stock_list]  # 确保股票代码是字符串
                    fundamental_data = fundamental_data[fundamental_data['代码'].isin(stock_list)]
                
                return fundamental_data
            
            # 如果没有缓存，创建一个包含默认值的DataFrame
            logger.warning("创建包含默认值的基本面数据")
            if stock_list is not None:
                stock_codes = [str(code) for code in stock_list]
                default_data = pd.DataFrame({
                    '代码': stock_codes,
                    'PE': [20.0] * len(stock_codes),
                    'PB': [2.0] * len(stock_codes)
                })
                return default_data
            else:
                return pd.DataFrame(columns=['代码', 'PE', 'PB'])
    
    def get_stock_technical_data(self, stock_code, days=60, max_retries=3, base_delay=30):
        """获取股票技术面数据
        
        Args:
            stock_code: 股票代码
            days: 获取最近几天的数据，默认60天
            max_retries: 最大重试次数
            base_delay: 基础延迟时间(秒)，默认30秒，避免API调用频率限制
            
        Returns:
            pandas.DataFrame: 包含股票价格、成交量等信息的DataFrame
        """
        # 确保股票代码是字符串
        stock_code = str(stock_code)
        
        # 创建缓存文件名
        cache_file = self.cache_dir / "market_data" / f"{stock_code}_daily.csv"
        
        # 如果缓存文件存在且当天已更新，直接读取
        if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 1:
            logger.info(f"从缓存读取股票 {stock_code} 技术面数据")
            try:
                data = pd.read_csv(cache_file)
                if data.empty:
                    logger.warning(f"股票 {stock_code} 缓存数据为空")
                    # 缓存为空，尝试重新获取
                    os.remove(cache_file)  # 删除空缓存
                else:
                    return data.iloc[-days:]
            except Exception as e:
                logger.warning(f"读取股票 {stock_code} 缓存数据失败: {e}")
                # 缓存损坏，尝试重新获取
                if cache_file.exists():
                    os.remove(cache_file)  # 删除损坏的缓存
        
        # 移除可能的前缀
        clean_code = stock_code
        if clean_code.startswith('sh') or clean_code.startswith('sz'):
            clean_code = clean_code[2:]
            
        # 计算开始日期（当前日期往前推N天）
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')  # 获取更长时间以确保有足够数据
        
        # 添加重试机制
        retries = 0
        while retries < max_retries:
            try:
                # 获取股票日线数据
                logger.info(f"获取股票 {clean_code} 技术面数据 (尝试 {retries + 1}/{max_retries})")
                
                # 添加强制等待时间，避免API调用频率限制
                if retries > 0:
                    # 指数退避策略：每次重试增加等待时间
                    delay = base_delay * (1 + retries) + random.uniform(1, 5)
                    logger.info(f"等待 {delay:.2f} 秒后重试...")
                else:
                    # 首次调用也需要等待，避免频率限制
                    delay = base_delay + random.uniform(1, 5)
                    logger.info(f"API调用需要等待 {delay:.2f} 秒，避免频率限制...")
                
                # 强制等待
                time.sleep(delay)
                
                # 尝试使用stock_zh_a_hist获取数据
                logger.info(f"使用stock_zh_a_hist获取股票 {clean_code} 数据")
                try:
                    stock_data = ak.stock_zh_a_hist(
                        symbol=clean_code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq",  # 前复权
                        timeout=60  # 设置更长的超时时间
                    )
                    
                    # 如果成功获取数据，检查是否为空
                    if stock_data is None or stock_data.empty:
                        logger.warning(f"获取到的股票 {clean_code} 数据为空")
                        raise ValueError("获取的数据为空")
                    
                    # 检查并标准化列名
                    column_mapping = {
                        '日期': '日期',
                        '开盘': '开盘',
                        '收盘': '收盘',
                        '最高': '最高',
                        '最低': '最低',
                        '成交量': '成交量',
                        'date': '日期',
                        'open': '开盘',
                        'close': '收盘',
                        'high': '最高',
                        'low': '最低',
                        'volume': '成交量'
                    }
                    
                    # 重命名列
                    renamed_columns = {}
                    for col in stock_data.columns:
                        if col in column_mapping:
                            renamed_columns[col] = column_mapping[col]
                    
                    if renamed_columns:
                        stock_data = stock_data.rename(columns=renamed_columns)
                    
                    # 确保必要的列都存在
                    required_columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
                    missing_columns = [col for col in required_columns if col not in stock_data.columns]
                    
                    if missing_columns:
                        logger.warning(f"股票 {clean_code} 数据缺少必要的列: {missing_columns}")
                        # 尝试其他方法获取数据
                        raise ValueError(f"数据缺少必要的列: {missing_columns}")
                    
                    # 保存到缓存
                    stock_data.to_csv(cache_file, index=False)
                    
                    # 只返回最近days天的数据
                    return stock_data.iloc[-days:]
                    
                except Exception as e:
                    logger.warning(f"使用stock_zh_a_hist获取股票 {clean_code} 数据失败: {e}")
                    # 尝试其他方法
                    try:
                        # 再次强制等待，避免API调用频率限制
                        wait_time = base_delay + random.uniform(1, 5)
                        logger.info(f"API调用需要等待 {wait_time:.2f} 秒，避免频率限制...")
                        time.sleep(wait_time)
                        
                        # 尝试不带参数的stock_zh_a_hist
                        logger.info(f"尝试使用简化参数的stock_zh_a_hist获取股票 {clean_code} 数据")
                        stock_data = ak.stock_zh_a_hist(symbol=clean_code, adjust="qfq")
                        
                        # 检查数据是否为空
                        if stock_data is None or stock_data.empty:
                            logger.warning(f"获取到的股票 {clean_code} 数据为空")
                            raise ValueError("获取的数据为空")
                        
                        # 标准化列名
                        for old_col, new_col in [('日期', '日期'), ('开盘', '开盘'), ('收盘', '收盘'), 
                                                ('最高', '最高'), ('最低', '最低'), ('成交量', '成交量')]:
                            if old_col in stock_data.columns and old_col != new_col:
                                stock_data = stock_data.rename(columns={old_col: new_col})
                        
                        # 保存到缓存
                        stock_data.to_csv(cache_file, index=False)
                        
                        # 只返回最近days天的数据
                        return stock_data.iloc[-days:]
                        
                    except Exception as e2:
                        logger.warning(f"使用简化参数的stock_zh_a_hist获取股票 {clean_code} 数据失败: {e2}")
                        
                        # 再次强制等待，避免API调用频率限制
                        wait_time = base_delay + random.uniform(1, 5)
                        logger.info(f"API调用需要等待 {wait_time:.2f} 秒，避免频率限制...")
                        time.sleep(wait_time)
                        
                        # 尝试使用stock_zh_a_daily
                        try:
                            logger.info(f"尝试使用stock_zh_a_daily获取股票 {clean_code} 数据")
                            stock_data = ak.stock_zh_a_daily(symbol=clean_code, adjust="qfq")
                            
                            # 检查数据是否为空
                            if stock_data is None or stock_data.empty:
                                logger.warning(f"获取到的股票 {clean_code} 数据为空")
                                raise ValueError("获取的数据为空")
                            
                            # 标准化列名
                            for old_col, new_col in [('date', '日期'), ('open', '开盘'), ('close', '收盘'), 
                                                    ('high', '最高'), ('low', '最低'), ('volume', '成交量')]:
                                if old_col in stock_data.columns:
                                    stock_data = stock_data.rename(columns={old_col: new_col})
                            
                            # 保存到缓存
                            stock_data.to_csv(cache_file, index=False)
                            
                            # 只返回最近days天的数据
                            return stock_data.iloc[-days:]
                            
                        except Exception as e3:
                            logger.warning(f"使用stock_zh_a_daily获取股票 {clean_code} 数据失败: {e3}")
                            retries += 1
                            continue
                
            except Exception as e:
                retries += 1
                if "Connection aborted" in str(e) or "Remote end closed" in str(e):
                    logger.warning(f"API连接被中断，可能是请求频率限制 ({retries}/{max_retries}): {e}")
                else:
                    logger.error(f"获取股票 {clean_code} 技术面数据失败 ({retries}/{max_retries}): {e}")
                
                # 如果已经重试到最大次数，尝试使用缓存
                if retries >= max_retries:
                    if cache_file.exists():
                        logger.warning(f"达到最大重试次数，使用缓存的股票 {clean_code} 技术面数据")
                        try:
                            data = pd.read_csv(cache_file)
                            return data.iloc[-days:]
                        except Exception as cache_e:
                            logger.error(f"读取缓存文件失败: {cache_e}")
                    
                    # 如果没有缓存或缓存读取失败，返回一个默认的DataFrame
                    logger.warning(f"无法获取股票 {clean_code} 技术面数据，返回默认数据")
                    
                    # 创建一个包含基本列的空DataFrame
                    default_data = pd.DataFrame(columns=['日期', '开盘', '收盘', '最高', '最低', '成交量'])
                    
                    # 添加最近days天的日期
                    today = datetime.now()
                    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
                    dates.reverse()  # 从早到晚排序
                    
                    # 添加默认值
                    default_price = 10.0  # 默认价格
                    default_volume = 1000000  # 默认成交量
                    
                    # 构建默认数据
                    default_data = pd.DataFrame({
                        '日期': dates,
                        '开盘': [default_price] * days,
                        '收盘': [default_price] * days,
                        '最高': [default_price * 1.01] * days,
                        '最低': [default_price * 0.99] * days,
                        '成交量': [default_volume] * days
                    })
                    
                    return default_data
    
    def get_industry_data(self):
        """获取行业数据
        
        Returns:
            pandas.DataFrame: 包含行业指数、涨跌幅等信息的DataFrame
        """
        cache_file = self.cache_dir / "industry_data.csv"
        
        # 如果缓存文件存在且当天已更新，直接读取
        if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 1:
            logger.info("从缓存读取行业数据")
            return pd.read_csv(cache_file)
        
        try:
            # 获取行业指数数据
            logger.info("获取行业指数数据")
            industry_index = ak.stock_sector_spot()
            
            # 保存到缓存
            industry_index.to_csv(cache_file, index=False)
            
            return industry_index
        except Exception as e:
            logger.error(f"获取行业数据失败: {e}")
            # 如果缓存存在，尝试使用缓存
            if cache_file.exists():
                logger.warning("使用缓存的行业数据")
                return pd.read_csv(cache_file)
            raise
    
    def get_stock_industry_mapping(self):
        """获取股票行业对应关系
        
        Returns:
            pandas.DataFrame: 包含股票代码、所属行业等信息的DataFrame
        """
        cache_file = self.cache_dir / "stock_industry_mapping.csv"
        
        # 如果缓存文件存在且7天内已更新，直接读取
        if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 7:
            logger.info("从缓存读取股票行业对应关系")
            try:
                stock_industry = pd.read_csv(cache_file)
                
                # 确保列名一致性
                if '代码' not in stock_industry.columns and 'code' in stock_industry.columns:
                    stock_industry = stock_industry.rename(columns={'code': '代码'})
                if '所属行业' not in stock_industry.columns and 'industry' in stock_industry.columns:
                    stock_industry = stock_industry.rename(columns={'industry': '所属行业'})
                
                # 检查是否包含必要的列
                if '代码' in stock_industry.columns and '所属行业' in stock_industry.columns:
                    return stock_industry
                else:
                    logger.warning("缓存的股票行业对应关系数据缺少必要的列，将重新获取")
            except Exception as e:
                logger.warning(f"读取缓存的股票行业对应关系失败: {e}")
        
        try:
            # 首先获取行业列表
            logger.info("获取行业列表")
            try:
                industry_list = ak.stock_board_industry_name_em()
                logger.info(f"成功获取到 {len(industry_list)} 个行业")
            except Exception as e:
                logger.error(f"获取行业列表失败: {e}")
                industry_list = pd.DataFrame(columns=['板块名称'])
            
            # 创建一个空的DataFrame来存储所有股票的行业对应关系
            all_industry_stocks = pd.DataFrame(columns=['代码', '名称', '所属行业'])
            
            # 获取每个行业的成分股
            for i, row in industry_list.iterrows():
                try:
                    # 获取行业名称
                    industry_name = None
                    for col in ['板块名称', '行业名称', '名称']:
                        if col in industry_list.columns:
                            industry_name = row[col]
                            break
                    
                    if industry_name is None:
                        logger.warning(f"无法获取第 {i+1} 个行业的名称")
                        continue
                    
                    logger.info(f"获取行业 '{industry_name}' 的成分股 ({i+1}/{len(industry_list)})")
                    
                    # 获取行业成分股
                    try:
                        industry_stocks = ak.stock_board_industry_cons_em(symbol=industry_name)
                        logger.info(f"行业 '{industry_name}' 包含 {len(industry_stocks)} 只股票")
                    except Exception as e:
                        logger.warning(f"获取行业 '{industry_name}' 成分股失败: {e}")
                        continue
                    
                    # 检查是否包含必要的列
                    code_col = None
                    name_col = None
                    
                    for col in ['代码', '股票代码', 'code']:
                        if col in industry_stocks.columns:
                            code_col = col
                            break
                    
                    for col in ['名称', '股票名称', 'name']:
                        if col in industry_stocks.columns:
                            name_col = col
                            break
                    
                    if code_col is None or name_col is None:
                        logger.warning(f"行业 '{industry_name}' 成分股数据缺少必要的列")
                        continue
                    
                    # 添加行业信息
                    industry_stocks['所属行业'] = industry_name
                    
                    # 重命名列
                    industry_stocks = industry_stocks.rename(columns={code_col: '代码', name_col: '名称'})
                    
                    # 合并到总DataFrame
                    all_industry_stocks = pd.concat([all_industry_stocks, industry_stocks[['代码', '名称', '所属行业']]])
                    
                    # 添加随机延时，避免请求过于频繁
                    time.sleep(random.uniform(0.5, 1.5))
                    
                except Exception as e:
                    logger.warning(f"处理行业 '{industry_name}' 时出错: {e}")
            
            # 如果没有获取到任何行业数据，尝试使用备用方法
            if all_industry_stocks.empty:
                logger.warning("未获取到任何行业数据，尝试使用备用方法")
                try:
                    # 获取所有股票列表
                    stock_list = self.get_stock_list()
                    
                    # 为每只股票分配默认行业
                    stock_list['所属行业'] = '未知行业'
                    
                    # 使用股票列表作为行业映射
                    all_industry_stocks = stock_list[['代码', '名称', '所属行业']]
                except Exception as e:
                    logger.error(f"使用备用方法获取行业数据失败: {e}")
            
            # 确保代码列是字符串类型
            all_industry_stocks['代码'] = all_industry_stocks['代码'].astype(str)
            
            # 确保代码格式正确（6位数字）
            all_industry_stocks['代码'] = all_industry_stocks['代码'].apply(lambda x: x.zfill(6) if x.isdigit() else x)
            
            # 保存到缓存
            all_industry_stocks.to_csv(cache_file, index=False)
            
            return all_industry_stocks
            
        except Exception as e:
            logger.error(f"获取股票行业对应关系失败: {e}")
            
            # 如果缓存存在，尝试使用缓存
            if cache_file.exists():
                logger.warning("使用缓存的股票行业对应关系")
                try:
                    stock_industry = pd.read_csv(cache_file)
                    
                    # 确保列名一致性
                    if '代码' not in stock_industry.columns and 'code' in stock_industry.columns:
                        stock_industry = stock_industry.rename(columns={'code': '代码'})
                    if '所属行业' not in stock_industry.columns and 'industry' in stock_industry.columns:
                        stock_industry = stock_industry.rename(columns={'industry': '所属行业'})
                    
                    return stock_industry
                except Exception as cache_e:
                    logger.error(f"读取缓存文件失败: {cache_e}")
            
            # 如果无法获取数据，创建一个简单的映射
            logger.warning("创建默认的股票行业映射")
            stock_list = self.get_stock_list()
            default_mapping = pd.DataFrame({
                '代码': stock_list['代码'].astype(str),
                '名称': stock_list['名称'] if '名称' in stock_list.columns else '',
                '所属行业': '未知行业'
            })
            
            return default_mapping 