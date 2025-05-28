#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据管理器模块：负责从不同来源获取数据
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import json

logger = logging.getLogger(__name__)

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
            return pd.read_csv(cache_file)
        
        try:
            # 获取北向资金流入数据
            logger.info(f"获取北向资金流入数据 ({days}天)")
            north_data = ak.stock_hsgt_north_net_flow_in_em(symbol="北向")
            
            # 获取行业资金流入数据
            logger.info("获取行业资金流入数据")
            industry_flow = ak.stock_sector_fund_flow_rank(indicator="今日")
            
            # 获取个股资金流入数据
            logger.info("获取个股资金流入数据")
            stock_flow = ak.stock_individual_fund_flow_rank(indicator="今日")
            
            # 处理数据
            stock_flow['代码'] = stock_flow['代码'].apply(lambda x: str(x).zfill(6))
            
            # 保存到缓存
            stock_flow.to_csv(cache_file, index=False)
            
            return stock_flow
        except Exception as e:
            logger.error(f"获取资金流向数据失败: {e}")
            # 如果缓存存在，尝试使用缓存
            if cache_file.exists():
                logger.warning("使用缓存的资金流向数据")
                return pd.read_csv(cache_file)
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
            return pd.read_csv(cache_file)
        
        try:
            # 获取东方财富股吧热度排行
            logger.info("获取东方财富股吧热度排行")
            stock_hot = ak.stock_hot_rank_em()
            
            # 获取新浪股吧热度排行
            logger.info("获取新浪股吧热度排行")
            stock_hot_sina = ak.stock_hot_rank_detail_realtime_em()
            
            # 合并数据
            stock_hot.rename(columns={"股票代码": "代码", "股票简称": "名称", "贴数": "讨论数量"}, inplace=True)
            stock_hot['代码'] = stock_hot['代码'].apply(lambda x: str(x).zfill(6))
            
            # 保存到缓存
            stock_hot.to_csv(cache_file, index=False)
            
            return stock_hot
        except Exception as e:
            logger.error(f"获取社交媒体讨论数据失败: {e}")
            # 如果缓存存在，尝试使用缓存
            if cache_file.exists():
                logger.warning("使用缓存的社交媒体讨论数据")
                return pd.read_csv(cache_file)
            raise
    
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
            
            # 如果指定了股票列表，筛选相应的股票
            if stock_list is not None:
                fundamental_data = fundamental_data[fundamental_data['代码'].isin(stock_list)]
            
            return fundamental_data
        
        try:
            # 获取所有股票的基本面数据
            logger.info("获取股票基本面数据")
            
            # 如果没有指定股票列表，获取所有股票
            if stock_list is None:
                stock_list = self.get_stock_list()['代码'].tolist()
            
            # 获取市盈率、市净率等数据
            pe_data = ak.stock_a_pe()
            pb_data = ak.stock_a_pb()
            
            # 获取财务指标数据
            financial_indicator = ak.stock_financial_analysis_indicator()
            
            # 合并数据
            pe_data.rename(columns={"代码": "代码", "市盈率-动态": "PE"}, inplace=True)
            pb_data.rename(columns={"代码": "代码", "市净率": "PB"}, inplace=True)
            
            # 合并PE和PB数据
            merged_data = pd.merge(pe_data[['代码', 'PE']], pb_data[['代码', 'PB']], on='代码', how='outer')
            
            # 保存到缓存
            merged_data.to_csv(cache_file, index=False)
            
            # 筛选指定的股票
            if stock_list is not None:
                merged_data = merged_data[merged_data['代码'].isin(stock_list)]
            
            return merged_data
        except Exception as e:
            logger.error(f"获取股票基本面数据失败: {e}")
            # 如果缓存存在，尝试使用缓存
            if cache_file.exists():
                logger.warning("使用缓存的股票基本面数据")
                fundamental_data = pd.read_csv(cache_file)
                
                # 如果指定了股票列表，筛选相应的股票
                if stock_list is not None:
                    fundamental_data = fundamental_data[fundamental_data['代码'].isin(stock_list)]
                
                return fundamental_data
            raise
    
    def get_stock_technical_data(self, stock_code, days=60):
        """获取股票技术面数据
        
        Args:
            stock_code: 股票代码
            days: 获取最近几天的数据，默认60天
            
        Returns:
            pandas.DataFrame: 包含股票价格、成交量等信息的DataFrame
        """
        cache_file = self.cache_dir / "market_data" / f"{stock_code}_daily.csv"
        
        # 如果缓存文件存在且当天已更新，直接读取
        if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 1:
            logger.info(f"从缓存读取股票 {stock_code} 技术面数据")
            data = pd.read_csv(cache_file)
            return data.iloc[-days:]
        
        try:
            # 获取股票日线数据
            logger.info(f"获取股票 {stock_code} 技术面数据")
            stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
            
            # 保存到缓存
            stock_data.to_csv(cache_file, index=False)
            
            return stock_data.iloc[-days:]
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 技术面数据失败: {e}")
            # 如果缓存存在，尝试使用缓存
            if cache_file.exists():
                logger.warning(f"使用缓存的股票 {stock_code} 技术面数据")
                data = pd.read_csv(cache_file)
                return data.iloc[-days:]
            raise
    
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
            return pd.read_csv(cache_file)
        
        try:
            # 获取股票所属行业数据
            logger.info("获取股票所属行业数据")
            stock_industry = ak.stock_sector_detail()
            
            # 处理数据
            stock_industry['代码'] = stock_industry['代码'].apply(lambda x: str(x).zfill(6))
            
            # 保存到缓存
            stock_industry.to_csv(cache_file, index=False)
            
            return stock_industry
        except Exception as e:
            logger.error(f"获取股票行业对应关系失败: {e}")
            # 如果缓存存在，尝试使用缓存
            if cache_file.exists():
                logger.warning("使用缓存的股票行业对应关系")
                return pd.read_csv(cache_file)
            raise 