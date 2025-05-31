#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
热点新闻资金流共振分析器：分析新闻热点与资金流入的共振效应
"""

import pandas as pd
import numpy as np
import re
import jieba
import jieba.analyse
from datetime import datetime, timedelta
from loguru import logger
import akshare as ak
import time
import os
from pathlib import Path

class HotNewsFundFlowAnalyzer:
    """热点新闻资金流共振分析器：分析新闻热点与资金流入的共振效应"""
    
    def __init__(self, cache_dir=None):
        """初始化分析器
        
        Args:
            cache_dir: 数据缓存目录，默认为项目根目录下的data/analysis目录
        """
        if cache_dir is None:
            # 默认缓存目录为项目根目录下的data/analysis目录
            self.cache_dir = Path(__file__).parent.parent.parent / "data" / "analysis"
        else:
            self.cache_dir = Path(cache_dir)
            
        # 创建缓存目录
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化jieba分词，添加自定义词典
        self._init_jieba()
        
        logger.info(f"热点新闻资金流共振分析器初始化完成，缓存目录: {self.cache_dir}")
    
    def _init_jieba(self):
        """初始化jieba分词，添加自定义词典"""
        # 加载股票名称和概念板块名称到jieba词典
        try:
            # 尝试获取股票列表
            stock_list = ak.stock_info_a_code_name()
            
            # 创建自定义词典文件
            dict_path = self.cache_dir / "stock_dict.txt"
            with open(dict_path, "w", encoding="utf-8") as f:
                # 添加股票名称
                for _, row in stock_list.iterrows():
                    f.write(f"{row['name']} 10 n\n")  # 股票名称作为名词，权重10
                
                # 尝试获取概念板块名称
                try:
                    concept_list = ak.stock_board_concept_name_em()
                    for _, row in concept_list.iterrows():
                        if '板块名称' in concept_list.columns:
                            f.write(f"{row['板块名称']} 12 n\n")  # 概念板块名称作为名词，权重12
                except Exception as e:
                    logger.warning(f"获取概念板块名称失败: {e}")
                
                # 尝试获取行业板块名称
                try:
                    industry_list = ak.stock_board_industry_name_em()
                    for _, row in industry_list.iterrows():
                        if '板块名称' in industry_list.columns:
                            f.write(f"{row['板块名称']} 12 n\n")  # 行业板块名称作为名词，权重12
                except Exception as e:
                    logger.warning(f"获取行业板块名称失败: {e}")
            
            # 加载自定义词典
            jieba.load_userdict(str(dict_path))
            logger.info("成功加载股票和板块名称到jieba词典")
            
        except Exception as e:
            logger.error(f"初始化jieba词典失败: {e}")
    
    def get_hot_news(self, days=1):
        """获取热点新闻
        
        Args:
            days: 获取最近几天的新闻，默认1天
            
        Returns:
            pandas.DataFrame: 包含新闻标题、内容、时间等信息的DataFrame
        """
        cache_file = self.cache_dir / f"hot_news_{days}d.csv"
        
        # 如果缓存文件存在且当天已更新，直接读取
        if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).seconds < 3600:
            logger.info(f"从缓存读取热点新闻数据 ({days}天)")
            return pd.read_csv(cache_file)
        
        # 合并的新闻DataFrame
        all_news = pd.DataFrame()
        
        # 1. 获取东方财富全球财经快讯
        try:
            logger.info("获取东方财富全球财经快讯")
            em_news = ak.stock_info_global_em()
            
            if not em_news.empty:
                # 添加来源标识
                em_news['来源'] = '东方财富'
                
                # 确保有发布时间列
                if '发布时间' in em_news.columns:
                    # 筛选最近days天的新闻
                    current_date = datetime.now().date()
                    em_news['日期'] = pd.to_datetime(em_news['发布时间']).dt.date
                    em_news = em_news[em_news['日期'] >= (current_date - timedelta(days=days))]
                
                # 重命名列
                if '标题' in em_news.columns and '摘要' in em_news.columns:
                    em_news = em_news.rename(columns={'标题': '标题', '摘要': '内容'})
                
                # 合并到总DataFrame
                if not em_news.empty:
                    if all_news.empty:
                        all_news = em_news[['标题', '内容', '来源']].copy()
                    else:
                        all_news = pd.concat([all_news, em_news[['标题', '内容', '来源']]])
        except Exception as e:
            logger.error(f"获取东方财富全球财经快讯失败: {e}")
        
        # 2. 获取新浪财经全球财经快讯
        try:
            logger.info("获取新浪财经全球财经快讯")
            sina_news = ak.stock_info_global_sina()
            
            if not sina_news.empty:
                # 添加来源标识
                sina_news['来源'] = '新浪财经'
                
                # 确保有时间列
                if '时间' in sina_news.columns:
                    # 筛选最近days天的新闻
                    current_date = datetime.now().date()
                    sina_news['日期'] = pd.to_datetime(sina_news['时间']).dt.date
                    sina_news = sina_news[sina_news['日期'] >= (current_date - timedelta(days=days))]
                
                # 重命名列
                if '内容' in sina_news.columns:
                    # 从内容中提取标题（假设内容的前20个字符作为标题）
                    sina_news['标题'] = sina_news['内容'].apply(lambda x: x[:min(20, len(x))] + '...')
                
                # 合并到总DataFrame
                if not sina_news.empty and '标题' in sina_news.columns and '内容' in sina_news.columns:
                    if all_news.empty:
                        all_news = sina_news[['标题', '内容', '来源']].copy()
                    else:
                        all_news = pd.concat([all_news, sina_news[['标题', '内容', '来源']]])
        except Exception as e:
            logger.error(f"获取新浪财经全球财经快讯失败: {e}")
        
        # 如果没有获取到任何新闻，返回空DataFrame
        if all_news.empty:
            logger.warning("未获取到任何热点新闻")
            return pd.DataFrame()
        
        # 保存到缓存
        all_news.to_csv(cache_file, index=False)
        
        return all_news
    
    def get_fund_flow_data(self, days_list=[1, 3, 5]):
        """获取资金流向数据
        
        Args:
            days_list: 获取的天数列表，默认[1, 3, 5]天
            
        Returns:
            dict: 包含不同天数的资金流向数据的字典
        """
        fund_flow_dict = {}
        
        # 对应的indicator参数
        days_map = {1: "今日", 3: "3日", 5: "5日", 10: "10日"}
        
        for days in days_list:
            if days not in days_map:
                logger.warning(f"不支持的天数: {days}，跳过")
                continue
                
            cache_file = self.cache_dir / f"fund_flow_{days}d.csv"
            
            # 如果缓存文件存在且当天已更新，直接读取
            if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).seconds < 3600:
                logger.info(f"从缓存读取资金流向数据 ({days}天)")
                fund_flow_dict[days] = pd.read_csv(cache_file)
                continue
            
            try:
                logger.info(f"获取{days}天资金流向数据")
                indicator = days_map[days]
                fund_flow = ak.stock_individual_fund_flow_rank(indicator=indicator)
                
                if not fund_flow.empty:
                    # 保存到缓存
                    fund_flow.to_csv(cache_file, index=False)
                    fund_flow_dict[days] = fund_flow
                else:
                    logger.warning(f"获取{days}天资金流向数据为空")
            except Exception as e:
                logger.error(f"获取{days}天资金流向数据失败: {e}")
                
                # 如果缓存存在，尝试使用缓存
                if cache_file.exists():
                    logger.warning(f"使用缓存的{days}天资金流向数据")
                    fund_flow_dict[days] = pd.read_csv(cache_file)
            
            # 添加延时，避免请求过于频繁
            time.sleep(1)
        
        return fund_flow_dict
    
    def extract_keywords_from_news(self, news_df, top_n=100):
        """从新闻中提取关键词
        
        Args:
            news_df: 新闻DataFrame
            top_n: 返回前N个关键词
            
        Returns:
            dict: 关键词及其权重的字典
        """
        if news_df.empty:
            logger.warning("新闻数据为空，无法提取关键词")
            return {}
        
        # 合并标题和内容
        news_text = ""
        
        if '标题' in news_df.columns:
            # 标题权重更高，重复3次
            for title in news_df['标题'].dropna():
                news_text += title + " " + title + " " + title + " "
        
        if '内容' in news_df.columns:
            for content in news_df['内容'].dropna():
                news_text += content + " "
        
        # 使用jieba提取关键词
        keywords = jieba.analyse.extract_tags(news_text, topK=top_n, withWeight=True)
        
        # 转换为字典
        keywords_dict = {word: weight for word, weight in keywords}
        
        logger.info(f"从新闻中提取了{len(keywords_dict)}个关键词")
        
        return keywords_dict
    
    def match_keywords_with_stocks(self, keywords_dict, stock_list=None):
        """将关键词与股票匹配
        
        Args:
            keywords_dict: 关键词及其权重的字典
            stock_list: 股票列表DataFrame，如果为None则获取所有A股
            
        Returns:
            pandas.DataFrame: 包含股票代码、名称、匹配分数等信息的DataFrame
        """
        if not keywords_dict:
            logger.warning("关键词为空，无法进行匹配")
            return pd.DataFrame()
        
        # 如果没有提供股票列表，获取所有A股
        if stock_list is None:
            try:
                stock_list = ak.stock_info_a_code_name()
            except Exception as e:
                logger.error(f"获取A股列表失败: {e}")
                return pd.DataFrame()
        
        # 确保股票列表包含必要的列
        code_column = None
        name_column = None
        
        for col in ['代码', 'code']:
            if col in stock_list.columns:
                code_column = col
                break
        
        for col in ['名称', 'name']:
            if col in stock_list.columns:
                name_column = col
                break
        
        if code_column is None or name_column is None:
            logger.error("股票列表缺少代码或名称列")
            return pd.DataFrame()
        
        # 创建结果DataFrame
        result = []
        
        # 对每只股票进行关键词匹配
        for _, row in stock_list.iterrows():
            stock_code = row[code_column]
            stock_name = row[name_column]
            
            # 计算匹配分数
            score = 0
            matched_keywords = []
            
            # 检查股票名称是否包含关键词
            for keyword, weight in keywords_dict.items():
                if keyword in stock_name:
                    score += weight * 2  # 股票名称匹配权重加倍
                    matched_keywords.append(keyword)
            
            # 如果有匹配，添加到结果中
            if score > 0:
                result.append({
                    '代码': stock_code,
                    '名称': stock_name,
                    '新闻匹配分数': score,
                    '匹配关键词': ','.join(matched_keywords)
                })
        
        # 转换为DataFrame
        result_df = pd.DataFrame(result)
        
        if not result_df.empty:
            # 按匹配分数降序排序
            result_df = result_df.sort_values('新闻匹配分数', ascending=False)
            
            # 确保代码列是字符串类型
            result_df['代码'] = result_df['代码'].astype(str)
            
            # 确保代码格式正确（6位数字）
            result_df['代码'] = result_df['代码'].apply(lambda x: x.zfill(6) if x.isdigit() else x)
        
        logger.info(f"关键词匹配到{len(result_df)}只股票")
        
        return result_df
    
    def match_keywords_with_concepts(self, keywords_dict):
        """将关键词与概念板块匹配
        
        Args:
            keywords_dict: 关键词及其权重的字典
            
        Returns:
            pandas.DataFrame: 包含概念板块名称、匹配分数等信息的DataFrame
        """
        if not keywords_dict:
            logger.warning("关键词为空，无法进行匹配")
            return pd.DataFrame()
        
        # 获取概念板块列表
        try:
            concept_list = ak.stock_board_concept_name_em()
        except Exception as e:
            logger.error(f"获取概念板块列表失败: {e}")
            return pd.DataFrame()
        
        # 确保概念板块列表包含必要的列
        name_column = None
        
        for col in ['板块名称', '名称']:
            if col in concept_list.columns:
                name_column = col
                break
        
        if name_column is None:
            logger.error("概念板块列表缺少名称列")
            return pd.DataFrame()
        
        # 创建结果DataFrame
        result = []
        
        # 对每个概念板块进行关键词匹配
        for _, row in concept_list.iterrows():
            concept_name = row[name_column]
            
            # 计算匹配分数
            score = 0
            matched_keywords = []
            
            # 检查概念板块名称是否包含关键词
            for keyword, weight in keywords_dict.items():
                if keyword in concept_name:
                    score += weight * 1.5  # 概念板块名称匹配权重适当提高
                    matched_keywords.append(keyword)
            
            # 如果有匹配，添加到结果中
            if score > 0:
                result.append({
                    '概念板块': concept_name,
                    '新闻匹配分数': score,
                    '匹配关键词': ','.join(matched_keywords)
                })
        
        # 转换为DataFrame
        result_df = pd.DataFrame(result)
        
        if not result_df.empty:
            # 按匹配分数降序排序
            result_df = result_df.sort_values('新闻匹配分数', ascending=False)
        
        logger.info(f"关键词匹配到{len(result_df)}个概念板块")
        
        return result_df
    
    def get_concept_stocks(self, concept_name):
        """获取概念板块的成分股
        
        Args:
            concept_name: 概念板块名称
            
        Returns:
            pandas.DataFrame: 包含股票代码、名称等信息的DataFrame
        """
        try:
            logger.info(f"获取概念板块 {concept_name} 的成分股")
            concept_stocks = ak.stock_board_concept_cons_em(symbol=concept_name)
            
            if not concept_stocks.empty:
                # 确保包含必要的列
                if '代码' in concept_stocks.columns:
                    # 确保代码列是字符串类型
                    concept_stocks['代码'] = concept_stocks['代码'].astype(str)
                    
                    # 确保代码格式正确（6位数字）
                    concept_stocks['代码'] = concept_stocks['代码'].apply(lambda x: x.zfill(6) if x.isdigit() else x)
                
                return concept_stocks
            else:
                logger.warning(f"概念板块 {concept_name} 成分股为空")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取概念板块 {concept_name} 成分股失败: {e}")
            return pd.DataFrame()
    
    def analyze(self, news_days=1, fund_flow_days_list=[1, 3, 5], top_n=100):
        """分析新闻热点与资金流入的共振效应
        
        Args:
            news_days: 获取最近几天的新闻，默认1天
            fund_flow_days_list: 获取的资金流向天数列表，默认[1, 3, 5]天
            top_n: 返回前N个结果
            
        Returns:
            pandas.DataFrame: 包含股票代码、名称、新闻匹配分数、资金流入等信息的DataFrame
        """
        # 1. 获取热点新闻
        news_df = self.get_hot_news(days=news_days)
        
        # 2. 提取新闻关键词
        keywords_dict = self.extract_keywords_from_news(news_df)
        
        # 3. 获取资金流向数据
        fund_flow_dict = self.get_fund_flow_data(days_list=fund_flow_days_list)
        
        # 4. 将关键词与股票直接匹配
        matched_stocks = self.match_keywords_with_stocks(keywords_dict)
        
        # 5. 将关键词与概念板块匹配
        matched_concepts = self.match_keywords_with_concepts(keywords_dict)
        
        # 6. 获取匹配概念板块的所有成分股
        concept_stocks = pd.DataFrame()
        
        if not matched_concepts.empty:
            for _, concept_row in matched_concepts.head(10).iterrows():  # 只处理前10个匹配度最高的概念板块
                concept_name = concept_row['概念板块']
                concept_score = concept_row['新闻匹配分数']
                
                # 获取概念板块成分股
                stocks = self.get_concept_stocks(concept_name)
                
                if not stocks.empty:
                    # 添加概念板块信息和匹配分数
                    stocks['概念板块'] = concept_name
                    stocks['新闻匹配分数'] = concept_score * 0.8  # 通过概念板块匹配的股票，分数稍低
                    
                    # 合并到总成分股DataFrame
                    if concept_stocks.empty:
                        concept_stocks = stocks.copy()
                    else:
                        concept_stocks = pd.concat([concept_stocks, stocks])
                
                # 添加延时，避免请求过于频繁
                time.sleep(1)
        
        # 7. 合并直接匹配的股票和通过概念板块匹配的股票
        all_matched_stocks = pd.DataFrame()
        
        if not matched_stocks.empty:
            all_matched_stocks = matched_stocks.copy()
        
        if not concept_stocks.empty:
            # 确保包含必要的列
            required_columns = ['代码', '名称', '新闻匹配分数']
            
            if all([col in concept_stocks.columns for col in required_columns]):
                if all_matched_stocks.empty:
                    all_matched_stocks = concept_stocks[required_columns].copy()
                else:
                    # 合并，如果有重复的股票，保留新闻匹配分数更高的记录
                    all_matched_stocks = pd.concat([all_matched_stocks[required_columns], concept_stocks[required_columns]])
                    all_matched_stocks = all_matched_stocks.sort_values('新闻匹配分数', ascending=False).drop_duplicates('代码', keep='first')
        
        # 8. 如果没有任何匹配的股票，返回空DataFrame
        if all_matched_stocks.empty:
            logger.warning("未找到任何与新闻热点匹配的股票")
            return pd.DataFrame()
        
        # 9. 与资金流向数据合并
        result = all_matched_stocks.copy()
        
        for days, fund_flow in fund_flow_dict.items():
            if fund_flow.empty:
                continue
                
            # 确保资金流向数据包含必要的列
            if '代码' not in fund_flow.columns:
                continue
                
            # 确保代码列是字符串类型
            fund_flow['代码'] = fund_flow['代码'].astype(str)
            
            # 确保代码格式正确（6位数字）
            fund_flow['代码'] = fund_flow['代码'].apply(lambda x: x.zfill(6) if x.isdigit() else x)
            
            # 选择需要的列
            columns_to_merge = ['代码']
            
            # 根据天数选择对应的列
            if days == 1:
                prefix = '今日'
            elif days == 3:
                prefix = '3日'
            elif days == 5:
                prefix = '5日'
            else:
                prefix = f'{days}日'
            
            # 添加主力净流入列
            main_flow_columns = [col for col in fund_flow.columns if f'{prefix}主力净流入' in col]
            columns_to_merge.extend(main_flow_columns)
            
            # 添加超大单净流入列
            super_flow_columns = [col for col in fund_flow.columns if f'{prefix}超大单净流入' in col]
            columns_to_merge.extend(super_flow_columns)
            
            # 添加大单净流入列
            big_flow_columns = [col for col in fund_flow.columns if f'{prefix}大单净流入' in col]
            columns_to_merge.extend(big_flow_columns)
            
            # 添加涨跌幅列
            change_columns = [col for col in fund_flow.columns if f'{prefix}涨跌幅' in col]
            columns_to_merge.extend(change_columns)
            
            # 如果有需要合并的列，进行合并
            if len(columns_to_merge) > 1:
                # 使用左连接合并，保留所有匹配的股票
                result = pd.merge(result, fund_flow[columns_to_merge], on='代码', how='left')
        
        # 10. 计算综合得分
        if not result.empty:
            # 初始化综合得分为新闻匹配分数
            result['综合得分'] = result['新闻匹配分数']
            
            # 添加资金流入得分
            for days in fund_flow_days_list:
                if days == 1:
                    prefix = '今日'
                    weight = 1.0  # 今日资金流入权重最高
                elif days == 3:
                    prefix = '3日'
                    weight = 0.8  # 3日资金流入权重次之
                elif days == 5:
                    prefix = '5日'
                    weight = 0.6  # 5日资金流入权重再次
                else:
                    prefix = f'{days}日'
                    weight = 0.4  # 其他天数资金流入权重最低
                
                # 主力净流入-净额
                col = f'{prefix}主力净流入-净额'
                if col in result.columns:
                    # 将净额转换为百万元单位，并加权
                    result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)
                    result['综合得分'] += result[col] / 1000000 * weight
                
                # 主力净流入-净占比
                col = f'{prefix}主力净流入-净占比'
                if col in result.columns:
                    # 将净占比直接加权
                    result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)
                    result['综合得分'] += result[col] * weight * 5
            
            # 按综合得分降序排序
            result = result.sort_values('综合得分', ascending=False)
            
            # 只保留前top_n个结果
            result = result.head(top_n)
        
        return result

if __name__ == "__main__":
    # 测试代码
    analyzer = HotNewsFundFlowAnalyzer()
    result = analyzer.analyze()
    print(result.head(20)) 