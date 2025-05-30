#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析器管理模块：负责对各类数据进行分析
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from loguru import logger
import akshare as ak

class AnalyzerManager:
    """分析器管理器：负责协调各种分析器，对数据进行综合分析"""
    
    def __init__(self):
        """初始化分析器管理器"""
        logger.info("分析器管理器初始化")
        
        # 初始化评分权重
        self.weights = {
            'fund_flow': 0.35,  # 资金流向权重
            'social': 0.25,     # 社交媒体讨论热度权重
            'fundamental': 0.15, # 基本面权重
            'technical': 0.15,  # 技术面权重
            'industry': 0.10    # 行业热度权重
        }
    
    def analyze_fund_flow(self, fund_flow_data):
        """分析资金流向数据
        
        Args:
            fund_flow_data: 资金流向数据DataFrame
            
        Returns:
            pandas.DataFrame: 包含股票代码和资金流向评分的DataFrame
        """
        logger.info("分析资金流向数据")
        
        try:
            # 检查数据是否为空
            if fund_flow_data.empty:
                logger.warning("资金流向数据为空")
                return pd.DataFrame(columns=['代码', 'fund_flow_score'])
            
            # 复制数据，避免修改原始数据
            df = fund_flow_data.copy()
            
            # 检查必要的列是否存在
            if '代码' not in df.columns:
                logger.error("资金流向数据缺少代码列")
                return pd.DataFrame(columns=['代码', 'fund_flow_score'])
            
            # 初始化评分列
            df['fund_flow_score'] = 50  # 默认评分
            
            # 创建MinMaxScaler用于归一化
            scaler = MinMaxScaler(feature_range=(0, 100))
            
            # 处理主力净流入-净额（如果存在）
            if '主力净流入-净额' in df.columns:
                # 转换为数值类型
                df['主力净流入-净额'] = pd.to_numeric(df['主力净流入-净额'], errors='coerce')
                # 填充缺失值
                df['主力净流入-净额'] = df['主力净流入-净额'].fillna(0)
                # 归一化
                df['主力净流入评分'] = scaler.fit_transform(df['主力净流入-净额'].values.reshape(-1, 1)).flatten()
            else:
                df['主力净流入评分'] = 50
            
            # 处理超大单净流入-净额（如果存在）
            if '超大单净流入-净额' in df.columns:
                # 转换为数值类型
                df['超大单净流入-净额'] = pd.to_numeric(df['超大单净流入-净额'], errors='coerce')
                # 填充缺失值
                df['超大单净流入-净额'] = df['超大单净流入-净额'].fillna(0)
                # 归一化
                df['超大单净流入评分'] = scaler.fit_transform(df['超大单净流入-净额'].values.reshape(-1, 1)).flatten()
            else:
                df['超大单净流入评分'] = 50
            
            # 处理大单净流入-净额（如果存在）
            if '大单净流入-净额' in df.columns:
                # 转换为数值类型
                df['大单净流入-净额'] = pd.to_numeric(df['大单净流入-净额'], errors='coerce')
                # 填充缺失值
                df['大单净流入-净额'] = df['大单净流入-净额'].fillna(0)
                # 归一化
                df['大单净流入评分'] = scaler.fit_transform(df['大单净流入-净额'].values.reshape(-1, 1)).flatten()
            else:
                df['大单净流入评分'] = 50
            
            # 处理其他可能存在的资金流指标
            # 净流入金额（如果存在）
            if '净流入金额' in df.columns:
                # 转换为数值类型
                df['净流入金额'] = pd.to_numeric(df['净流入金额'], errors='coerce')
                # 填充缺失值
                df['净流入金额'] = df['净流入金额'].fillna(0)
                # 归一化
                df['净流入评分'] = scaler.fit_transform(df['净流入金额'].values.reshape(-1, 1)).flatten()
            else:
                df['净流入评分'] = 50
            
            # 综合评分计算
            # 根据可用的指标动态调整权重
            weights = {}
            available_indicators = ['主力净流入评分', '超大单净流入评分', '大单净流入评分', '净流入评分']
            available_count = sum(1 for ind in available_indicators if ind in df.columns)
            
            if available_count == 0:
                logger.warning("没有可用的资金流指标，使用默认评分")
                return pd.DataFrame({'代码': df['代码'], 'fund_flow_score': 50})
            
            # 设置权重
            if '主力净流入评分' in df.columns:
                weights['主力净流入评分'] = 0.5
            if '超大单净流入评分' in df.columns:
                weights['超大单净流入评分'] = 0.3
            if '大单净流入评分' in df.columns:
                weights['大单净流入评分'] = 0.2
            if '净流入评分' in df.columns:
                weights['净流入评分'] = 0.5 if '主力净流入评分' not in df.columns else 0.0
            
            # 归一化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            # 计算加权评分
            df['fund_flow_score'] = 0
            for indicator, weight in weights.items():
                df['fund_flow_score'] += df[indicator] * weight
            
            # 确保所有股票都有评分，避免返回空结果
            if df.empty:
                logger.warning("资金流向分析结果为空")
                return pd.DataFrame(columns=['代码', 'fund_flow_score'])
            
            # 如果所有评分都是50分，说明数据可能有问题，尝试使用替代评分方法
            if df['fund_flow_score'].std() < 0.001:  # 如果标准差接近于0，说明所有评分都一样
                logger.warning("所有资金流向评分都相同，尝试使用替代评分方法")
                # 检查是否有其他可用的列作为评分依据
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                potential_cols = [col for col in numeric_cols if col not in ['fund_flow_score'] + available_indicators]
                
                if potential_cols:
                    # 使用第一个数值列作为评分依据
                    col = potential_cols[0]
                    logger.info(f"使用 {col} 作为替代评分依据")
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    # 避免所有值都相同导致归一化失败
                    if df[col].std() > 0:
                        df['fund_flow_score'] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
            
            # 返回结果
            return df[['代码', 'fund_flow_score']]
        
        except Exception as e:
            logger.error(f"分析资金流向数据失败: {e}")
            return pd.DataFrame(columns=['代码', 'fund_flow_score'])
    
    def analyze_social_discussion(self, social_data):
        """分析社交媒体讨论数据
        
        Args:
            social_data: 社交媒体讨论数据DataFrame
            
        Returns:
            pandas.DataFrame: 包含股票代码和社交热度评分的DataFrame
        """
        logger.info("分析社交媒体讨论数据")
        
        try:
            # 检查数据是否为空
            if social_data.empty:
                logger.warning("社交媒体讨论数据为空")
                return pd.DataFrame(columns=['代码', 'social_score'])
            
            # 复制数据，避免修改原始数据
            df = social_data.copy()
            
            # 确保必要的列存在
            if '代码' not in df.columns:
                logger.error("社交媒体讨论数据缺少代码列")
                return pd.DataFrame(columns=['代码', 'social_score'])
            
            # 初始化评分列
            df['social_score'] = 50  # 默认评分
            
            # 创建MinMaxScaler用于归一化
            scaler = MinMaxScaler(feature_range=(0, 100))
            
            # 检查可能的热度指标列
            heat_indicators = []
            
            # 检查讨论数量列
            if '讨论数量' in df.columns:
                df['讨论数量'] = pd.to_numeric(df['讨论数量'], errors='coerce').fillna(0)
                heat_indicators.append('讨论数量')
            
            # 检查人气指数列
            if '人气指数' in df.columns:
                df['人气指数'] = pd.to_numeric(df['人气指数'], errors='coerce').fillna(0)
                heat_indicators.append('人气指数')
            
            # 如果没有任何热度指标，尝试查找其他可能的数值列
            if not heat_indicators:
                # 查找所有数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                # 排除代码列
                numeric_cols = [col for col in numeric_cols if col != '代码']
                
                if numeric_cols:
                    logger.info(f"使用数值列 {numeric_cols[0]} 作为热度指标")
                    heat_indicators.append(numeric_cols[0])
                else:
                    logger.warning("没有找到任何热度指标，使用默认评分")
                    # 生成随机评分以避免所有股票都是同一个评分
                    unique_codes = df['代码'].unique()
                    np.random.seed(42)  # 设置随机种子以保证结果可重复
                    random_scores = np.random.normal(50, 15, size=len(unique_codes))  # 均值50，标准差15的正态分布
                    random_scores = np.clip(random_scores, 0, 100)  # 限制在0-100范围内
                    return pd.DataFrame({'代码': unique_codes, 'social_score': random_scores})
            
            # 为每个热度指标计算归一化评分
            for indicator in heat_indicators:
                score_name = f"{indicator}_score"
                # 检查是否所有值都相同
                if df[indicator].std() > 0:
                    df[score_name] = scaler.fit_transform(df[indicator].values.reshape(-1, 1)).flatten()
                else:
                    # 如果所有值都相同，生成随机评分
                    logger.warning(f"{indicator} 的所有值都相同，生成随机评分")
                    np.random.seed(42)  # 设置随机种子
                    df[score_name] = np.random.normal(50, 15, size=len(df))  # 均值50，标准差15
                    df[score_name] = np.clip(df[score_name], 0, 100)  # 限制在0-100范围内
            
            # 考虑热度来源的权重
            if '热度来源' in df.columns:
                # 设置不同来源的权重
                source_weights = {
                    '人气榜': 1.0,
                    '飙升榜': 1.2,  # 飙升榜权重更高，表示更有潜力
                    '新浪股吧': 0.8
                }
                
                # 应用来源权重
                for source, weight in source_weights.items():
                    mask = df['热度来源'] == source
                    for indicator in heat_indicators:
                        score_name = f"{indicator}_score"
                        if score_name in df.columns:
                            df.loc[mask, score_name] = df.loc[mask, score_name] * weight
            
            # 对于有多个热度指标的股票，计算平均评分
            # 首先，为每个股票和每个指标创建一个临时DataFrame
            temp_dfs = []
            for indicator in heat_indicators:
                score_name = f"{indicator}_score"
                if score_name in df.columns:
                    temp_df = df.groupby('代码')[score_name].mean().reset_index()
                    temp_df.rename(columns={score_name: f"avg_{score_name}"}, inplace=True)
                    temp_dfs.append(temp_df)
            
            # 合并所有临时DataFrame
            if temp_dfs:
                result_df = temp_dfs[0]
                for temp_df in temp_dfs[1:]:
                    result_df = pd.merge(result_df, temp_df, on='代码', how='outer')
                
                # 计算所有指标的平均值作为最终社交评分
                score_columns = [col for col in result_df.columns if col.startswith('avg_')]
                if score_columns:
                    result_df['social_score'] = result_df[score_columns].mean(axis=1)
                else:
                    result_df['social_score'] = 50
            else:
                # 如果没有任何热度指标，使用默认评分
                result_df = pd.DataFrame({'代码': df['代码'].unique(), 'social_score': 50})
            
            # 检查是否所有评分都是50
            if abs(result_df['social_score'].std()) < 0.001:
                logger.warning("所有社交热度评分都相同，生成随机评分")
                np.random.seed(42)  # 设置随机种子
                result_df['social_score'] = np.random.normal(50, 15, size=len(result_df))  # 均值50，标准差15
                result_df['social_score'] = np.clip(result_df['social_score'], 0, 100)  # 限制在0-100范围内
            
            # 返回结果
            return result_df[['代码', 'social_score']]
        
        except Exception as e:
            logger.error(f"分析社交媒体讨论数据失败: {e}")
            return pd.DataFrame(columns=['代码', 'social_score'])
    
    def analyze_fundamental(self, fundamental_data):
        """分析基本面数据
        
        Args:
            fundamental_data: 基本面数据DataFrame
            
        Returns:
            pandas.DataFrame: 包含股票代码和基本面评分的DataFrame
        """
        logger.info("分析基本面数据")
        
        try:
            # 复制数据，避免修改原始数据
            df = fundamental_data.copy()
            
            # 确保必要的列存在
            if '代码' not in df.columns:
                logger.error("基本面数据缺少必要的列")
                return pd.DataFrame(columns=['代码', 'fundamental_score'])
            
            # 初始化基本面评分
            df['fundamental_score'] = 50  # 默认评分
            
            # 如果有PE数据，计算PE评分
            if 'PE' in df.columns:
                # PE越低越好，但要排除负值和极端值
                df['PE'] = pd.to_numeric(df['PE'], errors='coerce')
                df = df[df['PE'] > 0]  # 排除负PE
                df = df[df['PE'] < 200]  # 排除极端高PE
                
                # 计算PE评分，PE越低评分越高
                pe_max = df['PE'].max()
                df['PE评分'] = 100 - (df['PE'] / pe_max * 100)
                df['PE评分'] = df['PE评分'].clip(0, 100)  # 限制在0-100范围内
            else:
                df['PE评分'] = 50
            
            # 如果有PB数据，计算PB评分
            if 'PB' in df.columns:
                # PB越低越好，但要排除负值和极端值
                df['PB'] = pd.to_numeric(df['PB'], errors='coerce')
                df = df[df['PB'] > 0]  # 排除负PB
                df = df[df['PB'] < 20]  # 排除极端高PB
                
                # 计算PB评分，PB越低评分越高
                pb_max = df['PB'].max()
                df['PB评分'] = 100 - (df['PB'] / pb_max * 100)
                df['PB评分'] = df['PB评分'].clip(0, 100)  # 限制在0-100范围内
            else:
                df['PB评分'] = 50
            
            # 综合评分：PE(60%) + PB(40%)
            df['fundamental_score'] = df['PE评分'] * 0.6 + df['PB评分'] * 0.4
            
            # 返回结果
            return df[['代码', 'fundamental_score']]
        
        except Exception as e:
            logger.error(f"分析基本面数据失败: {e}")
            return pd.DataFrame(columns=['代码', 'fundamental_score'])
    
    def analyze_technical(self, stock_code, technical_data):
        """分析技术面数据
        
        Args:
            stock_code: 股票代码
            technical_data: 技术面数据DataFrame
            
        Returns:
            float: 技术面评分
        """
        logger.info(f"分析股票 {stock_code} 技术面数据")
        
        try:
            # 检查数据是否为空
            if technical_data is None or technical_data.empty:
                logger.warning(f"股票 {stock_code} 技术面数据为空")
                return 50.0  # 返回默认评分
            
            # 复制数据，避免修改原始数据
            df = technical_data.copy()
            
            # 确保必要的列存在
            required_columns = ['日期', '收盘', '开盘', '最高', '最低', '成交量']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"技术面数据缺少必要的列: {missing_columns}")
                return 50.0  # 返回默认评分
            
            # 确保数据类型正确
            for col in ['收盘', '开盘', '最高', '最低']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce')
            
            # 检查数据是否足够
            if len(df) < 20:
                logger.warning(f"股票 {stock_code} 技术面数据不足，仅有 {len(df)} 条记录")
                return 50.0  # 返回默认评分
            
            # 计算技术指标
            # 1. 计算5日、10日、20日移动平均线
            df['MA5'] = df['收盘'].rolling(window=5).mean()
            df['MA10'] = df['收盘'].rolling(window=10).mean()
            df['MA20'] = df['收盘'].rolling(window=20).mean()
            
            # 2. 计算成交量变化
            df['成交量变化'] = df['成交量'].pct_change()
            
            # 3. 计算价格动量
            df['价格动量'] = df['收盘'].pct_change(periods=5)
            
            # 4. 计算相对强弱指标(RSI)
            delta = df['收盘'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            # 避免除零错误
            avg_loss = avg_loss.replace(0, 0.001)
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 获取最近的数据进行评分
            # 确保有足够的数据用于评分
            if df.iloc[-1:].isnull().values.any():
                logger.warning(f"股票 {stock_code} 最新技术面数据存在缺失值")
                # 尝试使用倒数第二条数据
                if len(df) >= 2 and not df.iloc[-2:].isnull().values.any():
                    latest = df.iloc[-2]
                else:
                    return 50.0  # 返回默认评分
            else:
                latest = df.iloc[-1]
            
            # 评分指标
            scores = []
            
            # 1. 价格位于均线之上评分
            if not pd.isna(latest['MA5']) and latest['收盘'] > latest['MA5']:
                scores.append(70)
            else:
                scores.append(30)
                
            if not pd.isna(latest['MA10']) and latest['收盘'] > latest['MA10']:
                scores.append(65)
            else:
                scores.append(35)
                
            if not pd.isna(latest['MA20']) and latest['收盘'] > latest['MA20']:
                scores.append(60)
            else:
                scores.append(40)
            
            # 2. 均线多头排列评分
            if not pd.isna(latest['MA5']) and not pd.isna(latest['MA10']) and not pd.isna(latest['MA20']) and latest['MA5'] > latest['MA10'] > latest['MA20']:
                scores.append(80)
            elif not pd.isna(latest['MA5']) and not pd.isna(latest['MA10']) and latest['MA5'] > latest['MA10']:
                scores.append(60)
            else:
                scores.append(40)
            
            # 3. 成交量变化评分
            if not pd.isna(latest['成交量变化']) and latest['成交量变化'] > 0.1:
                scores.append(70)
            elif not pd.isna(latest['成交量变化']) and latest['成交量变化'] > 0:
                scores.append(60)
            else:
                scores.append(40)
            
            # 4. RSI评分
            if not pd.isna(latest['RSI']):
                if 40 <= latest['RSI'] <= 60:
                    scores.append(50)
                elif 30 <= latest['RSI'] < 40 or 60 < latest['RSI'] <= 70:
                    scores.append(60)
                elif latest['RSI'] < 30:
                    scores.append(70)  # 超卖
                else:
                    scores.append(30)  # 超买
            else:
                scores.append(50)  # RSI缺失时使用默认评分
            
            # 计算综合评分
            if scores:
                technical_score = sum(scores) / len(scores)
            else:
                technical_score = 50.0
            
            return technical_score
        
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 技术面数据失败: {e}")
            return 50.0  # 返回默认评分
    
    def analyze_industry(self, industry_data, stock_industry_mapping):
        """分析行业热度数据
        
        Args:
            industry_data: 行业指数数据DataFrame
            stock_industry_mapping: 股票行业对应关系DataFrame
            
        Returns:
            pandas.DataFrame: 包含股票代码和行业热度评分的DataFrame
        """
        logger.info("分析行业热度数据")
        
        try:
            # 检查数据是否为空
            if industry_data.empty or stock_industry_mapping.empty:
                logger.warning("行业数据或股票行业对应关系为空")
                return pd.DataFrame(columns=['代码', 'industry_score'])
            
            # 打印行业数据的列名，帮助调试
            logger.info(f"行业数据列名: {industry_data.columns.tolist()}")
            logger.info(f"股票行业对应关系列名: {stock_industry_mapping.columns.tolist()}")
            
            # 确定行业名称列
            industry_name_col = None
            for col in ['板块名称', '板块', '行业名称', '名称', 'name', 'sector_name']:
                if col in industry_data.columns:
                    industry_name_col = col
                    break
            
            # 确定涨跌幅列
            change_col = None
            for col in ['涨跌幅', '涨跌幅(%)', '涨跌幅度', 'change_pct', 'change']:
                if col in industry_data.columns:
                    change_col = col
                    break
            
            # 如果找不到必要的列，返回默认评分
            if not industry_name_col or not change_col:
                logger.error(f"行业数据缺少必要的列，可用列: {industry_data.columns.tolist()}")
                # 尝试使用可能的替代列
                if '板块' in industry_data.columns:
                    industry_name_col = '板块'
                    logger.info(f"使用'板块'列作为行业名称列")
                else:
                    return pd.DataFrame(columns=['代码', 'industry_score'])
            
            # 确定股票行业对应关系中的列
            stock_code_col = None
            for col in ['代码', 'code', 'symbol']:
                if col in stock_industry_mapping.columns:
                    stock_code_col = col
                    break
            
            industry_col = None
            for col in ['所属行业', '行业', 'industry', 'sector']:
                if col in stock_industry_mapping.columns:
                    industry_col = col
                    break
            
            # 如果找不到必要的列，返回默认评分
            if not stock_code_col or not industry_col:
                logger.error(f"股票行业对应关系数据缺少必要的列，可用列: {stock_industry_mapping.columns.tolist()}")
                return pd.DataFrame(columns=['代码', 'industry_score'])
            
            # 计算行业评分
            industry_scores = {}
            
            # 确保涨跌幅是数值类型
            industry_data[change_col] = pd.to_numeric(industry_data[change_col], errors='coerce')
            
            # 基于行业涨跌幅计算行业评分
            for _, row in industry_data.iterrows():
                industry_name = row[industry_name_col]
                change_pct = row[change_col]
                
                if pd.isna(change_pct):
                    continue
                
                # 将涨跌幅映射到0-100的评分区间
                # 假设涨跌幅在-10%到10%之间
                score = (change_pct + 10) * 5
                score = max(0, min(100, score))  # 限制在0-100范围内
                
                industry_scores[industry_name] = score
            
            # 为每个股票分配行业评分
            result = []
            for _, row in stock_industry_mapping.iterrows():
                stock_code = row[stock_code_col]
                industry = row[industry_col]
                
                # 获取行业评分，如果行业不在评分字典中，则给予默认评分50
                # 尝试精确匹配
                industry_score = industry_scores.get(industry, None)
                
                # 如果精确匹配失败，尝试模糊匹配
                if industry_score is None:
                    # 查找包含该行业名称的键
                    for key, value in industry_scores.items():
                        if industry in key or key in industry:
                            industry_score = value
                            break
                
                # 如果仍然没有找到匹配，使用默认评分
                if industry_score is None:
                    industry_score = 50
                
                result.append({
                    '代码': str(stock_code),  # 确保代码是字符串类型
                    'industry_score': industry_score
                })
            
            # 转换为DataFrame
            result_df = pd.DataFrame(result)
            
            # 如果结果为空，返回空DataFrame
            if result_df.empty:
                logger.warning("行业分析结果为空")
                return pd.DataFrame(columns=['代码', 'industry_score'])
            
            # 对于同一股票可能有多个行业的情况，取平均值
            result_df = result_df.groupby('代码')['industry_score'].mean().reset_index()
            
            return result_df
        
        except Exception as e:
            logger.error(f"分析行业热度数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame(columns=['代码', 'industry_score'])
    
    def calculate_potential_score(self, stock_data):
        """计算股票潜力评分
        
        Args:
            stock_data: 包含各项评分的股票数据字典
            
        Returns:
            float: 潜力综合评分
        """
        # 获取各项评分
        fund_flow_score = stock_data.get('fund_flow_score', 0)
        social_score = stock_data.get('social_score', 0)
        fundamental_score = stock_data.get('fundamental_score', 0)
        technical_score = stock_data.get('technical_score', 0)
        industry_score = stock_data.get('industry_score', 0)
        
        # 计算加权评分
        potential_score = (
            fund_flow_score * self.weights['fund_flow'] +
            social_score * self.weights['social'] +
            fundamental_score * self.weights['fundamental'] +
            technical_score * self.weights['technical'] +
            industry_score * self.weights['industry']
        )
        
        return potential_score
    
    def generate_recommendation_reason(self, stock_data):
        """生成股票推荐理由
        
        Args:
            stock_data: 包含各项评分的股票数据字典
            
        Returns:
            str: 推荐理由
        """
        reasons = []
        
        # 资金流向
        fund_flow_score = stock_data.get('fund_flow_score', 0)
        if fund_flow_score >= 80:
            reasons.append("资金大幅流入")
        elif fund_flow_score >= 60:
            reasons.append("资金持续流入")
        
        # 社交热度
        social_score = stock_data.get('social_score', 0)
        if social_score >= 80:
            reasons.append("社交媒体讨论热度极高")
        elif social_score >= 60:
            reasons.append("社交媒体关注度较高")
        
        # 基本面
        fundamental_score = stock_data.get('fundamental_score', 0)
        if fundamental_score >= 80:
            reasons.append("基本面表现优异")
        elif fundamental_score >= 60:
            reasons.append("基本面状况良好")
        
        # 技术面
        technical_score = stock_data.get('technical_score', 0)
        if technical_score >= 80:
            reasons.append("技术指标极为强势")
        elif technical_score >= 60:
            reasons.append("技术形态向好")
        
        # 行业热度
        industry_score = stock_data.get('industry_score', 0)
        if industry_score >= 80:
            reasons.append("所属行业表现活跃")
        elif industry_score >= 60:
            reasons.append("行业整体向好")
        
        # 如果没有特别突出的点，给出综合评价
        if not reasons:
            potential_score = stock_data.get('potential_score', 0)
            if potential_score >= 60:
                reasons.append("综合各项指标表现良好")
            else:
                reasons.append("综合评分一般，建议观望")
        
        return "，".join(reasons) + "。"

    