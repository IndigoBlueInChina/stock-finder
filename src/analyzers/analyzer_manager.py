#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析器管理模块：负责对各类数据进行分析
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

logger = logging.getLogger(__name__)

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
            # 复制数据，避免修改原始数据
            df = fund_flow_data.copy()
            
            # 确保必要的列存在
            required_columns = ['代码', '主力净流入-净额', '超大单净流入-净额', '大单净流入-净额']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"资金流向数据缺少必要的列: {col}")
                    return pd.DataFrame(columns=['代码', 'fund_flow_score'])
            
            # 计算资金流向评分
            # 1. 主力净流入评分
            scaler = MinMaxScaler(feature_range=(0, 100))
            
            # 处理主力净流入-净额
            if '主力净流入-净额' in df.columns:
                df['主力净流入评分'] = scaler.fit_transform(df['主力净流入-净额'].values.reshape(-1, 1)).flatten()
            else:
                df['主力净流入评分'] = 0
            
            # 处理超大单净流入-净额
            if '超大单净流入-净额' in df.columns:
                df['超大单净流入评分'] = scaler.fit_transform(df['超大单净流入-净额'].values.reshape(-1, 1)).flatten()
            else:
                df['超大单净流入评分'] = 0
            
            # 处理大单净流入-净额
            if '大单净流入-净额' in df.columns:
                df['大单净流入评分'] = scaler.fit_transform(df['大单净流入-净额'].values.reshape(-1, 1)).flatten()
            else:
                df['大单净流入评分'] = 0
            
            # 综合评分：主力净流入(50%) + 超大单净流入(30%) + 大单净流入(20%)
            df['fund_flow_score'] = (df['主力净流入评分'] * 0.5 + 
                                     df['超大单净流入评分'] * 0.3 + 
                                     df['大单净流入评分'] * 0.2)
            
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
            # 复制数据，避免修改原始数据
            df = social_data.copy()
            
            # 确保必要的列存在
            if '代码' not in df.columns or '讨论数量' not in df.columns:
                logger.error("社交媒体讨论数据缺少必要的列")
                return pd.DataFrame(columns=['代码', 'social_score'])
            
            # 计算社交热度评分
            scaler = MinMaxScaler(feature_range=(0, 100))
            df['social_score'] = scaler.fit_transform(df['讨论数量'].values.reshape(-1, 1)).flatten()
            
            # 返回结果
            return df[['代码', 'social_score']]
        
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
            # 复制数据，避免修改原始数据
            df = technical_data.copy()
            
            # 确保必要的列存在
            required_columns = ['日期', '收盘', '开盘', '最高', '最低', '成交量']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"技术面数据缺少必要的列: {col}")
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
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 获取最近的数据进行评分
            latest = df.iloc[-1]
            
            # 评分指标
            scores = []
            
            # 1. 价格位于均线之上评分
            if latest['收盘'] > latest['MA5']:
                scores.append(70)
            else:
                scores.append(30)
                
            if latest['收盘'] > latest['MA10']:
                scores.append(65)
            else:
                scores.append(35)
                
            if latest['收盘'] > latest['MA20']:
                scores.append(60)
            else:
                scores.append(40)
            
            # 2. 均线多头排列评分
            if latest['MA5'] > latest['MA10'] > latest['MA20']:
                scores.append(80)
            elif latest['MA5'] > latest['MA10']:
                scores.append(60)
            else:
                scores.append(40)
            
            # 3. 成交量变化评分
            if latest['成交量变化'] > 0.1:
                scores.append(70)
            elif latest['成交量变化'] > 0:
                scores.append(60)
            else:
                scores.append(40)
            
            # 4. RSI评分
            if 40 <= latest['RSI'] <= 60:
                scores.append(50)
            elif 30 <= latest['RSI'] < 40 or 60 < latest['RSI'] <= 70:
                scores.append(60)
            elif latest['RSI'] < 30:
                scores.append(70)  # 超卖
            else:
                scores.append(30)  # 超买
            
            # 计算综合评分
            technical_score = sum(scores) / len(scores)
            
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
            # 确保必要的列存在
            if '板块名称' not in industry_data.columns or '涨跌幅' not in industry_data.columns:
                logger.error("行业数据缺少必要的列")
                return pd.DataFrame(columns=['代码', 'industry_score'])
                
            if '代码' not in stock_industry_mapping.columns or '所属行业' not in stock_industry_mapping.columns:
                logger.error("股票行业对应关系数据缺少必要的列")
                return pd.DataFrame(columns=['代码', 'industry_score'])
            
            # 计算行业评分
            industry_scores = {}
            
            # 基于行业涨跌幅计算行业评分
            for _, row in industry_data.iterrows():
                industry_name = row['板块名称']
                change_pct = row['涨跌幅']
                
                # 将涨跌幅映射到0-100的评分区间
                # 假设涨跌幅在-10%到10%之间
                score = (change_pct + 10) * 5
                score = max(0, min(100, score))  # 限制在0-100范围内
                
                industry_scores[industry_name] = score
            
            # 为每个股票分配行业评分
            result = []
            for _, row in stock_industry_mapping.iterrows():
                stock_code = row['代码']
                industry = row['所属行业']
                
                # 获取行业评分，如果行业不在评分字典中，则给予默认评分50
                industry_score = industry_scores.get(industry, 50)
                
                result.append({
                    '代码': stock_code,
                    'industry_score': industry_score
                })
            
            return pd.DataFrame(result)
        
        except Exception as e:
            logger.error(f"分析行业热度数据失败: {e}")
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