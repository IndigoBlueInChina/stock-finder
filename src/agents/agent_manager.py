#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent管理器模块：负责协调各个智能体
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import json
from pathlib import Path
import time
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

class AgentManager:
    """Agent管理器：负责协调各个智能体，完成股票潜力分析任务"""
    
    def __init__(self, data_manager, analyzer_manager):
        """初始化Agent管理器
        
        Args:
            data_manager: 数据管理器实例
            analyzer_manager: 分析器管理器实例
        """
        logger.info("初始化Agent管理器")
        
        self.data_manager = data_manager
        self.analyzer_manager = analyzer_manager
        
        # 初始化LLM
        self.llm = self._init_llm()
        
        # 初始化Agent
        self.data_collection_agent = self._init_data_collection_agent()
        self.analysis_agent = self._init_analysis_agent()
        self.recommendation_agent = self._init_recommendation_agent()
        
        # 结果缓存
        self.results_cache = Path("data/results_cache")
        self.results_cache.mkdir(exist_ok=True, parents=True)
    
    def _init_llm(self):
        """初始化大语言模型
        
        Returns:
            LLM实例
        """
        try:
            model_name = os.getenv("MODEL_NAME", "gpt-4")
            temperature = float(os.getenv("TEMPERATURE", "0.7"))
            
            logger.info(f"初始化LLM，模型: {model_name}, 温度: {temperature}")
            
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            return llm
        
        except Exception as e:
            logger.error(f"初始化LLM失败: {e}")
            # 返回一个模拟的LLM，避免程序崩溃
            return None
    
    def _init_data_collection_agent(self):
        """初始化数据收集Agent
        
        Returns:
            数据收集Agent实例
        """
        logger.info("初始化数据收集Agent")
        
        # 这里返回一个简单的函数而不是真正的Agent，因为我们已经有了数据管理器
        def collect_data(stock_list=None):
            """收集股票数据
            
            Args:
                stock_list: 股票代码列表，如果为None则获取所有股票
                
            Returns:
                dict: 包含各类数据的字典
            """
            logger.info("数据收集Agent开始工作")
            
            try:
                # 获取股票列表
                if stock_list is None:
                    all_stocks = self.data_manager.get_stock_list()
                    stock_list = all_stocks['代码'].tolist()
                
                # 获取资金流向数据
                fund_flow_data = self.data_manager.get_fund_flow_data()
                
                # 获取社交媒体讨论数据
                social_data = self.data_manager.get_social_discussion_data()
                
                # 获取基本面数据
                fundamental_data = self.data_manager.get_stock_fundamental_data(stock_list)
                
                # 获取行业数据
                industry_data = self.data_manager.get_industry_data()
                
                # 获取股票行业对应关系
                stock_industry_mapping = self.data_manager.get_stock_industry_mapping()
                
                return {
                    'stock_list': stock_list,
                    'fund_flow_data': fund_flow_data,
                    'social_data': social_data,
                    'fundamental_data': fundamental_data,
                    'industry_data': industry_data,
                    'stock_industry_mapping': stock_industry_mapping
                }
            
            except Exception as e:
                logger.error(f"数据收集Agent工作失败: {e}")
                return {}
        
        return collect_data
    
    def _init_analysis_agent(self):
        """初始化分析Agent
        
        Returns:
            分析Agent实例
        """
        logger.info("初始化分析Agent")
        
        # 这里返回一个简单的函数而不是真正的Agent，因为我们已经有了分析器管理器
        def analyze_data(data):
            """分析股票数据
            
            Args:
                data: 包含各类数据的字典
                
            Returns:
                list: 包含分析结果的列表
            """
            logger.info("分析Agent开始工作")
            
            try:
                # 获取数据
                stock_list = data.get('stock_list', [])
                fund_flow_data = data.get('fund_flow_data', pd.DataFrame())
                social_data = data.get('social_data', pd.DataFrame())
                fundamental_data = data.get('fundamental_data', pd.DataFrame())
                industry_data = data.get('industry_data', pd.DataFrame())
                stock_industry_mapping = data.get('stock_industry_mapping', pd.DataFrame())
                
                # 分析资金流向
                fund_flow_scores = self.analyzer_manager.analyze_fund_flow(fund_flow_data)
                
                # 分析社交媒体讨论热度
                social_scores = self.analyzer_manager.analyze_social_discussion(social_data)
                
                # 分析基本面
                fundamental_scores = self.analyzer_manager.analyze_fundamental(fundamental_data)
                
                # 分析行业热度
                industry_scores = self.analyzer_manager.analyze_industry(industry_data, stock_industry_mapping)
                
                # 合并各项评分
                # 首先创建一个包含所有股票的DataFrame
                all_stocks = pd.DataFrame({'代码': stock_list})
                
                # 与各项评分合并
                result = all_stocks.copy()
                result = pd.merge(result, fund_flow_scores, on='代码', how='left')
                result = pd.merge(result, social_scores, on='代码', how='left')
                result = pd.merge(result, fundamental_scores, on='代码', how='left')
                result = pd.merge(result, industry_scores, on='代码', how='left')
                
                # 填充缺失值
                result = result.fillna(50)  # 默认评分为50
                
                # 获取股票名称
                stock_info = self.data_manager.get_stock_list()
                result = pd.merge(result, stock_info, on='代码', how='left')
                
                # 转换为列表
                result_list = []
                for _, row in result.iterrows():
                    stock_code = row['代码']
                    
                    # 获取技术面评分
                    try:
                        technical_data = self.data_manager.get_stock_technical_data(stock_code)
                        technical_score = self.analyzer_manager.analyze_technical(stock_code, technical_data)
                    except Exception as e:
                        logger.warning(f"获取股票 {stock_code} 技术面评分失败: {e}")
                        technical_score = 50.0
                    
                    # 创建股票数据字典
                    stock_data = {
                        'code': stock_code,
                        'name': row.get('名称', '未知'),
                        'fund_flow_score': row.get('fund_flow_score', 50),
                        'social_score': row.get('social_score', 50),
                        'fundamental_score': row.get('fundamental_score', 50),
                        'technical_score': technical_score,
                        'industry_score': row.get('industry_score', 50)
                    }
                    
                    # 计算潜力评分
                    stock_data['potential_score'] = self.analyzer_manager.calculate_potential_score(stock_data)
                    
                    # 添加到结果列表
                    result_list.append(stock_data)
                
                return result_list
            
            except Exception as e:
                logger.error(f"分析Agent工作失败: {e}")
                return []
        
        return analyze_data
    
    def _init_recommendation_agent(self):
        """初始化推荐Agent
        
        Returns:
            推荐Agent实例
        """
        logger.info("初始化推荐Agent")
        
        # 这里我们使用LLM来生成推荐理由
        def recommend_stocks(analyzed_stocks, top_n=10, min_score=60):
            """推荐股票
            
            Args:
                analyzed_stocks: 分析后的股票列表
                top_n: 推荐的股票数量
                min_score: 最低潜力评分
                
            Returns:
                list: 推荐的股票列表
            """
            logger.info(f"推荐Agent开始工作，推荐数量: {top_n}, 最低评分: {min_score}")
            
            try:
                # 筛选评分达到最低要求的股票
                qualified_stocks = [s for s in analyzed_stocks if s['potential_score'] >= min_score]
                
                # 按潜力评分排序
                sorted_stocks = sorted(qualified_stocks, key=lambda x: x['potential_score'], reverse=True)
                
                # 取前N个
                top_stocks = sorted_stocks[:top_n]
                
                # 为每只股票生成推荐理由
                for stock in top_stocks:
                    stock['reason'] = self.analyzer_manager.generate_recommendation_reason(stock)
                    
                    # 如果有LLM，使用LLM增强推荐理由
                    if self.llm:
                        try:
                            enhanced_reason = self._enhance_recommendation_reason(stock)
                            if enhanced_reason:
                                stock['reason'] = enhanced_reason
                        except Exception as e:
                            logger.warning(f"使用LLM增强推荐理由失败: {e}")
                
                return top_stocks
            
            except Exception as e:
                logger.error(f"推荐Agent工作失败: {e}")
                return []
        
        return recommend_stocks
    
    def _enhance_recommendation_reason(self, stock_data):
        """使用LLM增强推荐理由
        
        Args:
            stock_data: 股票数据字典
            
        Returns:
            str: 增强后的推荐理由
        """
        if not self.llm:
            return None
        
        try:
            # 构建提示
            prompt = f"""
            你是一位专业的股票分析师，请根据以下数据为股票 {stock_data['name']}（{stock_data['code']}）生成一段专业、简洁的推荐理由：
            
            1. 资金流向评分: {stock_data['fund_flow_score']}/100
            2. 社交媒体热度评分: {stock_data['social_score']}/100
            3. 基本面评分: {stock_data['fundamental_score']}/100
            4. 技术面评分: {stock_data['technical_score']}/100
            5. 行业热度评分: {stock_data['industry_score']}/100
            6. 综合潜力评分: {stock_data['potential_score']}/100
            
            原始推荐理由: {stock_data['reason']}
            
            请生成一段更加专业、具体的推荐理由，不超过100字。
            """
            
            # 调用LLM
            messages = [
                SystemMessage(content="你是一位专业的股票分析师，擅长简洁、专业地分析股票潜力。"),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.generate([messages])
            enhanced_reason = response.generations[0][0].text.strip()
            
            return enhanced_reason
        
        except Exception as e:
            logger.warning(f"增强推荐理由失败: {e}")
            return None
    
    def run_analysis_pipeline(self, stock_list=None, top_n=10, min_score=60):
        """运行完整的分析流程
        
        Args:
            stock_list: 股票代码列表，如果为None则分析所有股票
            top_n: 推荐的股票数量
            min_score: 最低潜力评分
            
        Returns:
            list: 推荐的股票列表
        """
        logger.info(f"开始运行分析流程，股票数量: {len(stock_list) if stock_list else '全部'}, 推荐数量: {top_n}")
        
        # 1. 收集数据
        data = self.data_collection_agent(stock_list)
        
        # 2. 分析数据
        analyzed_stocks = self.analysis_agent(data)
        
        # 3. 推荐股票
        recommended_stocks = self.recommendation_agent(analyzed_stocks, top_n, min_score)
        
        # 4. 缓存结果
        self._cache_results(recommended_stocks)
        
        return recommended_stocks
    
    def _cache_results(self, results):
        """缓存分析结果
        
        Args:
            results: 分析结果列表
        """
        try:
            # 创建时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存结果
            cache_file = self.results_cache / f"results_{timestamp}.json"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"结果已缓存到: {cache_file}")
        
        except Exception as e:
            logger.error(f"缓存结果失败: {e}")
    
    def get_cached_results(self, latest=True):
        """获取缓存的结果
        
        Args:
            latest: 是否只获取最新的结果
            
        Returns:
            list: 缓存的结果列表
        """
        try:
            cache_files = list(self.results_cache.glob("results_*.json"))
            
            if not cache_files:
                return []
            
            if latest:
                # 获取最新的缓存文件
                latest_file = max(cache_files, key=lambda x: x.stat().st_mtime)
                
                with open(latest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 获取所有缓存文件
                all_results = []
                for file in cache_files:
                    with open(file, 'r', encoding='utf-8') as f:
                        all_results.append({
                            'timestamp': file.stem.replace('results_', ''),
                            'results': json.load(f)
                        })
                
                return all_results
        
        except Exception as e:
            logger.error(f"获取缓存结果失败: {e}")
            return [] 