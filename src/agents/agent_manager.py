#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent管理器模块：负责协调各个智能体
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional
import json
from pathlib import Path
import time
from dotenv import load_dotenv
from loguru import logger

# 更新LangChain导入
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# 使用langchain包中的模块
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory

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
        
        # 进度回调函数
        self.progress_callback = None
        
        # 保存所有分析过的股票数据
        self.all_analyzed_stocks = []
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """设置进度回调函数
        
        Args:
            callback: 回调函数，接收当前进度、总进度和当前处理的股票代码
        """
        self.progress_callback = callback
    
    def _init_llm(self):
        """初始化大语言模型
        
        Returns:
            LLM实例
        """
        try:
            # 从.env文件获取配置
            model_name = os.getenv("MODEL_NAME", "qwen-plus")
            temperature = float(os.getenv("TEMPERATURE", "0.7"))
            api_key = os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("OPENAI_API_BASE")  # OpenAI兼容API基础URL
            
            # 检查API密钥是否设置
            if not api_key or api_key == "your_openai_api_key":
                logger.warning("未设置有效的OpenAI API密钥，LLM功能将被禁用")
                return None
                
            # 检查API基础URL是否设置
            if not api_base:
                logger.warning("未设置OpenAI API基础URL，将使用默认URL")
            
            logger.info(f"初始化LLM，模型: {model_name}, 温度: {temperature}")
            logger.info(f"使用API基础URL: {api_base}")
            
            # 使用OpenAI兼容模式配置
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                api_key=api_key,
                base_url=api_base,  # 使用自定义的API基础URL
                max_tokens=int(os.getenv("MAX_TOKENS", "4096"))
            )
            
            return llm
        
        except Exception as e:
            logger.error(f"初始化LLM失败: {e}")
            # 返回None，禁用LLM功能
            return None
    
    def _init_data_collection_agent(self):
        """初始化数据收集Agent
        
        Returns:
            数据收集Agent实例
        """
        logger.info("初始化数据收集Agent")
        
        # 返回self.collect_data方法
        return self.collect_data
    
    def collect_data(self, stock_list=None):
        """收集股票数据
        
        Args:
            stock_list: 股票代码列表，如果为None则获取所有股票
            
        Returns:
            dict: 包含各类数据的字典
        """
        logger.info("数据收集Agent开始工作")
        
        try:
            # 获取股票列表
            all_stocks = None
            if stock_list is None:
                all_stocks = self.data_manager.get_stock_list()
                # 确保代码列是字符串类型
                if not all_stocks.empty:
                    all_stocks = all_stocks.copy()
                    # 检查列名，确保使用正确的列名
                    code_column = '代码' if '代码' in all_stocks.columns else 'code'
                    if code_column in all_stocks.columns:
                        all_stocks[code_column] = all_stocks[code_column].astype(str)
                        stock_list = all_stocks[code_column].tolist()
                    else:
                        logger.warning(f"股票列表中没有代码列 (既没有 '代码' 也没有 'code')")
                        stock_list = []
            
            # 收集各类数据
            fund_flow_data = self.data_manager.get_fund_flow_data()
            if not fund_flow_data.empty:
                fund_flow_data = fund_flow_data.copy()
                # 检查列名
                code_column = '代码' if '代码' in fund_flow_data.columns else 'code'
                if code_column in fund_flow_data.columns:
                    fund_flow_data[code_column] = fund_flow_data[code_column].astype(str)
                
            social_data = self.data_manager.get_social_discussion_data()
            if not social_data.empty:
                social_data = social_data.copy()
                # 检查列名
                code_column = '代码' if '代码' in social_data.columns else 'code'
                if code_column in social_data.columns:
                    social_data[code_column] = social_data[code_column].astype(str)
                
            industry_data = self.data_manager.get_industry_data()
            
            stock_industry_mapping = self.data_manager.get_stock_industry_mapping()
            if not stock_industry_mapping.empty:
                stock_industry_mapping = stock_industry_mapping.copy()
                # 检查列名
                code_column = '代码' if '代码' in stock_industry_mapping.columns else 'code'
                if code_column in stock_industry_mapping.columns:
                    stock_industry_mapping[code_column] = stock_industry_mapping[code_column].astype(str)
            
            collected_data = {
                'stock_list': all_stocks,
                'fund_flow_data': fund_flow_data,
                'social_data': social_data,
                'industry_data': industry_data,
                'stock_industry_mapping': stock_industry_mapping
            }
            
            logger.info("数据收集完成")
            return collected_data
            
        except Exception as e:
            logger.error(f"数据收集Agent工作失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
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
                
                # 确保DataFrame有正确的列名
                # 检查并标准化fund_flow_data的列名
                if not fund_flow_data.empty and '代码' not in fund_flow_data.columns and 'code' in fund_flow_data.columns:
                    fund_flow_data = fund_flow_data.rename(columns={'code': '代码'})
                
                # 检查并标准化social_data的列名
                if not social_data.empty and '代码' not in social_data.columns and 'code' in social_data.columns:
                    social_data = social_data.rename(columns={'code': '代码'})
                
                # 检查并标准化fundamental_data的列名
                if not fundamental_data.empty and '代码' not in fundamental_data.columns and 'code' in fundamental_data.columns:
                    fundamental_data = fundamental_data.rename(columns={'code': '代码'})
                
                # 检查并标准化stock_industry_mapping的列名
                if not stock_industry_mapping.empty and '代码' not in stock_industry_mapping.columns and 'code' in stock_industry_mapping.columns:
                    stock_industry_mapping = stock_industry_mapping.rename(columns={'code': '代码'})
                
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
                # 检查列名
                name_column = '名称' if '名称' in stock_info.columns else 'name'
                code_column = '代码' if '代码' in stock_info.columns else 'code'
                
                # 重命名列以确保匹配
                stock_info_renamed = stock_info.rename(columns={
                    code_column: '代码',
                    name_column: '名称'
                })
                
                result = pd.merge(result, stock_info_renamed[['代码', '名称']], on='代码', how='left')
                
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
        def recommend_stocks(results):
            """推荐股票
            
            Args:
                results: 分析结果列表
                
            Returns:
                list: 包含推荐股票信息的列表
            """
            logger.info(f"推荐Agent开始工作，推荐数量: {len(results)}, 最低评分: {min([r['total_score'] for r in results]) if results else 0}")
            
            try:
                # 如果没有配置OpenAI API，跳过生成推荐理由
                if not self.llm:
                    logger.warning("未配置OpenAI API，跳过生成推荐理由")
                    return results
                
                # 为每只股票生成推荐理由
                for stock in results:
                    try:
                        # 准备提示词
                        prompt = self._prepare_stock_prompt(stock)
                        
                        # 调用LLM生成推荐理由
                        reason = self._generate_recommendation_reason(prompt)
                        
                        # 添加到结果中
                        stock['reason'] = reason
                        
                    except Exception as e:
                        logger.error(f"为股票 {stock['code']} 生成推荐理由失败: {e}")
                        stock['reason'] = "无法生成推荐理由"
                
                return results
            
            except Exception as e:
                logger.error(f"推荐Agent工作失败: {e}")
                return results
        
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
            
            # 使用新版API，添加超时处理
            import time
            from requests.exceptions import Timeout, RequestException
            
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # 设置超时时间
                    response = self.llm.invoke(messages, timeout=30)
                    enhanced_reason = response.content
                    return enhanced_reason
                except Timeout:
                    logger.warning(f"LLM请求超时，尝试重试 {attempt+1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        logger.error("LLM请求多次超时，使用原始推荐理由")
                        return stock_data['reason']
                except RequestException as e:
                    logger.error(f"LLM请求错误: {e}")
                    return stock_data['reason']
                except Exception as e:
                    logger.error(f"增强推荐理由失败: {e}")
                    return stock_data['reason']
            
            return stock_data['reason']  # 如果所有尝试都失败，返回原始理由
        
        except Exception as e:
            logger.warning(f"增强推荐理由失败: {e}")
            return stock_data['reason']  # 返回原始推荐理由
    
    def get_candidate_stocks(self, data, fund_flow_scores, social_scores, max_stocks_to_process=10):
        """获取候选股票列表
        
        Args:
            data: 收集的数据字典
            fund_flow_scores: 资金流向评分DataFrame
            social_scores: 社交热度评分DataFrame
            max_stocks_to_process: 最大处理的股票数量
            
        Returns:
            list: 候选股票列表，每个元素是包含代码和名称的字典
        """
        try:
            # 合并初步评分，筛选出有热度的股票
            initial_scores = pd.DataFrame()
            
            # 确保代码列都是字符串类型
            if not fund_flow_scores.empty and '代码' in fund_flow_scores.columns:
                fund_flow_scores = fund_flow_scores.copy()
                fund_flow_scores['代码'] = fund_flow_scores['代码'].astype(str)
            
            if not social_scores.empty and '代码' in social_scores.columns:
                social_scores = social_scores.copy()
                social_scores['代码'] = social_scores['代码'].astype(str)
            
            # 检查资金流向数据是否有效
            valid_fund_flow = not fund_flow_scores.empty and 'fund_flow_score' in fund_flow_scores.columns
            if valid_fund_flow:
                # 检查资金流向评分是否有变化（不全是默认值）
                if fund_flow_scores['fund_flow_score'].std() < 0.001:
                    logger.warning("资金流向评分全部相同，可能是默认值，不作为选股依据")
                    valid_fund_flow = False
            
            # 检查社交热度数据是否有效
            valid_social = not social_scores.empty and 'social_score' in social_scores.columns
            if valid_social:
                # 检查社交热度评分是否有变化（不全是默认值）
                if social_scores['social_score'].std() < 0.001:
                    logger.warning("社交热度评分全部相同，可能是默认值，不作为选股依据")
                    valid_social = False
            
            # 根据有效数据构建初始评分
            if valid_fund_flow and valid_social:
                # 两种数据都有效，合并它们
                logger.info("使用资金流向和社交热度数据选股")
                initial_scores = pd.merge(fund_flow_scores, social_scores, on='代码', how='outer')
            elif valid_fund_flow:
                # 只有资金流向数据有效
                logger.info("仅使用资金流向数据选股")
                initial_scores = fund_flow_scores.copy()
                # 添加默认的社交热度评分
                initial_scores['social_score'] = 50
            elif valid_social:
                # 只有社交热度数据有效
                logger.info("仅使用社交热度数据选股")
                initial_scores = social_scores.copy()
                # 添加默认的资金流向评分
                initial_scores['fund_flow_score'] = 50
            else:
                # 两种数据都无效，使用随机选择
                logger.warning("资金流向和社交热度数据都无效，使用随机选择")
                stock_list = data.get('stock_list')
                if stock_list is not None and not stock_list.empty:
                    # 检查列名，确保使用正确的列名
                    code_column = '代码' if '代码' in stock_list.columns else 'code'
                    if code_column in stock_list.columns:
                        # 确保代码列是字符串类型
                        stock_list = stock_list.copy()
                        stock_list[code_column] = stock_list[code_column].astype(str)
                        
                        # 创建初始评分DataFrame
                        initial_scores = pd.DataFrame({
                            '代码': stock_list[code_column],
                            'fund_flow_score': np.random.normal(50, 15, size=len(stock_list)),
                            'social_score': np.random.normal(50, 15, size=len(stock_list))
                        })
                        
                        # 限制评分范围在0-100之间
                        initial_scores['fund_flow_score'] = np.clip(initial_scores['fund_flow_score'], 0, 100)
                        initial_scores['social_score'] = np.clip(initial_scores['social_score'], 0, 100)
            
            # 计算初步评分（资金流向和社交热度的加权平均值）
            candidate_stocks = None
            selection_reason = "基于综合热度评分"
            
            if not initial_scores.empty:
                # 填充缺失值
                if 'fund_flow_score' in initial_scores.columns:
                    initial_scores['fund_flow_score'] = initial_scores['fund_flow_score'].fillna(50)
                else:
                    initial_scores['fund_flow_score'] = 50
                
                if 'social_score' in initial_scores.columns:
                    initial_scores['social_score'] = initial_scores['social_score'].fillna(50)
                else:
                    initial_scores['social_score'] = 50
                
                # 计算加权平均
                fund_flow_weight = self.analyzer_manager.weights['fund_flow']
                social_weight = self.analyzer_manager.weights['social']
                total_weight = fund_flow_weight + social_weight
                
                initial_scores['initial_score'] = (
                    initial_scores['fund_flow_score'] * fund_flow_weight + 
                    initial_scores['social_score'] * social_weight
                ) / total_weight
                
                # 为每只股票生成候选理由
                initial_scores['selection_reason'] = initial_scores.apply(
                    lambda row: self._generate_selection_reason(row), axis=1
                )
                
                # 记录评分分布情况
                logger.info(f"资金流向评分统计: 最小={initial_scores['fund_flow_score'].min():.2f}, 最大={initial_scores['fund_flow_score'].max():.2f}, 平均={initial_scores['fund_flow_score'].mean():.2f}, 标准差={initial_scores['fund_flow_score'].std():.2f}")
                logger.info(f"社交热度评分统计: 最小={initial_scores['social_score'].min():.2f}, 最大={initial_scores['social_score'].max():.2f}, 平均={initial_scores['social_score'].mean():.2f}, 标准差={initial_scores['social_score'].std():.2f}")
                logger.info(f"综合评分统计: 最小={initial_scores['initial_score'].min():.2f}, 最大={initial_scores['initial_score'].max():.2f}, 平均={initial_scores['initial_score'].mean():.2f}, 标准差={initial_scores['initial_score'].std():.2f}")
                
                # 筛选出初步评分较高的股票（例如前50名）作为候选
                candidate_stocks = initial_scores.sort_values('initial_score', ascending=False).head(50)
                candidate_stock_codes = candidate_stocks['代码'].tolist()
                
                # 记录选出的候选股票信息
                top_candidates = initial_scores.sort_values('initial_score', ascending=False).head(10)
                logger.info(f"初步筛选出的前10只股票:\n{top_candidates[['代码', 'fund_flow_score', 'social_score', 'initial_score', 'selection_reason']]}")
                logger.info(f"初步筛选出 {len(candidate_stock_codes)} 只有热度的股票进行深入分析")
                logger.info(f"初步筛选出的股票代码: {candidate_stock_codes}")
            else:
                logger.warning("初步评分为空，将分析所有股票")
                stock_list = data.get('stock_list')
                if stock_list is not None and not stock_list.empty:
                    # 检查列名，确保使用正确的列名
                    code_column = '代码' if '代码' in stock_list.columns else 'code'
                    if code_column in stock_list.columns:
                        # 确保代码列是字符串类型
                        stock_list = stock_list.copy()
                        stock_list[code_column] = stock_list[code_column].astype(str)
                        candidate_stock_codes = stock_list[code_column].tolist()
                        selection_reason = "随机选择"
                    else:
                        logger.error(f"股票列表中没有代码列 (既没有 '代码' 也没有 'code')")
                        return []
                else:
                    logger.error("无法获取股票列表")
                    return []
                
                if len(candidate_stock_codes) > 50:
                    logger.warning(f"股票数量过多 ({len(candidate_stock_codes)})，将限制为前50只")
                    candidate_stock_codes = candidate_stock_codes[:50]
            
            # 限制处理的股票数量，避免API调用过多
            if len(candidate_stock_codes) > max_stocks_to_process:
                logger.info(f"限制处理的股票数量为 {max_stocks_to_process} 只，避免API调用频率限制导致分析时间过长")
                candidate_stock_codes = candidate_stock_codes[:max_stocks_to_process]
            
            # 获取股票名称
            stock_list = data.get('stock_list')
            result = []
            
            if stock_list is not None and not stock_list.empty:
                # 检查列名，确保使用正确的列名
                code_column = '代码' if '代码' in stock_list.columns else 'code'
                name_column = '名称' if '名称' in stock_list.columns else 'name'
                
                if code_column in stock_list.columns:
                    # 确保代码列是字符串类型
                    stock_list = stock_list.copy()
                    stock_list[code_column] = stock_list[code_column].astype(str)
                    
                    # 为每个候选股票代码获取名称和候选理由
                    for code in candidate_stock_codes:
                        stock_info = stock_list[stock_list[code_column] == code]
                        name = stock_info[name_column].values[0] if not stock_info.empty and name_column in stock_info.columns else "未知"
                        
                        # 获取候选理由
                        reason = selection_reason
                        if candidate_stocks is not None and '代码' in candidate_stocks.columns and 'selection_reason' in candidate_stocks.columns:
                            stock_row = candidate_stocks[candidate_stocks['代码'] == code]
                            if not stock_row.empty:
                                reason = stock_row['selection_reason'].values[0]
                        
                        result.append({
                            'code': code,
                            'name': name,
                            'selection_reason': reason
                        })
                else:
                    # 如果没有代码列，只返回代码
                    logger.warning(f"股票列表中没有代码列 (既没有 '{code_column}' 也没有 'code')")
                    result = [{'code': code, 'name': "未知", 'selection_reason': selection_reason} for code in candidate_stock_codes]
            else:
                # 如果没有股票列表，只返回代码
                result = [{'code': code, 'name': "未知", 'selection_reason': selection_reason} for code in candidate_stock_codes]
            
            return result
            
        except Exception as e:
            logger.error(f"获取候选股票列表失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _generate_selection_reason(self, row):
        """为股票生成候选理由
        
        Args:
            row: DataFrame中的一行数据，包含fund_flow_score和social_score
            
        Returns:
            str: 候选理由
        """
        fund_flow_score = row.get('fund_flow_score', 0)
        social_score = row.get('social_score', 0)
        
        reasons = []
        
        if fund_flow_score >= 70:
            reasons.append("资金流入强劲")
        elif fund_flow_score >= 60:
            reasons.append("资金流入较好")
        
        if social_score >= 70:
            reasons.append("社交热度高")
        elif social_score >= 60:
            reasons.append("社交热度较高")
        
        if not reasons:
            if fund_flow_score > social_score:
                reasons.append("资金流向相对较好")
            else:
                reasons.append("社交讨论相对活跃")
        
        return "、".join(reasons)
    
    def run_analysis_pipeline(self, top_n=10, min_score=60, max_stocks_to_process=10, candidate_stocks=None):
        """运行完整的分析流程
        
        Args:
            top_n: 推荐的股票数量
            min_score: 最低推荐评分
            max_stocks_to_process: 最大处理的股票数量，避免API调用过多，默认10只
            candidate_stocks: 预先筛选的候选股票列表，如果为None则重新筛选
            
        Returns:
            list: 包含推荐股票信息的列表
        """
        try:
            # 收集数据
            data = self.collect_data()
            
            # 如果没有提供候选股票列表，则筛选候选股票
            if candidate_stocks is None:
                # 先分析资金流向和社交热度数据，这两个不需要逐只股票API调用
                logger.info("分析资金流向数据...")
                fund_flow_scores = self.analyzer_manager.analyze_fund_flow(data.get('fund_flow_data', pd.DataFrame()))
                
                # 确保代码列是字符串类型
                if not fund_flow_scores.empty and '代码' in fund_flow_scores.columns:
                    fund_flow_scores = fund_flow_scores.copy()
                    fund_flow_scores['代码'] = fund_flow_scores['代码'].astype(str)
                
                logger.info("分析社交热度数据...")
                social_scores = self.analyzer_manager.analyze_social_discussion(data.get('social_data', pd.DataFrame()))
                
                # 确保代码列是字符串类型
                if not social_scores.empty and '代码' in social_scores.columns:
                    social_scores = social_scores.copy()
                    social_scores['代码'] = social_scores['代码'].astype(str)
                
                # 获取候选股票列表
                candidate_stocks_info = self.get_candidate_stocks(data, fund_flow_scores, social_scores, max_stocks_to_process)
                candidate_stock_codes = [stock['code'] for stock in candidate_stocks_info]
            else:
                # 使用提供的候选股票列表
                candidate_stock_codes = [stock['code'] for stock in candidate_stocks]
                
                # 先分析资金流向和社交热度数据，这两个不需要逐只股票API调用
                logger.info("分析资金流向数据...")
                fund_flow_scores = self.analyzer_manager.analyze_fund_flow(data.get('fund_flow_data', pd.DataFrame()))
                
                # 确保代码列是字符串类型
                if not fund_flow_scores.empty and '代码' in fund_flow_scores.columns:
                    fund_flow_scores = fund_flow_scores.copy()
                    fund_flow_scores['代码'] = fund_flow_scores['代码'].astype(str)
                
                logger.info("分析社交热度数据...")
                social_scores = self.analyzer_manager.analyze_social_discussion(data.get('social_data', pd.DataFrame()))
                
                # 确保代码列是字符串类型
                if not social_scores.empty and '代码' in social_scores.columns:
                    social_scores = social_scores.copy()
                    social_scores['代码'] = social_scores['代码'].astype(str)
            
            # 只对候选股票进行详细分析
            logger.info("开始对候选股票进行详细分析...")
            results = []
            
            # 获取基本面数据（这个通常是批量获取的，不需要逐只股票调用API）
            logger.info("获取基本面数据...")
            fundamental_data = self.data_manager.get_stock_fundamental_data(candidate_stock_codes)
            
            # 确保代码列是字符串类型
            if not fundamental_data.empty:
                fundamental_data = fundamental_data.copy()
                # 检查列名
                code_column = '代码' if '代码' in fundamental_data.columns else 'code'
                if code_column in fundamental_data.columns:
                    fundamental_data[code_column] = fundamental_data[code_column].astype(str)
            
            logger.info("分析基本面数据...")
            fundamental_scores = self.analyzer_manager.analyze_fundamental(fundamental_data)
            
            # 确保代码列是字符串类型
            if not fundamental_scores.empty:
                fundamental_scores = fundamental_scores.copy()
                # 检查列名
                code_column = '代码' if '代码' in fundamental_scores.columns else 'code'
                if code_column in fundamental_scores.columns:
                    fundamental_scores[code_column] = fundamental_scores[code_column].astype(str)
            
            # 获取行业数据
            logger.info("获取行业数据...")
            industry_data = data.get('industry_data', pd.DataFrame())
            
            # 只获取候选股票的行业映射数据
            stock_industry_mapping = data.get('stock_industry_mapping', pd.DataFrame())
            if not stock_industry_mapping.empty:
                stock_industry_mapping = stock_industry_mapping.copy()
                # 检查列名
                code_column = '代码' if '代码' in stock_industry_mapping.columns else 'code'
                if code_column in stock_industry_mapping.columns:
                    stock_industry_mapping[code_column] = stock_industry_mapping[code_column].astype(str)
                    # 只保留候选股票的行业映射
                    stock_industry_mapping = stock_industry_mapping[stock_industry_mapping[code_column].isin(candidate_stock_codes)]
            
            logger.info("分析行业数据...")
            industry_scores = self.analyzer_manager.analyze_industry(industry_data, stock_industry_mapping)
            
            # 确保代码列是字符串类型
            if not industry_scores.empty:
                industry_scores = industry_scores.copy()
                # 检查列名
                code_column = '代码' if '代码' in industry_scores.columns else 'code'
                if code_column in industry_scores.columns:
                    industry_scores[code_column] = industry_scores[code_column].astype(str)
            
            # 逐只分析候选股票的技术面
            total_stocks = len(candidate_stock_codes)
            success_count = 0
            error_count = 0
            
            logger.info(f"开始逐只分析 {total_stocks} 只股票的技术面数据...")
            
            # 清空之前的分析结果
            self.all_analyzed_stocks = []
            
            # 获取股票名称和当前价格的映射表
            stock_names = {}
            stock_prices = {}
            
            # 从股票列表中获取名称
            if 'stock_list' in data and not data['stock_list'].empty:
                stock_list = data['stock_list'].copy()
                # 检查列名
                code_column = '代码' if '代码' in stock_list.columns else 'code'
                name_column = '名称' if '名称' in stock_list.columns else 'name'
                
                if code_column in stock_list.columns and name_column in stock_list.columns:
                    stock_list[code_column] = stock_list[code_column].astype(str)
                    for _, row in stock_list.iterrows():
                        stock_names[row[code_column]] = row[name_column]
            
            # 从基本面数据中获取当前价格
            if not fundamental_data.empty:
                # 检查列名
                code_column = '代码' if '代码' in fundamental_data.columns else 'code'
                price_column = '最新价' if '最新价' in fundamental_data.columns else 'close'
                
                if code_column in fundamental_data.columns and price_column in fundamental_data.columns:
                    for _, row in fundamental_data.iterrows():
                        try:
                            stock_prices[row[code_column]] = float(row[price_column])
                        except (ValueError, TypeError):
                            stock_prices[row[code_column]] = 0.0
            
            for i, stock_code in enumerate(candidate_stock_codes):
                try:
                    # 记录进度
                    logger.info(f"正在分析第 {i+1}/{total_stocks} 只股票: {stock_code} (成功: {success_count}, 失败: {error_count})")
                    
                    # 如果设置了进度回调函数，调用它
                    if self.progress_callback:
                        self.progress_callback(i+1, total_stocks, stock_code, "分析中")
                    
                    # 确保股票代码是字符串类型
                    stock_code = str(stock_code)
                    
                    # 获取并分析技术面数据
                    technical_data = self.data_manager.get_stock_technical_data(stock_code)
                    technical_score = self.analyzer_manager.analyze_technical(stock_code, technical_data)
                    
                    # 合并该股票的所有评分
                    stock_scores = {}
                    
                    # 添加资金流向评分
                    if not fund_flow_scores.empty:
                        # 检查列名
                        code_column = '代码' if '代码' in fund_flow_scores.columns else 'code'
                        if code_column in fund_flow_scores.columns and stock_code in fund_flow_scores[code_column].values:
                            stock_fund_flow = fund_flow_scores[fund_flow_scores[code_column] == stock_code]
                            if not stock_fund_flow.empty and 'fund_flow_score' in stock_fund_flow.columns:
                                stock_scores['fund_flow_score'] = float(stock_fund_flow['fund_flow_score'].values[0])
                            else:
                                stock_scores['fund_flow_score'] = 50
                        else:
                            stock_scores['fund_flow_score'] = 50
                    else:
                        stock_scores['fund_flow_score'] = 50
                    
                    # 添加社交热度评分
                    if not social_scores.empty:
                        # 检查列名
                        code_column = '代码' if '代码' in social_scores.columns else 'code'
                        if code_column in social_scores.columns and stock_code in social_scores[code_column].values:
                            stock_social = social_scores[social_scores[code_column] == stock_code]
                            if not stock_social.empty and 'social_score' in stock_social.columns:
                                stock_scores['social_score'] = float(stock_social['social_score'].values[0])
                            else:
                                stock_scores['social_score'] = 50
                        else:
                            stock_scores['social_score'] = 50
                    else:
                        stock_scores['social_score'] = 50
                    
                    # 添加基本面评分
                    if not fundamental_scores.empty:
                        # 检查列名
                        code_column = '代码' if '代码' in fundamental_scores.columns else 'code'
                        if code_column in fundamental_scores.columns and stock_code in fundamental_scores[code_column].values:
                            stock_fundamental = fundamental_scores[fundamental_scores[code_column] == stock_code]
                            if not stock_fundamental.empty and 'fundamental_score' in stock_fundamental.columns:
                                stock_scores['fundamental_score'] = float(stock_fundamental['fundamental_score'].values[0])
                            else:
                                stock_scores['fundamental_score'] = 50
                        else:
                            stock_scores['fundamental_score'] = 50
                    else:
                        stock_scores['fundamental_score'] = 50
                    
                    # 添加行业评分
                    if not industry_scores.empty:
                        # 检查列名
                        code_column = '代码' if '代码' in industry_scores.columns else 'code'
                        if code_column in industry_scores.columns and stock_code in industry_scores[code_column].values:
                            stock_industry = industry_scores[industry_scores[code_column] == stock_code]
                            if not stock_industry.empty and 'industry_score' in stock_industry.columns:
                                stock_scores['industry_score'] = float(stock_industry['industry_score'].values[0])
                            else:
                                stock_scores['industry_score'] = 50
                        else:
                            stock_scores['industry_score'] = 50
                    else:
                        stock_scores['industry_score'] = 50
                    
                    # 添加技术面评分
                    stock_scores['technical_score'] = technical_score if isinstance(technical_score, (int, float)) else 50
                    
                    # 获取股票名称和当前价格
                    stock_name = stock_names.get(stock_code, "未知")
                    current_price = stock_prices.get(stock_code, 0.0)
                    
                    # 计算综合评分（使用分析器管理器的权重）
                    potential_score = self.analyzer_manager.calculate_potential_score(stock_scores)
                    
                    # 添加到结果列表
                    stock_result = {
                        'code': stock_code,
                        'name': stock_name,
                        'current_price': current_price,
                        'fund_flow_score': stock_scores.get('fund_flow_score', 50),
                        'social_score': stock_scores.get('social_score', 50),
                        'fundamental_score': stock_scores.get('fundamental_score', 50),
                        'technical_score': stock_scores.get('technical_score', 50),
                        'industry_score': stock_scores.get('industry_score', 50),
                        'potential_score': potential_score
                    }
                    
                    # 添加到所有分析过的股票列表
                    self.all_analyzed_stocks.append(stock_result)
                    
                    # 添加到结果列表
                    results.append(stock_result)
                    
                    success_count += 1
                    
                    # 如果设置了进度回调函数，更新状态为完成
                    if self.progress_callback:
                        self.progress_callback(i+1, total_stocks, stock_code, "分析完成")
                    
                except Exception as e:
                    logger.warning(f"分析股票 {stock_code} 时出错: {e}")
                    error_count += 1
                    
                    # 如果设置了进度回调函数，更新状态为失败
                    if self.progress_callback:
                        self.progress_callback(i+1, total_stocks, stock_code, "分析失败")
            
            logger.info(f"股票分析完成，共 {total_stocks} 只，成功 {success_count} 只，失败 {error_count} 只")
            
            # 按潜力评分排序
            results = sorted(results, key=lambda x: x['potential_score'], reverse=True)
            
            # 筛选符合最低评分要求的股票
            filtered_results = [r for r in results if r['potential_score'] >= min_score]
            
            # 记录日志，显示筛选情况
            logger.info(f"最低评分要求: {min_score}, 符合要求的股票数量: {len(filtered_results)}")
            logger.info(f"所有股票评分: {[round(r['potential_score'], 2) for r in results]}")
            
            # 取前N名
            filtered_results = filtered_results[:top_n]
            
            # 生成推荐理由
            if filtered_results and hasattr(self, 'recommendation_agent') and callable(self.recommendation_agent):
                filtered_results = self.recommendation_agent(filtered_results)
            
            # 缓存结果
            self._cache_results(filtered_results if filtered_results else results[:top_n])
            
            # 如果没有符合最低评分要求的股票，但用户设置的最低评分很低（低于30），则返回评分最高的几只股票
            if not filtered_results and min_score < 30:
                logger.info(f"没有股票符合最低评分要求 {min_score}，但最低评分设置较低，返回评分最高的 {top_n} 只股票")
                top_results = results[:top_n]
                if hasattr(self, 'recommendation_agent') and callable(self.recommendation_agent):
                    top_results = self.recommendation_agent(top_results)
                return top_results
            
            return filtered_results
        except Exception as e:
            logger.error(f"分析流程执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
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
    
    def _prepare_stock_prompt(self, stock):
        """准备股票推荐提示词
        
        Args:
            stock: 股票信息字典
            
        Returns:
            str: 提示词
        """
        prompt = f"""请为以下股票生成一段专业的投资推荐理由，字数控制在200字以内：

股票代码：{stock['code']}
股票名称：{stock['name']}

评分情况：
"""
        
        # 添加各项评分
        for score_name, score_value in stock['scores'].items():
            # 将评分名称转换为更易读的形式
            readable_name = {
                'fund_flow_score': '资金流向',
                'social_score': '社交热度',
                'fundamental_score': '基本面',
                'industry_score': '行业表现',
                'technical_score': '技术面'
            }.get(score_name, score_name)
            
            prompt += f"- {readable_name}评分：{score_value:.2f}/100\n"
        
        prompt += f"\n总评分：{stock['total_score']:.2f}/100\n\n"
        prompt += "请根据以上评分，分析该股票的投资价值，并给出推荐理由。语言要专业、简洁，重点突出其投资亮点和潜在风险。"
        
        return prompt
    
    def _generate_recommendation_reason(self, prompt):
        """使用LLM生成推荐理由
        
        Args:
            prompt: 提示词
            
        Returns:
            str: 生成的推荐理由
        """
        try:
            # 使用LangChain调用OpenAI API
            response = self.llm.invoke(prompt)
            
            # 返回生成的文本
            return response.content
        except Exception as e:
            logger.error(f"生成推荐理由失败: {e}")
            return "无法生成推荐理由，请检查API配置或网络连接。" 