#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit Web应用入口
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# 配置页面
st.set_page_config(
    page_title="股票潜力分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面标题
st.title("股票潜力分析系统")
st.markdown("基于资金热点和用户讨论热点，寻找具有上升潜力的股票")

def run_analysis():
    """运行分析流程"""
    try:
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 初始化数据管理器
        status_text.text("正在初始化数据管理器...")
        data_manager = DataManager()
        progress_bar.progress(10)
        
        # 初始化分析管理器
        status_text.text("正在初始化分析管理器...")
        analyzer_manager = AnalyzerManager()
        progress_bar.progress(20)
        
        # 初始化Agent管理器
        status_text.text("正在初始化Agent管理器...")
        agent_manager = AgentManager(data_manager, analyzer_manager)
        progress_bar.progress(30)
        
        # 将agent_manager保存到session_state中
        st.session_state['agent_manager'] = agent_manager
        
        # 收集数据
        status_text.text("正在收集市场数据...")
        data = agent_manager.collect_data()
        progress_bar.progress(40)
        
        # 分析资金流向
        status_text.text("正在分析资金流向数据...")
        fund_flow_scores = analyzer_manager.analyze_fund_flow(data.get('fund_flow_data', pd.DataFrame()))
        progress_bar.progress(50)
        
        # 分析社交热度
        status_text.text("正在分析社交媒体讨论数据...")
        social_scores = analyzer_manager.analyze_social_discussion(data.get('social_data', pd.DataFrame()))
        progress_bar.progress(60)
        
        # 获取分析参数
        top_n = st.session_state.get('stock_count', 5)
        min_score = st.session_state.get('min_score', 60)
        max_stocks = st.session_state.get('max_stocks', 8)
        
        # 创建一个占位符用于显示技术面分析进度
        tech_analysis_status = st.empty()
        api_limit_notice = st.info("注意：由于API调用频率限制，每次获取股票数据需要等待约30秒，分析过程可能需要较长时间，请耐心等待。")
        
        # 获取候选股票列表
        status_text.text("正在筛选候选股票...")
        candidate_stocks = agent_manager.get_candidate_stocks(data, fund_flow_scores, social_scores, max_stocks)
        
        # 将候选股票列表保存到session_state中
        st.session_state['candidate_stocks'] = candidate_stocks
        
        if not candidate_stocks:
            st.error("无法获取候选股票列表")
            return []
        
        # 显示候选股票列表和状态
        st.subheader("候选股票分析状态")
        
        # 创建一个DataFrame来存储候选股票信息
        candidate_df = pd.DataFrame({
            '代码': [stock['code'] for stock in candidate_stocks],
            '名称': [stock['name'] for stock in candidate_stocks],
            '候选理由': [stock.get('selection_reason', '未知') for stock in candidate_stocks],
            '状态': ['等待分析'] * len(candidate_stocks)
        })
        
        # 创建一个占位符来显示候选股票状态表格
        candidate_table = st.empty()
        candidate_table.dataframe(candidate_df)
        
        # 使用自定义的进度回调函数
        def progress_callback(current, total, stock_code="", status="分析中"):
            progress_value = 60 + (current / total) * 30
            progress_bar.progress(int(progress_value))
            tech_analysis_status.text(f"正在分析技术面数据 ({current}/{total}): {stock_code} - 由于API限制，每次请求间隔约30秒")
            
            # 更新候选股票状态
            if stock_code:
                idx = candidate_df[candidate_df['代码'] == stock_code].index
                if len(idx) > 0:
                    candidate_df.loc[idx[0], '状态'] = status
                    candidate_table.dataframe(candidate_df)
        
        # 将回调函数传递给agent_manager
        agent_manager.set_progress_callback(progress_callback)
        
        # 运行分析流程
        results = agent_manager.run_analysis_pipeline(top_n=top_n, min_score=min_score, max_stocks_to_process=max_stocks, candidate_stocks=candidate_stocks)
        
        # 更新所有已完成股票的状态
        completed_codes = [result['code'] for result in results]
        for i, row in candidate_df.iterrows():
            if row['代码'] in completed_codes:
                candidate_df.loc[i, '状态'] = '分析完成'
            elif row['状态'] == '等待分析' or row['状态'] == '分析中':
                candidate_df.loc[i, '状态'] = '分析失败或跳过'
        
        candidate_table.dataframe(candidate_df)
        
        # 完成进度
        progress_bar.progress(100)
        status_text.text("分析完成！")
        tech_analysis_status.empty()
        
        return results
        
    except Exception as e:
        logger.error(f"分析过程出错: {e}")
        st.error(f"分析过程出错: {str(e)}")
        return []

def display_results(results):
    """显示分析结果"""
    if not results:
        st.warning("没有找到符合条件的股票")
        # 尝试从AgentManager获取所有分析过的股票数据
        try:
            agent_manager = st.session_state.get('agent_manager')
            if agent_manager and hasattr(agent_manager, 'all_analyzed_stocks'):
                all_stocks = agent_manager.all_analyzed_stocks
                if all_stocks and len(all_stocks) > 0:
                    st.subheader("所有分析过的股票数据")
                    # 转换为DataFrame以便显示
                    df = pd.DataFrame(all_stocks)
                    
                    # 显示结果表格
                    st.dataframe(
                        df[['code', 'name', 'current_price', 'potential_score']],
                        column_config={
                            'code': '股票代码',
                            'name': '股票名称',
                            'current_price': '当前价格',
                            'potential_score': st.column_config.ProgressColumn(
                                '潜力评分',
                                min_value=0,
                                max_value=100,
                                format="%d%%",
                            )
                        },
                        hide_index=True
                    )
                    
                    # 显示详细评分
                    st.subheader("详细评分情况")
                    
                    # 获取候选理由
                    selection_reasons = {}
                    candidate_stocks = st.session_state.get('candidate_stocks', [])
                    for stock in candidate_stocks:
                        selection_reasons[stock['code']] = stock.get('selection_reason', '未知')
                    
                    score_df = pd.DataFrame({
                        '股票代码': [s['code'] for s in all_stocks],
                        '股票名称': [s['name'] for s in all_stocks],
                        '候选理由': [selection_reasons.get(s['code'], '未知') for s in all_stocks],
                        '资金流入评分': [s.get('fund_flow_score', 0) for s in all_stocks],
                        '社交热度评分': [s.get('social_score', 0) for s in all_stocks],
                        '基本面评分': [s.get('fundamental_score', 0) for s in all_stocks],
                        '技术面评分': [s.get('technical_score', 0) for s in all_stocks],
                        '行业热度评分': [s.get('industry_score', 0) for s in all_stocks],
                        '潜力总评分': [s.get('potential_score', 0) for s in all_stocks]
                    })
                    st.dataframe(score_df, hide_index=True)
                    
                    st.info("以上是所有分析过的股票，但它们的潜力评分未达到设定的最低标准。您可以在侧边栏降低最低潜力评分阈值后重新分析。")
        except Exception as e:
            st.error(f"无法获取分析过的股票数据: {e}")
        return
    
    # 转换为DataFrame以便显示
    df = pd.DataFrame(results)
    
    # 显示结果表格
    st.subheader("潜力股票列表")
    st.dataframe(
        df[['code', 'name', 'current_price', 'potential_score']],
        column_config={
            'code': '股票代码',
            'name': '股票名称',
            'current_price': '当前价格',
            'potential_score': st.column_config.ProgressColumn(
                '潜力评分',
                min_value=0,
                max_value=100,
                format="%d%%",
            )
        },
        hide_index=True
    )
    
    # 显示详细分析
    st.subheader("详细分析")
    for i, stock in enumerate(results):
        with st.expander(f"{stock['name']} ({stock['code']})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**推荐理由**: {stock['reason']}")
                
                st.markdown("### 评分详情")
                scores_df = pd.DataFrame({
                    '指标': ['资金流入评分', '社交讨论热度', '基本面评分', '技术面评分', '行业热度'],
                    '分数': [
                        stock.get('fund_flow_score', 0), 
                        stock.get('social_score', 0),
                        stock.get('fundamental_score', 0),
                        stock.get('technical_score', 0),
                        stock.get('industry_score', 0)
                    ]
                })
                st.dataframe(scores_df, hide_index=True)
            
            with col2:
                # 绘制雷达图
                categories = ['资金流', '社交热度', '基本面', '技术面', '行业热度']
                values = [
                    stock.get('fund_flow_score', 0) / 100, 
                    stock.get('social_score', 0) / 100,
                    stock.get('fundamental_score', 0) / 100,
                    stock.get('technical_score', 0) / 100,
                    stock.get('industry_score', 0) / 100
                ]
                
                # 闭合雷达图
                values = np.append(values, values[0])
                categories = np.append(categories, categories[0])
                
                fig = plt.figure(figsize=(4, 4))
                ax = fig.add_subplot(111, polar=True)
                ax.plot(np.linspace(0, 2*np.pi, len(values)), values)
                ax.fill(np.linspace(0, 2*np.pi, len(values)), values, alpha=0.25)
                ax.set_thetagrids(np.degrees(np.linspace(0, 2*np.pi, len(categories), endpoint=False)), categories)
                ax.set_ylim(0, 1)
                ax.grid(True)
                st.pyplot(fig)

# 侧边栏配置
with st.sidebar:
    st.header("分析参数")
    
    analysis_type = st.selectbox(
        "分析类型",
        ["综合分析", "资金流向分析", "社交媒体热点分析", "基本面分析", "技术面分析"]
    )
    
    stock_count = st.slider("推荐股票数量", 3, 10, 5, help="由于API调用频率限制，建议选择较少的股票数量")
    st.session_state['stock_count'] = stock_count
    
    min_score = st.slider("最低潜力评分", 0, 100, 60)
    st.session_state['min_score'] = min_score
    
    max_stocks = st.slider("最大分析股票数量", 3, 15, 8, help="分析的股票数量越多，等待时间越长（每只股票约需30秒）")
    st.session_state['max_stocks'] = max_stocks
    
    include_sectors = st.multiselect(
        "包含行业",
        ["全部", "科技", "医药", "金融", "消费", "工业", "能源", "材料", "公用事业", "房地产", "通信"],
        default=["全部"]
    )
    
    run_button = st.button("开始分析", type="primary")

# 主界面
if run_button:
    # 运行分析
    results = run_analysis()
    
    # 显示结果
    display_results(results)
else:
    st.info("点击侧边栏中的「开始分析」按钮开始分析")
    
    # 显示项目信息
    st.markdown("""
    ## 关于本系统
    
    本系统通过分析多维度数据，为您推荐具有上升潜力的股票：
    
    1. **资金流向分析**：追踪大资金流入流出情况
    2. **社交媒体热点分析**：分析用户讨论热度和情感
    3. **基本面分析**：分析公司财务状况和估值
    4. **技术面分析**：分析价格走势和交易量
    5. **行业分析**：分析行业整体趋势
    
    系统利用大语言模型进行综合分析，多智能体协同工作，提供全面分析结果。
    """) 