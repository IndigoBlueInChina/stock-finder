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
    with st.spinner("正在分析数据，请稍候..."):
        try:
            # 初始化数据管理器
            data_manager = DataManager()
            
            # 初始化分析管理器
            analyzer_manager = AnalyzerManager()
            
            # 初始化Agent管理器
            agent_manager = AgentManager(data_manager, analyzer_manager)
            
            # 运行分析流程
            results = agent_manager.run_analysis_pipeline()
            
            return results
        
        except Exception as e:
            logger.error(f"分析过程出错: {e}")
            st.error(f"分析过程出错: {str(e)}")
            return []

def display_results(results):
    """显示分析结果"""
    if not results:
        st.warning("没有找到符合条件的股票")
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
    
    stock_count = st.slider("推荐股票数量", 3, 20, 10)
    
    min_score = st.slider("最低潜力评分", 0, 100, 60)
    
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