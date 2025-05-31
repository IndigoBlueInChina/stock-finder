#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
热点新闻资金流共振分析器的Streamlit界面
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from loguru import logger
import time
import plotly.express as px

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.analyzers.hot_news_fund_flow_analyzer import HotNewsFundFlowAnalyzer

# 配置日志 - 只使用控制台日志
logger.remove()
logger.add(sys.stderr, level="INFO")

def format_number(num):
    """格式化数字，将大数字转换为带单位的形式"""
    if pd.isna(num):
        return "N/A"
    
    try:
        # 确保num是数值类型
        num = float(num)
        abs_num = abs(num)
        if abs_num >= 1e8:
            return f"{num/1e8:.2f}亿"
        elif abs_num >= 1e4:
            return f"{num/1e4:.2f}万"
        else:
            return f"{num:.2f}"
    except (ValueError, TypeError):
        # 如果无法转换为数值，则返回原值
        return str(num)

def format_percentage(value):
    """格式化百分比"""
    if pd.isna(value):
        return "N/A"
    
    try:
        # 确保value是数值类型
        value = float(value)
        return f"{value:.2f}%"
    except (ValueError, TypeError):
        # 如果无法转换为数值，则返回原值
        return str(value)

def main():
    """Streamlit应用主函数"""
    st.set_page_config(
        page_title="热点新闻资金流共振分析器",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("📊 热点新闻资金流共振分析器")
    st.markdown("""
    本工具分析新闻热点与资金流入的共振效应，帮助发现市场热点和资金流向一致的股票。
    """)
    
    # 侧边栏参数设置
    with st.sidebar:
        st.header("参数设置")
        news_days = st.slider("新闻天数", min_value=1, max_value=7, value=1, help="获取最近几天的新闻")
        
        fund_flow_options = ["今日(1天)", "3天", "5天", "10天"]
        fund_flow_selected = st.multiselect(
            "资金流向周期",
            options=fund_flow_options,
            default=["今日(1天)", "3天", "5天"],
            help="选择要分析的资金流向周期"
        )
        
        # 将选择转换为天数
        fund_flow_days_map = {"今日(1天)": 1, "3天": 3, "5天": 5, "10天": 10}
        fund_flow_days_list = [fund_flow_days_map[option] for option in fund_flow_selected]
        
        top_n = st.slider("显示结果数量", min_value=5, max_value=50, value=20, help="显示前N个结果")
        
        run_button = st.button("运行分析", type="primary")
    
    # 初始化会话状态
    if "result" not in st.session_state:
        st.session_state.result = None
    if "keywords" not in st.session_state:
        st.session_state.keywords = None
    if "concepts" not in st.session_state:
        st.session_state.concepts = None
    if "news" not in st.session_state:
        st.session_state.news = None
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    
    # 运行分析
    if run_button:
        with st.spinner("正在分析热点新闻和资金流向..."):
            # 创建分析器
            analyzer = HotNewsFundFlowAnalyzer()
            
            # 获取热点新闻
            news_df = analyzer.get_hot_news(days=news_days)
            st.session_state.news = news_df
            
            # 提取关键词
            keywords_dict = analyzer.extract_keywords_from_news(news_df)
            st.session_state.keywords = keywords_dict
            
            # 匹配概念板块
            matched_concepts = analyzer.match_keywords_with_concepts(keywords_dict)
            st.session_state.concepts = matched_concepts
            
            # 运行分析
            result = analyzer.analyze(
                news_days=news_days, 
                fund_flow_days_list=fund_flow_days_list, 
                top_n=top_n
            )
            
            st.session_state.result = result
            st.session_state.last_run = {
                "news_days": news_days,
                "fund_flow_days_list": fund_flow_days_list,
                "top_n": top_n,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    # 显示结果
    if st.session_state.result is not None and not st.session_state.result.empty:
        st.success(f"分析完成! 发现 {len(st.session_state.result)} 只热点股票")
        
        if st.session_state.last_run:
            st.caption(f"最后运行时间: {st.session_state.last_run['time']}")
        
        # 创建选项卡
        tab1, tab2, tab3, tab4 = st.tabs(["热点股票", "热点概念", "新闻关键词", "原始新闻"])
        
        # 选项卡1: 热点股票
        with tab1:
            result = st.session_state.result.copy()
            
            # 创建可视化图表
            if len(result) > 0:
                # 综合得分柱状图
                st.subheader("热点股票综合得分排名")
                fig = px.bar(
                    result.head(15),
                    x="名称",
                    y="综合得分",
                    color="综合得分",
                    text="代码",
                    color_continuous_scale="Viridis",
                    title="热点股票综合得分排名 (前15名)"
                )
                fig.update_layout(xaxis_title="股票名称", yaxis_title="综合得分")
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示结果表格
                st.subheader("热点股票详细数据")
                
                # 格式化表格数据
                display_cols = ["代码", "名称", "综合得分", "新闻匹配分数"]
                
                # 添加资金流向列
                for days in fund_flow_days_list:
                    if days == 1:
                        prefix = "今日"
                    elif days == 3:
                        prefix = "3日"
                    elif days == 5:
                        prefix = "5日"
                    else:
                        prefix = f"{days}日"
                    
                    main_flow_col = f"{prefix}主力净流入-净额"
                    if main_flow_col in result.columns:
                        display_cols.append(main_flow_col)
                    
                    main_flow_pct_col = f"{prefix}主力净流入-净占比"
                    if main_flow_pct_col in result.columns:
                        display_cols.append(main_flow_pct_col)
                
                # 添加涨跌幅列
                for days in fund_flow_days_list:
                    if days == 1:
                        prefix = "今日"
                    elif days == 3:
                        prefix = "3日"
                    elif days == 5:
                        prefix = "5日"
                    else:
                        prefix = f"{days}日"
                    
                    change_col = f"{prefix}涨跌幅"
                    if change_col in result.columns:
                        display_cols.append(change_col)
                
                # 只显示存在的列
                display_cols = [col for col in display_cols if col in result.columns]
                
                # 创建显示表格
                display_df = result[display_cols].copy()
                
                # 格式化资金流向数据
                for col in display_df.columns:
                    if "净流入-净额" in col:
                        display_df[col] = display_df[col].apply(format_number)
                    elif "净占比" in col or "涨跌幅" in col:
                        display_df[col] = display_df[col].apply(format_percentage)
                    elif col in ["综合得分", "新闻匹配分数"]:
                        display_df[col] = display_df[col].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x))
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config={
                        "代码": st.column_config.TextColumn("股票代码"),
                        "名称": st.column_config.TextColumn("股票名称"),
                        "综合得分": st.column_config.TextColumn("综合得分"),
                        "新闻匹配分数": st.column_config.TextColumn("新闻匹配分数")
                    }
                )
        
        # 选项卡2: 热点概念
        with tab2:
            if st.session_state.concepts is not None and not st.session_state.concepts.empty:
                concepts = st.session_state.concepts.copy()
                
                # 创建概念板块柱状图
                st.subheader("热点概念板块排名")
                fig = px.bar(
                    concepts.head(15),
                    x="概念板块",
                    y="新闻匹配分数",
                    color="新闻匹配分数",
                    color_continuous_scale="Teal",
                    title="热点概念板块排名 (前15名)"
                )
                fig.update_layout(xaxis_title="概念板块", yaxis_title="新闻匹配分数")
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示概念板块表格
                st.subheader("热点概念板块详细数据")
                st.dataframe(
                    concepts,
                    use_container_width=True,
                    column_config={
                        "概念板块": st.column_config.TextColumn("概念板块"),
                        "新闻匹配分数": st.column_config.NumberColumn("新闻匹配分数", format="%.2f"),
                        "匹配关键词": st.column_config.TextColumn("匹配关键词")
                    }
                )
            else:
                st.info("未找到匹配的概念板块")
        
        # 选项卡3: 新闻关键词
        with tab3:
            if st.session_state.keywords:
                # 将关键词字典转换为DataFrame
                keywords_df = pd.DataFrame(
                    list(st.session_state.keywords.items()),
                    columns=["关键词", "权重"]
                ).sort_values("权重", ascending=False)
                
                # 创建关键词柱状图
                st.subheader("热点新闻关键词排名")
                fig = px.bar(
                    keywords_df.head(30),
                    x="关键词",
                    y="权重",
                    color="权重",
                    color_continuous_scale="Viridis",
                    title="热点新闻关键词排名 (前30名)"
                )
                fig.update_layout(xaxis_title="关键词", yaxis_title="权重")
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示关键词表格
                st.subheader("热点新闻关键词列表")
                st.dataframe(
                    keywords_df,
                    use_container_width=True,
                    column_config={
                        "关键词": st.column_config.TextColumn("关键词"),
                        "权重": st.column_config.NumberColumn("权重", format="%.4f")
                    }
                )
            else:
                st.info("未找到关键词")
        
        # 选项卡4: 原始新闻
        with tab4:
            if st.session_state.news is not None and not st.session_state.news.empty:
                st.subheader("热点新闻列表")
                
                # 显示新闻来源分布
                news_source_count = st.session_state.news['来源'].value_counts().reset_index()
                news_source_count.columns = ['来源', '数量']
                
                fig = px.pie(
                    news_source_count,
                    values='数量',
                    names='来源',
                    title="热点新闻来源分布"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示新闻表格
                st.dataframe(
                    st.session_state.news,
                    use_container_width=True,
                    column_config={
                        "标题": st.column_config.TextColumn("新闻标题"),
                        "内容": st.column_config.TextColumn("新闻内容"),
                        "来源": st.column_config.TextColumn("来源")
                    }
                )
            else:
                st.info("未找到新闻数据")
    elif run_button:
        st.warning("未找到任何与新闻热点匹配且资金流入的股票")
    else:
        st.info("请点击侧边栏中的'运行分析'按钮开始分析")
    
    # 页脚
    st.divider()
    st.caption("热点新闻资金流共振分析器 © 2024")

if __name__ == "__main__":
    main() 