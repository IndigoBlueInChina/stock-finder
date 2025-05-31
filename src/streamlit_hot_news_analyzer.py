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
import os
import json
from dotenv import load_dotenv

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.analyzers.hot_news_fund_flow_analyzer import HotNewsFundFlowAnalyzer
from src.data_fetchers.data_manager import DataManager

# 配置日志 - 只使用控制台日志
logger.remove()
logger.add(sys.stderr, level="INFO")

# 加载环境变量
load_dotenv()

# 从.env文件获取LLM配置
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-plus")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("OPENAI_API_BASE", "")  # OpenAI兼容API基础URL
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# 导入LangChain相关模块
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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

def init_llm(api_key=None):
    """初始化大语言模型
    
    Args:
        api_key: 可选的API密钥，如果提供则覆盖环境变量中的密钥
        
    Returns:
        LLM实例或None（如果配置无效）
    """
    try:
        # 使用提供的API密钥或环境变量中的密钥
        api_key = api_key or API_KEY
        
        # 检查API密钥是否设置
        if not api_key or api_key == "your_openai_api_key":
            logger.warning("未设置有效的OpenAI API密钥，LLM功能将被禁用")
            return None
                
        # 检查API基础URL是否设置
        if not API_BASE:
            logger.warning("未设置OpenAI API基础URL，将使用默认URL")
        
        logger.info(f"初始化LLM，模型: {MODEL_NAME}, 温度: {TEMPERATURE}")
        logger.info(f"使用API基础URL: {API_BASE}")
        
        # 使用OpenAI兼容模式配置
        llm = ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            api_key=api_key,
            base_url=API_BASE,  # 使用自定义的API基础URL
            max_tokens=MAX_TOKENS
        )
        
        return llm
    
    except Exception as e:
        logger.error(f"初始化LLM失败: {e}")
        # 返回None，禁用LLM功能
        return None

def get_stock_industry_info(stock_codes, data_manager=None):
    """获取股票的行业信息
    
    Args:
        stock_codes: 股票代码列表
        data_manager: 数据管理器实例，如果为None则创建新实例
        
    Returns:
        dict: 包含股票代码、名称和行业信息的字典
    """
    if data_manager is None:
        from src.data_fetchers.data_manager import DataManager
        data_manager = DataManager()
    
    # 获取股票行业对应关系
    try:
        industry_mapping = data_manager.get_stock_industry_mapping()
        
        # 创建结果字典
        result = {}
        for code in stock_codes:
            # 确保代码格式一致（6位数字）
            code_str = str(code).zfill(6)
            
            # 在行业映射中查找
            stock_info = industry_mapping[industry_mapping['代码'] == code_str]
            
            if not stock_info.empty:
                result[code_str] = {
                    '名称': stock_info.iloc[0]['名称'] if '名称' in stock_info.columns else '未知',
                    '行业': stock_info.iloc[0]['所属行业'] if '所属行业' in stock_info.columns else '未知'
                }
            else:
                # 如果找不到行业信息，添加默认值
                result[code_str] = {
                    '名称': '未知',
                    '行业': '未知'
                }
        
        return result
    except Exception as e:
        logger.error(f"获取股票行业信息失败: {e}")
        
        # 返回默认值
        result = {}
        for code in stock_codes:
            code_str = str(code).zfill(6)
            result[code_str] = {
                '名称': '未知',
                '行业': '未知'
            }
        
        return result

def categorize_stocks_by_industry(industry_info):
    """根据行业对股票进行分类
    
    Args:
        industry_info: 股票行业信息字典
        
    Returns:
        dict: 按行业分类的股票字典
    """
    categories = {
        '汽车相关': [],
        '电子科技': [],
        '金融银行': [],
        '能源电力': [],
        '消费零售': [],
        '医药健康': [],
        '其他行业': []
    }
    
    industry_keywords = {
        '汽车相关': ['汽车', '零部件', '智能驾驶', '新能源车'],
        '电子科技': ['电子', '科技', '半导体', '芯片', '通信', '软件', '互联网', '人工智能', 'AI'],
        '金融银行': ['银行', '证券', '保险', '金融', '基金', '信托'],
        '能源电力': ['电力', '能源', '石油', '煤炭', '天然气', '新能源', '风电', '光伏', '核能'],
        '消费零售': ['消费', '零售', '食品', '饮料', '白酒', '餐饮', '服装', '家电'],
        '医药健康': ['医药', '生物', '医疗', '健康', '疫苗', '制药']
    }
    
    for code, info in industry_info.items():
        if not info or '行业' not in info:
            categories['其他行业'].append({
                '代码': code,
                '名称': info.get('名称', '未知'),
                '行业': '未知'
            })
            continue
            
        industry = info['行业']
        name = info.get('名称', '未知')
        
        # 根据行业关键词进行分类
        categorized = False
        for category, keywords in industry_keywords.items():
            if any(keyword in industry for keyword in keywords):
                categories[category].append({
                    '代码': code,
                    '名称': name,
                    '行业': industry
                })
                categorized = True
                break
        
        # 如果没有匹配到任何分类，放入其他行业
        if not categorized:
            categories['其他行业'].append({
                '代码': code,
                '名称': name,
                '行业': industry
            })
    
    return categories

def generate_industry_reports(news_df, matched_stocks_df=None, concepts_df=None, api_key=None):
    """使用LLM生成行业主题报告
    
    Args:
        news_df: 新闻DataFrame
        matched_stocks_df: 匹配到的股票DataFrame
        concepts_df: 匹配到的概念板块DataFrame
        api_key: 可选的API密钥，如果提供则覆盖环境变量中的密钥
        
    Returns:
        str: 行业主题报告文本
    """
    # 初始化LLM
    llm = init_llm(api_key)
    if not llm:
        return "API配置无效，请检查环境变量中的LLM配置或在侧边栏中输入有效的API密钥。"
    
    # 准备新闻内容
    news_content = ""
    if not news_df.empty:
        for _, row in news_df.head(50).iterrows():  # 限制处理的新闻数量
            if '标题' in row and '内容' in row:
                news_content += f"标题: {row['标题']}\n内容: {row['内容']}\n"
    
    # 准备匹配到的股票信息
    stocks_info = ""
    if matched_stocks_df is not None and not matched_stocks_df.empty:
        # 获取股票行业信息
        stock_codes = matched_stocks_df['代码'].tolist() if '代码' in matched_stocks_df.columns else []
        industry_info = get_stock_industry_info(stock_codes)
        
        # 按行业分类股票
        categorized_stocks = categorize_stocks_by_industry(industry_info)
        
        # 格式化股票信息
        stocks_info = "股票信息（已按行业分类）:\n"
        for category, stocks in categorized_stocks.items():
            if stocks:
                stocks_info += f"\n【{category}】\n"
                for stock in stocks:
                    stocks_info += f"{stock['代码']} {stock['名称']} - {stock['行业']}\n"
    
    # 准备匹配到的概念板块信息
    concepts_info = ""
    if concepts_df is not None and not concepts_df.empty:
        concepts_info = "匹配到的概念板块:\n"
        for _, row in concepts_df.iterrows():
            if '名称' in row:
                concepts_info += f"{row['名称']}\n"
    
    # 构建提示词
    prompt = f"""你是一位专业的财经分析师，请根据以下新闻内容，提炼出3-5个主要的行业主题，并为每个主题撰写一篇简短的分析报告。

新闻内容:
{news_content}

股票信息（已按行业分类）:
{stocks_info}

{concepts_info}

对于每个行业主题，请按照以下格式提供分析:
主题名称: [行业主题]
主题内容: [对该主题的新闻内容进行提炼和总结，200-300字]
相关股票: [仅列出与该主题直接相关的股票代码和名称，必须确保股票与主题行业高度相关，如果没有明确相关的股票，则只写"无直接相关股票"]
影响分析: [该主题对相关行业和股票的潜在影响，以及未来可能的走势，100-150字]

请确保分析客观、专业，避免过度乐观或悲观的预测。每个主题之间用"---"分隔。

特别注意事项：
1. 相关股票必须与主题行业高度相关，例如智能汽车主题应从【汽车相关】分类中选择相关股票
2. 在选择相关股票时，请参考上面提供的股票行业分类，确保股票的主营业务与主题直接相关
3. 如果某个主题在分类中没有相关股票，请只写"无直接相关股票"，不要添加任何股票代码
4. 不要强行关联不相关的股票，如果不确定股票是否相关，请直接写"无直接相关股票"
5. 如果新闻中明确提到某些上市公司与主题相关，请检查这些公司是否在提供的股票列表中，只有在列表中找到时才列出
6. 严格遵守行业对应关系：
   - 智能汽车、新能源汽车主题 -> 【汽车相关】分类中的股票
   - 银行、保险、证券主题 -> 【金融银行】分类中的股票
   - 科技、芯片、通信主题 -> 【电子科技】分类中的股票
   - 能源、电力、石油主题 -> 【能源电力】分类中的股票
   - 医药、健康、生物主题 -> 【医药健康】分类中的股票
   - 消费、零售、食品主题 -> 【消费零售】分类中的股票
7. 不要混淆不同行业，例如不要将银行股票与汽车主题关联，也不要将能源股票与科技主题关联"""

    # 打印提示词，用于调试
    logger.info("行业主题报告提示词:")
    logger.info(prompt)
    
    # 设置重试参数
    max_retries = 3
    retry_delay = 5  # 初始延迟5秒
    
    # 尝试生成报告
    for attempt in range(max_retries):
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": "你是一位专业的财经分析师，擅长分析新闻与股票市场的关系。"},
                {"role": "user", "content": prompt}
            ]
            
            # 设置超时时间
            response = llm.invoke(messages, timeout=60)
            result = response.content
            
            # 打印LLM响应，用于调试
            logger.info("行业主题报告LLM响应:")
            logger.info(result)
            
            return result
            
        except Exception as e:
            logger.error(f"生成行业主题报告失败: {e}")
            if attempt < max_retries - 1:
                logger.info(f"尝试重试 ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                return f"生成行业主题报告失败: {str(e)}"

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
    
    # 初始化会话状态变量
    if 'news_df' not in st.session_state:
        st.session_state.news_df = pd.DataFrame()
    if 'fund_flow_df' not in st.session_state:
        st.session_state.fund_flow_df = pd.DataFrame()
    if 'matched_stocks_df' not in st.session_state:
        st.session_state.matched_stocks_df = pd.DataFrame()
    if 'matched_concepts_df' not in st.session_state:
        st.session_state.matched_concepts_df = pd.DataFrame()
    if 'industry_reports' not in st.session_state:
        st.session_state.industry_reports = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "keywords" not in st.session_state:
        st.session_state.keywords = None
    if "concepts" not in st.session_state:
        st.session_state.concepts = None
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    
    # 加载环境变量
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # 创建侧边栏
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
        
        # API密钥隐藏处理
        user_api_key = API_KEY
        
        # 添加运行按钮
        run_button = st.button("运行分析", type="primary")
        
        if run_button:
            with st.spinner("正在分析热点新闻和资金流向..."):
                try:
                    # 创建分析器
                    analyzer = HotNewsFundFlowAnalyzer()
                    
                    # 获取热点新闻
                    news_df = analyzer.get_hot_news(days=news_days)
                    st.session_state.news_df = news_df
                    
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
                    st.session_state.matched_stocks_df = result
                    st.session_state.matched_concepts_df = matched_concepts
                    st.session_state.last_run = {
                        "news_days": news_days,
                        "fund_flow_days_list": fund_flow_days_list,
                        "top_n": top_n,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # 获取资金流向数据
                    data_manager = DataManager()
                    st.session_state.fund_flow_df = data_manager.get_fund_flow_data()
                    
                    # 检查是否有有效的LLM配置
                    api_key = user_api_key or API_KEY
                    if api_key and api_key != "your_openai_api_key":
                        with st.spinner("正在生成行业主题报告..."):
                            st.session_state.industry_reports = generate_industry_reports(
                                news_df, 
                                result, 
                                matched_concepts,
                                api_key
                            )
                    
                    st.success("分析完成！")
                except Exception as e:
                    st.error(f"分析过程中出错: {str(e)}")
                    logger.error(f"分析过程中出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    
    # 显示结果
    if st.session_state.result is not None and not st.session_state.result.empty:
        if st.session_state.last_run:
            st.caption(f"最后运行时间: {st.session_state.last_run['time']}")
        
        # 创建主要内容区域的选项卡
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["热点股票", "热点概念", "新闻关键词", "原始新闻", "行业主题报告"])
        
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
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{float(x):.2f}" if pd.notna(x) and not isinstance(x, str) else (
                                x if isinstance(x, str) else str(x)
                            )
                        )
                
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
            if st.session_state.news_df is not None and not st.session_state.news_df.empty:
                st.subheader("热点新闻列表")
                
                # 显示新闻来源分布
                if '来源' in st.session_state.news_df.columns:
                    news_source_count = st.session_state.news_df['来源'].value_counts().reset_index()
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
                    st.session_state.news_df,
                    use_container_width=True,
                    column_config={
                        "标题": st.column_config.TextColumn("新闻标题"),
                        "内容": st.column_config.TextColumn("新闻内容"),
                        "来源": st.column_config.TextColumn("来源")
                    }
                )
            else:
                st.info("未找到新闻数据")
        
        # 选项卡5: 行业主题报告
        with tab5:
            # 检查LLM配置是否有效
            api_key = user_api_key or API_KEY
            if not api_key or api_key == "your_openai_api_key":
                st.warning("请在环境变量中设置有效的API密钥以使用行业主题报告功能")
            elif st.session_state.industry_reports:
                st.subheader("行业主题报告")
                
                # 将报告文本按"---"分割成多个主题报告
                reports = st.session_state.industry_reports.split("---")
                
                # 显示每个报告
                for i, report in enumerate(reports):
                    if report.strip():  # 确保报告不是空字符串
                        # 尝试提取主题名称作为标题
                        title = "行业主题报告"
                        if "主题名称:" in report or "主题名称：" in report:
                            for line in report.split("\n"):
                                if line.startswith("主题名称:") or line.startswith("主题名称："):
                                    title = line.split(":", 1)[1].strip() if ":" in line else line.split("：", 1)[1].strip()
                                    break
                        
                        # 使用expander显示报告
                        with st.expander(title, expanded=i==0):
                            # 使用markdown格式显示报告内容
                            st.markdown(report.strip())
                        st.divider()
            else:
                # 添加手动生成按钮
                if st.button("生成行业主题报告"):
                    with st.spinner("正在生成行业主题报告..."):
                        st.session_state.industry_reports = generate_industry_reports(
                            st.session_state.news_df, 
                            st.session_state.matched_stocks_df, 
                            st.session_state.matched_concepts_df,
                            api_key
                        )
                        st.experimental_rerun()
                else:
                    st.info("点击上方按钮生成行业主题报告")
    elif run_button:
        st.warning("未找到任何与新闻热点匹配且资金流入的股票")
    else:
        st.info("请点击侧边栏中的'运行分析'按钮开始分析")
    
    # 页脚
    st.divider()
    st.caption("热点新闻资金流共振分析器 © 2024")

if __name__ == "__main__":
    main() 