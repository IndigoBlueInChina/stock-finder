# Stock Finder

基于资金热点和用户讨论热点，寻找具有上升潜力的股票的智能分析工具。

## 功能特点

- 资金流向分析：追踪大资金流入流出情况
- 社交媒体热点分析：分析用户讨论热度和情感
- LLM智能分析：利用大语言模型进行综合分析
- Agent框架：多智能体协同工作，提供全面分析

## 安装方法

确保已安装Python 3.11和Poetry

```bash
# 克隆仓库
git clone https://github.com/yourusername/stock-finder.git
cd stock-finder

# 安装依赖
poetry install
```

## 使用方法

```bash
# 激活环境
poetry shell

# 运行主程序
python src/main.py

# 或者运行Web界面
streamlit run src/app.py
```

## 项目结构

```
stock-finder/
├── data/               # 数据存储目录
├── src/                # 源代码
│   ├── agents/         # 智能体定义
│   ├── data_fetchers/  # 数据获取模块
│   ├── analyzers/      # 分析模块
│   ├── models/         # 模型定义
│   ├── utils/          # 工具函数
│   ├── main.py         # 主程序
│   └── app.py          # Web应用
├── tests/              # 测试代码
├── pyproject.toml      # 项目配置
└── README.md           # 项目说明
```

## 环境变量

创建`.env`文件并设置以下变量：

```
OPENAI_API_KEY=your_openai_api_key
```

## 许可证

MIT 