"""
预设示例数据

用于快速演示Prompt压缩效果。
注意: context 必须是 List[str] 格式，匹配真实接口
"""

EXAMPLES = {
    "示例1 - 科技新闻": {
        "context": [
            "Apple is planning to build a new factory in California. The company has been exploring locations since last year.",
            "The factory will produce electric vehicle batteries and is expected to create 5,000 jobs.",
            "Apple aims to begin construction in early 2024 and complete the project within two years.",
        ],
        "query": "Where is Apple planning to build the factory and what will it produce?",
        "baseline_answer": "Apple is planning to build a factory in California to produce electric vehicle batteries. The project will create 5,000 jobs, with construction starting in early 2024.",
        "compressed_answer": "Apple plans a California factory for EV batteries. It will create 5,000 jobs with 2024 construction start.",
    },
    "示例2 - 产品描述": {
        "context": [
            "The iPhone 15 Pro Max features a titanium design that is both durable and lightweight.",
            "It includes the new A17 Pro chip for enhanced performance and gaming capabilities.",
            "The device supports USB-C for faster charging and data transfer speeds up to 10 Gbps.",
        ],
        "query": "What are the key features of the iPhone 15 Pro Max?",
        "baseline_answer": "The iPhone 15 Pro Max features a titanium design, A17 Pro chip for performance and gaming, and USB-C support for faster charging and data transfer up to 10 Gbps.",
        "compressed_answer": "iPhone 15 Pro Max has titanium design, A17 Pro chip, and USB-C for 10 Gbps transfer.",
    },
    "示例3 - 财务报告": {
        "context": [
            "Tesla reported Q3 revenue of $23.35 billion, beating analyst expectations of $21.8 billion.",
            "The company delivered 435,059 vehicles during the quarter, a 27% increase year-over-year.",
            "Gross margin for the quarter stood at 16.3%, down from 27.9% in the same period last year.",
        ],
        "query": "What was Tesla's Q3 revenue and how many vehicles did they deliver?",
        "baseline_answer": "Tesla reported Q3 revenue of $23.35 billion, beating expectations of $21.8 billion. The company delivered 435,059 vehicles in the quarter, representing a 27% year-over-year increase.",
        "compressed_answer": "Tesla Q3 revenue: $23.35B (beat $21.8B estimate). Delivered 435,059 vehicles (+27% YoY).",
    },
}

# 用于Gradio Dropdown的choices列表
EXAMPLE_CHOICES = list(EXAMPLES.keys())


def get_example(name: str):
    """获取指定名称的示例数据"""
    return EXAMPLES.get(name)
