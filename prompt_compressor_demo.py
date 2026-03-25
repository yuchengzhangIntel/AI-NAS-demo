"""
Prompt Compressor Interactive Demo

使用Gradio构建的交互式Demo，展示Prompt压缩算法的效果。
"""

import gradio as gr
import plotly.graph_objects as go
import os
from typing import List, Tuple

from compressor import MockCompressor, CompressionResult
from data.examples import EXAMPLE_CHOICES, get_example


def parse_labeled_original_to_highlighted(labeled_original: str) -> List[Tuple[str, str]]:
    """
    解析真实接口返回的 fn_labeled_original_prompt

    格式: "word1 1\t\t|\t\tword2 0\t\t|\t\tword3 1..."
    返回: [("word1", "keep"), ("word2", "drop"), ("word3", "keep")]

    Label映射:
    - "entity": 黄色 - 命名实体（大写且非句首）
    - "keep": 绿色 - 关键保留 (label="1")
    - "drop": 浅灰 - 冗余删除 (label="0")
    """
    if not labeled_original:
        return []

    word_sep = "\t\t|\t\t"
    segments = labeled_original.split(word_sep)

    result = []
    for segment in segments:
        if not segment.strip():
            continue
        # 格式: "word label"
        parts = segment.rsplit(" ", 1)
        if len(parts) == 2:
            word, label = parts

            # label="1" -> keep, label="0" -> drop
            if label.strip() == "1":
                label_name = "keep"
            else:
                label_name = "drop"

            # 简单NER规则: 大写且非句首 -> entity
            # 排除常见的句首大写词和标点
            if word and word[0].isupper() and len(word) > 2 and word not in ["The", "A", "An", "It", "This", "That"]:
                label_name = "entity"

            result.append((word, label_name))

    return result


def create_performance_plot(result: CompressionResult):
    """使用Plotly生成分组柱状图"""
    metrics = ["KV Cache (Tokens)", "TTFT (s)", "Total Time (s)"]
    original_values = [
        result.origin_tokens,
        result.ttft_original,
        result.total_time_original,
    ]
    compressed_values = [
        result.compressed_tokens,
        result.ttft_compressed,
        result.total_time_compressed,
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Original",
        x=metrics,
        y=original_values,
        marker_color="#636EFA"
    ))
    fig.add_trace(go.Bar(
        name="Compressed",
        x=metrics,
        y=compressed_values,
        marker_color="#00CC96"
    ))

    fig.update_layout(
        barmode="group",
        title="Performance Comparison",
        yaxis_title="Value",
        height=400,
    )

    return fig


def run_compression(
    compression_ratio: float,
    query_aware: bool,
    selected_example: str,
    custom_prompt: str,
    custom_query: str,
) -> Tuple[
        List[Tuple[str, str]],  # token_heatmap_data
        str,                   # baseline_context
        str,                   # baseline_query
        str,                   # baseline_answer
        str,                   # compressed_context
        str,                   # compressed_query
        str,                   # compressed_answer
        go.Figure,             # plot_data
        float,                 # kv_cache_metric
        float,                 # ttft_improvement
        float,                 # total_time_saved
    ]:
    """
    运行压缩并返回所有结果

    Args:
        compression_ratio: 压缩率
        query_aware: query-aware模式
        selected_example: 选中的示例名称
        custom_prompt: 自定义prompt
        custom_query: 自定义query

    Returns:
        (token_heatmap_data, baseline_context, baseline_query, baseline_answer,
         compressed_context, compressed_query, compressed_answer,
         plot_data, kv_cache_metric, ttft_improvement, total_time_saved)
    """
    # 获取输入（优先使用自定义输入）
    # 真实接口需要 context: List[str]
    if custom_prompt:
        context = [custom_prompt]
        query = custom_query or "What is the main point?"
    else:
        example = get_example(selected_example)
        if example is None:
            return [], "", "", "", "", "", "", go.Figure(), 0.0, 0.0, 0.0
        context = example["context"]  # List[str]
        query = example.get("query", "What is the main point?")

    # 调用压缩器
    compressor = MockCompressor()
    result = compressor.compress(
        context=context,
        rate=compression_ratio,
        return_word_label=True,
        use_token_level_filter=True,
        query_aware=query_aware,
    )

    # 格式化输出
    token_heatmap_data = parse_labeled_original_to_highlighted(result.labeled_original)

    baseline_context_str = "\n".join(context)
    compressed_context_str = result.compressed_prompt

    plot_data = create_performance_plot(result)

    # 计算指标数值
    kv_cache_metric_val = ((result.origin_tokens - result.compressed_tokens) / result.origin_tokens * 100) if result.origin_tokens > 0 else 0
    ttft_improvement_val = ((result.ttft_original - result.ttft_compressed) / result.ttft_original * 100) if result.ttft_original > 0 else 0
    total_time_saved_val = ((result.total_time_original - result.total_time_compressed) / result.total_time_original * 100) if result.total_time_original > 0 else 0

    return (
        token_heatmap_data,
        baseline_context_str,
        query,
        result.baseline_answer,
        compressed_context_str,
        query,
        result.compressed_answer,
        plot_data,
        kv_cache_metric_val,
        ttft_improvement_val,
        total_time_saved_val,
    )


def on_example_change(example_name: str):
    """当选择示例时，返回该示例的context和query用于自定义输入区域"""
    example = get_example(example_name)
    if example is None:
        return "", ""
    return "\n".join(example["context"]), example.get("query", "")


# 创建Gradio界面
with gr.Blocks(title="Prompt Compressor Demo") as demo:
    gr.Markdown(
        """
        # 🗜️ Prompt Compressor Interactive Demo

        可视化展示Prompt压缩算法的效果。调整压缩率，观察Token级别的决策过程，
        比较压缩前后的QA质量，以及性能收益。
        """
    )

    with gr.Row():
        # 侧边栏 - 控制面板
        with gr.Sidebar():
            gr.Markdown("### 控制面板")

            # 压缩率滑块
            compression_ratio = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="压缩率",
                info="保留的Token比例",
                interactive=True,
            )

            # Query-Aware开关
            query_aware = gr.Checkbox(
                label="**Query-Aware 模式**",
                value=True,
                info="根据Query动态调整压缩策略",
                interactive=True,
                elem_id="query-aware-checkbox",
            )

            # 示例数据选择
            gr.Markdown("### 示例数据")
            example_selector = gr.Dropdown(
                choices=EXAMPLE_CHOICES,
                value=EXAMPLE_CHOICES[0],
                label="选择示例",
                info="快速加载预设的Prompt+Query组合",
                interactive=True,
            )

            # 手动输入区域
            with gr.Accordion("自定义输入", open=False):
                prompt_input = gr.Textbox(
                    lines=8,
                    label="原始Prompt",
                    placeholder="输入待压缩的Prompt文本...",
                    interactive=True,
                )
                query_input = gr.Textbox(
                    label="用户Query",
                    placeholder="输入用户查询...",
                    interactive=True,
                )

            run_btn = gr.Button("运行压缩", variant="primary", size="lg")

        # 主面板
        with gr.Column():
            # 面板一：Token权重热力图
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Token 权重可视化")
                    gr.Markdown(
                        "* 🟡 黄色=命名实体 | 🟢 绿色=关键保留 | ⬜ 浅灰=冗余删除*"
                    )

                    token_heatmap = gr.HighlightedText(
                        label="Token级别压缩决策",
                        color_map={
                            "entity": "#FFD700",  # 黄色 - 命名实体
                            "keep": "#4CAF50",    # 绿色 - 关键保留
                            "drop": "#F5F5F5",    # 浅灰 - 冗余删除
                        },
                        show_legend=True,
                        combine_adjacent=True,
                        interactive=False,
                    )

            # 面板二：QA质量双栏对比
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Baseline (原始)", elem_classes="panel-header")
                    baseline_context = gr.Textbox(
                        label="Original Context",
                        lines=4,
                        interactive=False,
                    )
                    baseline_query = gr.Textbox(label="Query", interactive=False)
                    baseline_answer = gr.Textbox(
                        label="Generated Answer",
                        lines=6,
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📦 Compressed (压缩后)", elem_classes="panel-header")
                    compressed_context = gr.Textbox(
                        label="Compressed Context",
                        lines=4,
                        interactive=False,
                    )
                    compressed_query = gr.Textbox(label="Query", interactive=False)
                    compressed_answer = gr.Textbox(
                        label="Generated Answer",
                        lines=6,
                        interactive=False,
                    )

            # 底部面板：性能指标对比
            with gr.Column():
                gr.Markdown("### 📊 性能指标对比")

                # Plotly柱状图
                performance_chart = gr.Plot(label="性能指标对比")

                # 数值指标
                with gr.Row():
                    kv_cache_metric = gr.Number(
                        label="KV Cache节省 (%)",
                        precision=2,
                        interactive=False,
                    )
                    ttft_improvement = gr.Number(
                        label="TTFT改善 (%)",
                        precision=2,
                        interactive=False,
                    )
                    total_time_saved = gr.Number(
                        label="总时间节省 (%)",
                        precision=2,
                        interactive=False,
                    )

    # 事件处理 - 实时更新
    compression_ratio.change(
        fn=run_compression,
        inputs=[compression_ratio, query_aware, example_selector, prompt_input, query_input],
        outputs=[
            token_heatmap,
            baseline_context,
            baseline_query,
            baseline_answer,
            compressed_context,
            compressed_query,
            compressed_answer,
            performance_chart,
            kv_cache_metric,
            ttft_improvement,
            total_time_saved,
        ],
        show_progress="minimal",
    )

    query_aware.change(
        fn=run_compression,
        inputs=[compression_ratio, query_aware, example_selector, prompt_input, query_input],
        outputs=[
            token_heatmap,
            baseline_context,
            baseline_query,
            baseline_answer,
            compressed_context,
            compressed_query,
            compressed_answer,
            performance_chart,
            kv_cache_metric,
            ttft_improvement,
            total_time_saved,
        ],
        show_progress="minimal",
    )

    example_selector.change(
        fn=on_example_change,
        inputs=[example_selector],
        outputs=[prompt_input, query_input],
    ).then(
        fn=run_compression,
        inputs=[compression_ratio, query_aware, example_selector, prompt_input, query_input],
        outputs=[
            token_heatmap,
            baseline_context,
            baseline_query,
            baseline_answer,
            compressed_context,
            compressed_query,
            compressed_answer,
            performance_chart,
            kv_cache_metric,
            ttft_improvement,
            total_time_saved,
        ],
        show_progress="minimal",
    )

    prompt_input.input(
        fn=run_compression,
        inputs=[compression_ratio, query_aware, example_selector, prompt_input, query_input],
        outputs=[
            token_heatmap,
            baseline_context,
            baseline_query,
            baseline_answer,
            compressed_context,
            compressed_query,
            compressed_answer,
            performance_chart,
            kv_cache_metric,
            ttft_improvement,
            total_time_saved,
        ],
        show_progress="minimal",
    )

    query_input.input(
        fn=run_compression,
        inputs=[compression_ratio, query_aware, example_selector, prompt_input, query_input],
        outputs=[
            token_heatmap,
            baseline_context,
            baseline_query,
            baseline_answer,
            compressed_context,
            compressed_query,
            compressed_answer,
            performance_chart,
            kv_cache_metric,
            ttft_improvement,
            total_time_saved,
        ],
        show_progress="minimal",
    )

    run_btn.click(
        fn=run_compression,
        inputs=[compression_ratio, query_aware, example_selector, prompt_input, query_input],
        outputs=[
            token_heatmap,
            baseline_context,
            baseline_query,
            baseline_answer,
            compressed_context,
            compressed_query,
            compressed_answer,
            performance_chart,
            kv_cache_metric,
            ttft_improvement,
            total_time_saved,
        ],
        show_progress="full",
    )

    # 初始化运行
    demo.load(
        fn=run_compression,
        inputs=[compression_ratio, query_aware, example_selector, prompt_input, query_input],
        outputs=[
            token_heatmap,
            baseline_context,
            baseline_query,
            baseline_answer,
            compressed_context,
            compressed_query,
            compressed_answer,
            performance_chart,
            kv_cache_metric,
            ttft_improvement,
            total_time_saved,
        ],
    )


if __name__ == "__main__":
    # 默认绑定 localhost，避免直接访问 0.0.0.0 导致浏览器报错。
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        theme=gr.themes.Soft(),
    )
