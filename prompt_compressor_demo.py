"""
Prompt Compressor Interactive Demo

使用Gradio构建的交互式Demo，展示Prompt压缩算法的效果。
"""
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"


import gradio as gr
import html
import plotly.graph_objects as go
import re
from functools import lru_cache
from typing import List, Tuple

import spacy

from compressor import MockCompressor, CompressionResult
from data.examples import EXAMPLE_CHOICES, get_example


CUSTOM_CSS = """
#token-heatmap-html {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
    background: #ffffff;
}

#token-heatmap-html .token-visualization {
    white-space: pre-wrap;
    line-height: 1.9;
    font-size: 0.95rem;
    font-family: "Times New Roman", Times, serif;
}

#token-heatmap-html .token-span {
    border-radius: 0.3rem;
    padding: 0.03rem 0.08rem;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}

#token-heatmap-html .token-keep {
    background: rgba(76, 175, 80, 0.18);
}

#token-heatmap-html .token-drop {
    background: rgba(148, 163, 184, 0.18);
    color: #475569;
}

#token-heatmap-html .token-entity {
    display: inline-block;
    line-height: 1;
    padding-bottom: 0.08em;
    border-bottom: 2px solid #c2410c;
}

.times-display textarea,
.times-display input {
    font-family: "Times New Roman", Times, serif;
}
"""

WORD_SEPARATOR = "\t\t|\t\t"
TOKEN_PATTERN = re.compile(r"\S+|\s+")
CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def parse_labeled_original(labeled_original: str) -> List[Tuple[str, str]]:
    """
    解析真实接口返回的 fn_labeled_original_prompt。

    格式: "word1 1\t\t|\t\tword2 0\t\t|\t\tword3 1..."
    返回: [("word1", "keep"), ("word2", "drop"), ("word3", "keep")]
    """
    if not labeled_original:
        return []

    segments = labeled_original.split(WORD_SEPARATOR)

    result = []
    for segment in segments:
        if not segment.strip():
            continue

        parts = segment.rsplit(" ", 1)
        if len(parts) == 2:
            word, label = parts
            label_name = "keep" if label.strip() == "1" else "drop"
            result.append((word, label_name))

    return result


@lru_cache(maxsize=2)
def get_spacy_model(language: str):
    """按语言懒加载 spaCy 模型，未安装模型时退化为空模型。"""
    model_name = "zh_core_web_sm" if language == "zh" else "en_core_web_sm"
    blank_language = "zh" if language == "zh" else "en"

    try:
        return spacy.load(model_name, disable=["tagger", "parser", "lemmatizer", "textcat"])
    except OSError:
        return spacy.blank(blank_language)


def detect_spacy_language(text: str) -> str:
    """含中文字符时优先使用中文模型，否则使用英文模型。"""
    return "zh" if CHINESE_CHAR_PATTERN.search(text) else "en"


def get_entity_spans(text: str) -> List[Tuple[int, int]]:
    """返回实体的字符级区间。"""
    if not text:
        return []

    nlp = get_spacy_model(detect_spacy_language(text))
    doc = nlp(text)
    return [(ent.start_char, ent.end_char) for ent in doc.ents]


def align_labels_to_text(text: str, labeled_original: str) -> List[Tuple[str, str | None, int, int]]:
    """把 split() 产生的词标签重新对齐回原始文本，保留空格和换行。"""
    parsed_labels = parse_labeled_original(labeled_original)
    segments: List[Tuple[str, str | None, int, int]] = []
    label_index = 0

    for match in TOKEN_PATTERN.finditer(text):
        segment_text = match.group(0)
        label_name = None
        if not segment_text.isspace() and label_index < len(parsed_labels):
            _, label_name = parsed_labels[label_index]
            label_index += 1

        segments.append((segment_text, label_name, match.start(), match.end()))

    return segments


def is_entity_segment(start: int, end: int, entity_spans: List[Tuple[int, int]]) -> bool:
    """判断当前片段是否与任一实体区间重叠。"""
    return any(start < entity_end and end > entity_start for entity_start, entity_end in entity_spans)


def render_annotated_text_html(text: str, labeled_original: str) -> str:
    """渲染可叠加的底色与下划线标注 HTML。"""
    if not text:
        return "<div class=\"token-visualization\"></div>"

    segments = align_labels_to_text(text, labeled_original)
    entity_spans = get_entity_spans(text)

    html_segments: List[str] = []
    for segment_text, label_name, start, end in segments:
        escaped_text = html.escape(segment_text)
        is_entity = (not segment_text.isspace()) and is_entity_segment(start, end, entity_spans)

        if label_name is None and not is_entity:
            html_segments.append(escaped_text)
            continue

        classes = ["token-span"]
        if label_name is not None:
            classes.append(f"token-{label_name}")
        content = escaped_text
        if is_entity:
            content = f'<span class="token-entity">{escaped_text}</span>'

        html_segments.append(f'<span class="{" ".join(classes)}">{content}</span>')

    return f'<div class="token-visualization">{"".join(html_segments)}</div>'


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
    str,                    # token_heatmap_html
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
            return "", "", "", "", "", "", "", go.Figure(), 0.0, 0.0, 0.0
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
    baseline_context_str = "\n".join(context)
    token_heatmap_html = render_annotated_text_html(baseline_context_str, result.labeled_original)
    compressed_context_str = result.compressed_prompt

    plot_data = create_performance_plot(result)

    # 计算指标数值
    kv_cache_metric_val = ((result.origin_tokens - result.compressed_tokens) / result.origin_tokens * 100) if result.origin_tokens > 0 else 0
    ttft_improvement_val = ((result.ttft_original - result.ttft_compressed) / result.ttft_original * 100) if result.ttft_original > 0 else 0
    total_time_saved_val = ((result.total_time_original - result.total_time_compressed) / result.total_time_original * 100) if result.total_time_original > 0 else 0

    return (
        token_heatmap_html,
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
                label="Query-Aware 模式",
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
                        "* 下划线=命名实体 | 🟢 绿色底色=关键保留 | ⬜ 浅灰底色=冗余删除*"
                    )

                    token_heatmap = gr.HTML(
                        value="",
                        elem_id="token-heatmap-html",
                    )

            # 面板二：QA质量双栏对比
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Baseline (原始)", elem_classes="panel-header")
                    baseline_context = gr.Textbox(
                        label="Original Context",
                        lines=4,
                        interactive=False,
                        elem_classes="times-display",
                    )
                    baseline_query = gr.Textbox(
                        label="Query",
                        interactive=False,
                        elem_classes="times-display",
                    )
                    baseline_answer = gr.Textbox(
                        label="Generated Answer",
                        lines=6,
                        interactive=False,
                        elem_classes="times-display",
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📦 Compressed (压缩后)", elem_classes="panel-header")
                    compressed_context = gr.Textbox(
                        label="Compressed Context",
                        lines=4,
                        interactive=False,
                        elem_classes="times-display",
                    )
                    compressed_query = gr.Textbox(
                        label="Query",
                        interactive=False,
                        elem_classes="times-display",
                    )
                    compressed_answer = gr.Textbox(
                        label="Generated Answer",
                        lines=6,
                        interactive=False,
                        elem_classes="times-display",
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
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )
