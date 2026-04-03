# Prompt Compressor Interactive Demo

这是一个展示 Prompt 压缩算法效果的 Gradio 交互式 Demo 项目。通过对原文进行切分、Token 级别的去重和实体识别保留，减少冗余上下文以提升大模型推理速度，同时可视化对比压缩前后的 Token 决策、回答质量及性能指标（KV Cache、TTFT、总时延）。项目现阶段使用了 mock 数据驱动。

## 快速运行

本项目使用 Python 构建。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 spaCy 语言模型

本项目使用了 spaCy 模型以高亮命名实体（NER）。默认使用英文或中文的小型模型。在运行项目前，需提前下载这些模型：

```bash
# 下载英文处理模型
python -m spacy download en_core_web_sm

# 下载中文处理模型
python -m spacy download zh_core_web_sm
```

### 3. 启动展示页面

```bash
python prompt_compressor_demo.py
```
默认会在 `localhost:7860` 启动网页端。

---

## 下一步工作：接入真实服务

当前版本压缩逻辑由 `compressor/mock.py` 承担。如果需要接入真实的“词法压缩服务（LLMLingua2等）”和“Ollama 本地 LLM 生成”，需要实现并替换这部分 mock，主要涉及下述接口对齐。

### 1. 压缩服务（TTC）

已有函数签名参考如下，你需要将其封装为一个 HTTP 请求链路：
```python
def compress_prompt_llmlingua2(
    self,
    context: List[str],
    rate: float = 0.5,
    target_token: int = -1,
    use_context_level_filter: bool = False,
    use_token_level_filter: bool = True,
    target_context: int = -1,
    context_level_rate: float = 1.0,
    context_level_target_token: int = -1,
    force_context_ids: List[int] = [],
    return_word_label: bool = False,
    word_sep: str = "\t\t|\t\t",
    label_sep: str = " ",
    token_to_word: str = "mean",
    force_tokens: List[str] = [],
    force_reserve_digit: bool = False,
    drop_consecutive: bool = False,
    chunk_end_tokens: List[str] = [".", "\n"],
)
```

**待确认细节：**
- **地址与协议：** 压缩服务（TTC）部署在网络什么位置？是怎样的 HTTP 接口（URL、请求体JSON和返回JSON样例）？
- **Query 传参规则：** 当用户在页面取消勾选 `Query-Aware 模式` 并且留空 Query 时，请求这个服务到底传什么？（传空字符串还是忽略该字段？）
- **返回体解析：** 目前 `parse_labeled_original` 假设压缩服务返回的标注文本是由 `\t\t|\t\t` 与 ` ` 分割的字符串（用 1 表示保留，0 表示废弃）。真实联调时如果返回词和标签分离的 JSON 数组，解析函数需要一并修改。

### 2. Ollama 本地生成服务 (LLM Generator)

需要在前端通过 Python 的 `requests`（或直接使用浏览器 JS）控制逻辑：
- 如果没有做压缩：直接只调一次 Ollama
- 如果做了压缩：先调用压缩服务拿到结果 > 原样显示在压缩侧 > 拿压缩结果调用 Ollama

**待确认细节：**
- **Ollama 地址与模型：** 配置如 `http://localhost:11434/api/generate` 及你使用的轻量语言模型名（如 `llama3`, `qwen2` 等）。
- **指标统计：** UI 底部有 TTFT 和 KV Cache 的雷达对比视图。这要求无论是走 Ollama 还是压缩服务（TTC），返回的 HTTP Header 或 Body 中得有这部分统计数据（如 `eval_count`, `prompt_eval_duration`），或者需要自己在 Python 发请求前后分别埋点打时间戳计时计算。如果没有对应指标且不自己埋点，只能在界面上隐藏这一面板。

---

## 开发参考

后续编写或扩展 Gradio 前端代码时，推荐先阅读代码库自带的开发指南：[`.agents/skills/gradio/SKILL.md`](.agents/skills/gradio/SKILL.md)。该文档涵盖了 Gradio 的核心组件用法、自定义 HTML/CSS 嵌入技巧以及事件监听等高级用法。