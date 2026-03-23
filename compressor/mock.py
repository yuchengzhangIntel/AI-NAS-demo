"""
Mock Compressor for demonstration.

提供模拟的压缩器实现，用于演示UI效果。
"""

from typing import List

from .base import CompressionResult, PromptCompressor


class MockCompressor(PromptCompressor):
    """Mock压缩器实现，用于演示"""

    def compress(
        self,
        context: List[str],
        rate: float = 0.5,
        target_token: int = -1,
        use_context_level_filter: bool = False,
        use_token_level_filter: bool = True,
        return_word_label: bool = True,
        query_aware: bool = True,
    ) -> CompressionResult:
        """
        模拟压缩过程

        Args:
            context: 输入的上下文文本列表
            rate: 压缩率 (保留的比例)
            query_aware: query-aware模式（Mock中暂未实际使用）

        Returns:
            CompressionResult - 模拟的压缩结果
        """
        # 合并所有context
        combined_prompt = "\n".join(context)
        tokens = combined_prompt.split()

        if not tokens:
            return CompressionResult(
                compressed_prompt="",
                compressed_prompt_list=[],
                labeled_original="",
                origin_tokens=0,
                compressed_tokens=0,
                ratio=rate,
                baseline_answer="",
                compressed_answer="",
                ttft_original=0.0,
                ttft_compressed=0.0,
                total_time_original=0.0,
                total_time_compressed=0.0,
            )

        # 计算压缩后的token数量
        origin_token_count = len(tokens)
        compressed_token_count = int(origin_token_count * rate)

        # 生成压缩后的prompt
        compressed_tokens_list = tokens[:compressed_token_count]
        compressed_prompt = " ".join(compressed_tokens_list)

        # 生成带标签的原始文本
        # 格式: "word1 1\t\t|\t\tword2 0\t\t|\t\tword3 1..."
        # 1=保留, 0=删除
        labeled_original = self._generate_labeled_output(tokens, rate)

        # 生成压缩后的prompt列表（用于显示）
        compressed_prompt_list = [c[:int(len(c) * rate)] for c in context]

        # 模拟性能数据
        # TTFT与token数量成正比
        ttft_original = origin_token_count * 0.005 + 0.5
        ttft_compressed = compressed_token_count * 0.005 + 0.3

        # 总时间
        total_time_original = ttft_original + origin_token_count * 0.01
        total_time_compressed = ttft_compressed + compressed_token_count * 0.01

        # 模拟回答（基于原始回答进行压缩）
        baseline_answer = self._generate_mock_answer(context)
        compressed_answer = self._compress_answer(baseline_answer, rate)

        return CompressionResult(
            compressed_prompt=compressed_prompt,
            compressed_prompt_list=compressed_prompt_list,
            labeled_original=labeled_original,
            origin_tokens=origin_token_count,
            compressed_tokens=compressed_token_count,
            ratio=rate,
            baseline_answer=baseline_answer,
            compressed_answer=compressed_answer,
            ttft_original=ttft_original,
            ttft_compressed=ttft_compressed,
            total_time_original=total_time_original,
            total_time_compressed=total_time_compressed,
        )

    def _generate_labeled_output(self, tokens: List[str], rate: float) -> str:
        """
        生成带标签的原始文本

        格式: word label_sep label word_sep word label_sep label...
        真实接口中 label_sep=" ", word_sep="\t\t|\t\t"
        """
        words_with_labels = []
        threshold = int(len(tokens) * rate)

        for i, token in enumerate(tokens):
            # 简单规则：前threshold比例保留，命名实体也保留
            if i < threshold:
                label = "1"  # 保留
            elif token and token[0].isupper() and len(token) > 2:
                label = "1"  # 命名实体保留
            else:
                label = "0"  # 删除

            words_with_labels.append(f"{token} {label}")

        return "\t\t|\t\t".join(words_with_labels)

    def _generate_mock_answer(self, context: List[str]) -> str:
        """根据上下文生成模拟回答"""
        full_context = " ".join(context)
        if not full_context:
            return ""

        # 简单的模拟：返回上下文的前半部分总结
        words = full_context.split()
        if len(words) > 50:
            summary = " ".join(words[:40]) + "..."
        else:
            summary = full_context

        # 根据内容选择回答模板
        if "Apple" in summary or "factory" in summary:
            return "Apple is planning to build a new factory in California. The factory will produce electric vehicle batteries and is expected to create 5,000 jobs. Construction aims to begin in early 2024."
        elif "iPhone" in summary or "features" in summary:
            return "The iPhone 15 Pro Max features a titanium design that is both durable and lightweight. It includes the new A17 Pro chip for enhanced performance and gaming capabilities. The device supports USB-C for faster charging and data transfer speeds up to 10 Gbps."
        elif "Tesla" in summary or "revenue" in summary:
            return "Tesla reported Q3 revenue of $23.35 billion, beating analyst expectations. The company delivered 435,059 vehicles during the quarter, representing a 27% year-over-year increase. Gross margin stood at 16.3%."
        else:
            return f"Based on the context: {summary}"

    def _compress_answer(self, answer: str, rate: float) -> str:
        """对回答进行压缩"""
        words = answer.split()
        target_words = int(len(words) * rate)
        if target_words < 10:
            target_words = 10  # 至少保留10个词
        return " ".join(words[:target_words])
