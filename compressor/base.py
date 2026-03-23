"""
Base classes for Prompt Compressor.

定义压缩器的抽象接口和结果类型。
"""

from abc import ABC, abstractmethod
from typing import List, NamedTuple


class CompressionResult(NamedTuple):
    """压缩结果数据结构"""
    compressed_prompt: str           # 压缩后的prompt
    compressed_prompt_list: List[str] # 压缩后的prompt列表
    labeled_original: str             # 原始词+标签 (word label_sep label)
    origin_tokens: int                # 原始token数
    compressed_tokens: int            # 压缩后token数
    ratio: float                      # 压缩比例
    baseline_answer: str               # 基线回答 (mock/ollama)
    compressed_answer: str            # 压缩后回答 (mock/ollama)
    ttft_original: float             # 原始TTFT
    ttft_compressed: float            # 压缩后TTFT
    total_time_original: float        # 原始总时间
    total_time_compressed: float      # 压缩后总时间


class PromptCompressor(ABC):
    """Prompt压缩器抽象基类"""

    @abstractmethod
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
        压缩给定的context

        Args:
            context: List[str] - 输入的上下文文本列表
            rate: float - 目标压缩率 (0.1-0.9)
            target_token: int - 目标token数 (-1表示使用rate)
            use_context_level_filter: bool - 是否使用上下文级过滤
            use_token_level_filter: bool - 是否使用token级过滤
            return_word_label: bool - 是否返回词级标签
            query_aware: bool - 是否启用query-aware模式

        Returns:
            CompressionResult - 压缩结果
        """
        pass
