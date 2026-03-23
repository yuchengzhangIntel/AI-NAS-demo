"""
Prompt Compressor Module

提供Mock实现和真实LLMLingua2集成接口。
"""

from .base import PromptCompressor, CompressionResult
from .mock import MockCompressor

__all__ = ["PromptCompressor", "CompressionResult", "MockCompressor"]
