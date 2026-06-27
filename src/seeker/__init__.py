"""CandleSeeker — Chinese NLP corpus collector and annotator."""

__version__ = "0.7.1"
__author__ = "Candle Seeker Maintainers"
__all__ = [
    "CorpusCrawler",
    "DomainClassifier",
    "ChineseTokenizer",
    "CorpusDocument",
    "DomainLabel",
    "SeekerConfig",
]

from .crawler import CorpusCrawler, CorpusDocument, SeekerConfig
from .classifier import DomainClassifier, DomainLabel
from .tokenizer import ChineseTokenizer
