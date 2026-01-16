"""
Utility functions for the SPLADE project.

Provides common utilities for:
- Reproducibility (seed setting)
- Text processing (tokenization, stopwords)
- Device management
- Data source validation
- Statistical analysis (confidence intervals, significance tests)
"""

import random
import numpy as np
import torch

from .text import load_stopwords, simple_tokenizer
from .common import get_device, validate_data_sources
from .statistics import (
    StatisticalResult,
    McNemarResult,
    bootstrap_ci,
    mcnemar_test,
    paired_t_test,
    aggregate_results,
    effect_size_cohens_d,
)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = [
    "set_seed",
    "load_stopwords",
    "simple_tokenizer",
    "get_device",
    "validate_data_sources",
    "StatisticalResult",
    "McNemarResult",
    "bootstrap_ci",
    "mcnemar_test",
    "paired_t_test",
    "aggregate_results",
    "effect_size_cohens_d",
]
