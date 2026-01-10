import torch
import pytest
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import TextClassificationDataset

def test_dataset_loading(tmp_path):
    # 1. Create a dummy TSV file
    d = tmp_path / "dummy_data.txt"
    content = "ID-1\tThis is a good movie\t1\nID-2\tThis is a bad movie\t0\n"
    d.write_text(content, encoding='utf-8')
    
    # 2. Initialize Dataset
    dataset = TextClassificationDataset(str(d), max_vocab_size=10)
    
    # 3. Check Length
    assert len(dataset) == 2
    
    # 4. Check Vocabulary
    # "this", "is", "a" are stopwords and removed.
    # Remaining: "good", "movie", "bad", "movie"
    # Unique: "good", "bad", "movie" -> 3
    assert dataset.vocab_size == 3
    assert "good" in dataset.vocabulary
    assert "bad" in dataset.vocabulary
    
    # 5. Check Item Retrieval
    item = dataset[0]
    vector = item['vector']
    label = item['label']
    
    assert isinstance(vector, torch.Tensor)
    assert vector.shape == (3,)
    assert label == 1.0
    
    # Check that "good" count is 1
    good_idx = dataset.vocabulary["good"]
    assert vector[good_idx] == 1.0

if __name__ == "__main__":
    pytest.main([__file__])
