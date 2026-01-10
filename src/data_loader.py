import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
from typing import Dict, Tuple

class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for text classification using Transformer Tokenizer.
    """
    def __init__(self, file_path: str, model_name: str = 'distilbert-base-uncased', max_len: int = 128):
        """
        Args:
            file_path (str): Path to the TSV file (no header: ID, Review, Label).
            model_name (str): HuggingFace model name for the tokenizer.
            max_len (int): Maximum sequence length for truncation/padding.
        """
        self.data = self._load_data(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        
    def _load_data(self, file_path: str) -> pd.DataFrame:
        # The file format is: ID <tab> Review <tab> Label
        df = pd.read_csv(file_path, sep='\t', header=None, names=['id', 'review', 'label'])
        return df

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.data.iloc[idx]['review'])
        label = self.data.iloc[idx]['label']
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float32)
        }

def get_data_loaders(train_path: str, test_path: str, batch_size: int = 32, max_len: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Helper to create Train and Test loaders.
    """
    train_dataset = TextClassificationDataset(train_path, max_len=max_len)
    test_dataset = TextClassificationDataset(test_path, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader