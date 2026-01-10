import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

class TransformerNeuralVectorizer(nn.Module):
    """
    SPLADE-inspired Neural Vectorizer.
    
    Uses a Transformer (DistilBERT) to predict sparse weights for the entire vocabulary.
    Performs Term Expansion and Weighting simultaneously.
    """
    def __init__(self, model_name='distilbert-base-uncased'):
        super(TransformerNeuralVectorizer, self).__init__()
        self.model_name = model_name
        
        # Load pre-trained MLM model (includes the Linear layer to project to vocab)
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            sparse_vec: [batch_size, vocab_size] - The document-level sparse vector.
        """
        # 1. Get Logits from Transformer [batch, seq_len, vocab_size]
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        
        # 2. Log-Saturation Activation (SPLADE formula)
        #    w_ij = log(1 + ReLU(logit_ij))
        #    This ensures non-negativity and saturation.
        weights = torch.log1p(F.relu(logits)) 
        
        # 3. Mask out padding tokens (set their weights to 0)
        #    attention_mask is [batch, seq_len] -> expand to [batch, seq_len, vocab_size]
        mask = attention_mask.unsqueeze(-1).expand_as(weights)
        weights = weights * mask
        
        # 4. Max Pooling over Sequence Length (Aggregation)
        #    Get the maximum weight for each term across all tokens in the document.
        #    [batch, seq_len, vocab_size] -> [batch, vocab_size]
        doc_vector, _ = torch.max(weights, dim=1)
        
        return doc_vector

class SparseClassifier(nn.Module):
    """
    Combines the TransformerNeuralVectorizer with a Linear Classifier.
    """
    def __init__(self, num_classes=1, model_name='distilbert-base-uncased'):
        super(SparseClassifier, self).__init__()
        self.vectorizer = TransformerNeuralVectorizer(model_name)
        
        # The input dimension to the classifier is the Transformer's vocabulary size
        # We need to get this size dynamically or hardcode for DistilBERT (30522)
        vocab_size = self.vectorizer.transformer.config.vocab_size
        self.classifier = nn.Linear(vocab_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # 1. Get sparse representation (SPLADE vector)
        sparse_vec = self.vectorizer(input_ids, attention_mask)
        
        # 2. Predict
        logits = self.classifier(sparse_vec)
        
        return logits, sparse_vec