"""Causal faithfulness metrics via MLM-driven counterfactual generation.

Uses a Masked Language Model to generate contextually valid counterfactuals 
by masking salient tokens and sampling plausible alternatives.
"""

import numpy
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.utils.cuda import DEVICE

class MLMCounterfactualGenerator:
    """Generates counterfactuals using a Masked Language Model."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def generate(self, text: str, target_word: str, top_k: int = 5) -> str | None:
        """
        Replace `target_word` in `text` with a contextually plausible alternative 
        predicted by the MLM.
        """
        # 1. Tokenize and find position of target word
        # This is heuristic; precise alignment is hard without span info.
        # We search for the first occurrence of the word.
        
        # Naive tokenization check
        # We need to ensure we mask the *correct* tokens corresponding to the word.
        word_tokens = self.tokenizer.encode(target_word, add_special_tokens=False)
        if not word_tokens:
            return None
            
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"][0]
        
        # Search for the sequence of word_tokens
        # Sliding window search
        seq_len = len(word_tokens)
        match_idx = -1
        for i in range(len(input_ids) - seq_len + 1):
            if torch.equal(input_ids[i : i + seq_len], torch.tensor(word_tokens, device=DEVICE)):
                match_idx = i
                break
                
        if match_idx == -1:
            # Fallback: try case-insensitive
            # (Skipping complex alignment for efficiency in this simplified impl)
            return None

        # 2. Mask the tokens
        masked_input_ids = input_ids.clone()
        masked_input_ids[match_idx : match_idx + seq_len] = self.tokenizer.mask_token_id
        
        # 3. Predict
        with torch.no_grad():
            outputs = self.model(masked_input_ids.unsqueeze(0))
            logits = outputs.logits[0, match_idx] # Logits at the mask position (using first mask token if multi-token)
            
        # 4. Sample
        # We want a word that is NOT the original word.
        probs = F.softmax(logits, dim=-1)
        top_ids = torch.topk(probs, k=top_k + 1).indices
        
        for idx in top_ids:
            if idx not in word_tokens: # Check if it's different
                # Decode the new token
                new_word = self.tokenizer.decode([idx]).strip()
                # Basic filter: ignore subwords or special chars if possible
                if new_word.isalnum() and new_word.lower() != target_word.lower():
                    # Construct new text
                    # We replace the tokens in the input_ids and decode back
                    new_input_ids = input_ids.clone()
                    # For multi-token replacement, this is tricky. 
                    # Simpler: replace in string if we trust the word.
                    # Or replace just the first token and drop the others?
                    # Let's simple string replace for stability.
                    return text.replace(target_word, new_word, 1)
                    
        return None

def compute_causal_faithfulness(
    model,
    tokenizer,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    max_length: int,
) -> float:
    """
    Compute Causal Faithfulness using MLM Counterfactuals.
    """
    from splade.inference import predict_proba_model 
    
    # Initialize generator (singleton-ish)
    generator = MLMCounterfactualGenerator()
    
    shifts = []
    attrib_scores = []
    
    # Batch processing for original probabilities
    original_probs = predict_proba_model(model, tokenizer, texts, max_length)
    
    valid_samples = 0
    
    for i, text in enumerate(texts):
        if not attributions[i]:
            continue
            
        # Get top attributed token (normalized word)
        top_token = attributions[i][0][0]
        score = attributions[i][0][1]
        
        # Generate counterfactual
        new_text = generator.generate(text, top_token)
        
        if new_text:
            # Predict on new text
            new_prob = predict_proba_model(model, tokenizer, [new_text], max_length)[0]
            
            # Target class of original
            target = numpy.argmax(original_probs[i])
            
            # Shift magnitude
            shift = abs(original_probs[i][target] - new_prob[target])
            
            shifts.append(shift)
            attrib_scores.append(abs(score))
            valid_samples += 1
            
    if valid_samples < 2:
        return 0.0
        
    return float(numpy.corrcoef(attrib_scores, shifts)[0, 1])
