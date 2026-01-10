import torch
import pytest
import sys
import os

# Add the project root to the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_vectorizer import NeuralVectorizer

def test_initialization():
    vocab_size = 100
    model = NeuralVectorizer(vocab_size)
    assert model.term_weights.weight.shape == (vocab_size, vocab_size)
    assert model.term_weights.bias.shape == (vocab_size,)

def test_forward_pass_dimensions():
    vocab_size = 50
    batch_size = 4
    model = NeuralVectorizer(vocab_size)
    
    # Create a random Bag-of-Words input (batch of term counts)
    dummy_input = torch.randn(batch_size, vocab_size).abs() # absolute to simulate counts
    
    output = model(dummy_input)
    
    assert output.shape == (batch_size, vocab_size)

def test_forward_pass_values():
    """Test the forward pass logic by manually setting weights."""
    vocab_size = 10
    model = NeuralVectorizer(vocab_size)
    
    # Manually set weights to ones and bias to zero for deterministic testing
    with torch.no_grad():
        model.term_weights.weight.fill_(1.0)
        model.term_weights.bias.fill_(0.0)
    
    # Input: All ones
    input_tensor = torch.ones(1, vocab_size)
    
    # Forward pass: 
    # Input (1x10) @ Weight.T (10x10) + Bias (10)
    # Since Input is all 1s and Weight is all 1s:
    # Each output neuron i receives sum(input_j * weight_ji) = sum(1 * 1) over j=0..9 = 10.0
    output = model(input_tensor)
    
    # Expected value: sum of 1s (10) for each output unit
    # But now we apply log1p: log(1 + 10)
    linear_output = 10.0
    expected_val = torch.log1p(torch.tensor(linear_output))
    
    assert torch.allclose(output, torch.full((1, vocab_size), expected_val.item()), atol=1e-5)

if __name__ == "__main__":
    test_initialization()
    test_forward_pass_dimensions()
    test_forward_pass_values()
    print("All tests passed!")
