import time
import torch
import numpy as np
import argparse
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from src.neural_vectorizer import SparseClassifier
from src.data_loader import get_data_loaders
from src.evaluation import evaluate
from src.utils import load_stopwords, simple_tokenizer
from src.config import DEFAULT_BATCH_SIZE
import pandas as pd

def train_sklearn_baseline(train_path, test_path):
    print("\n--- Training Sklearn Baseline ---")
    
    # Load raw text
    train_df = pd.read_csv(train_path, sep='\t', header=None, names=['id', 'review', 'label'])
    test_df = pd.read_csv(test_path, sep='\t', header=None, names=['id', 'review', 'label'])
    
    start_time = time.time()
    
    # 1. Vectorize
    stopwords = load_stopwords()
    # Wrap tokenizer to fix stopwords argument
    tokenizer_func = lambda text: simple_tokenizer(text, stopwords)
    
    # We use token_pattern=None because we provide a custom tokenizer
    vectorizer = TfidfVectorizer(max_features=1000, tokenizer=tokenizer_func, token_pattern=None)
    X_train = vectorizer.fit_transform(train_df['review'])
    X_test = vectorizer.transform(test_df['review'])
    y_train = train_df['label']
    y_test = test_df['label']
    
    # 2. Train Classifier
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # 3. Inference & Metrics
    start_inf = time.time()
    preds = clf.predict(X_test)
    inf_time = time.time() - start_inf
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    # Calculate Sparsity (of the feature matrix)
    sparsity = 1.0 - (X_test.nnz / (X_test.shape[0] * X_test.shape[1]))
    
    print(f"Sklearn Accuracy: {acc:.4f}")
    print(f"Sklearn F1: {f1:.4f}")
    print(f"Inference Latency: {inf_time*1000:.2f} ms")
    print(f"Sparsity: {sparsity*100:.2f}%")
    
    return {
        "model": "Sklearn TF-IDF",
        "accuracy": acc,
        "f1": f1,
        "latency_ms": inf_time * 1000,
        "sparsity": sparsity * 100
    }

def evaluate_neural_model(model_path, data_dir, batch_size=DEFAULT_BATCH_SIZE):
    print(f"\n--- Evaluating Neural Model ({model_path}) ---")
    
    # Load Data
    _, test_loader = get_data_loaders(
        os.path.join(data_dir, 'movie_reviews_train.txt'),
        os.path.join(data_dir, 'movie_reviews_test.txt'),
        batch_size=batch_size
    )
    
    # Load Model
    model = SparseClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print("Model file not found. Skipping neural eval.")
        return None
        
    model.eval()
    
    start_time = time.time()
    # Using threshold=1e-2 to match training logic
    acc, f1, sparsity = evaluate(model, test_loader, device, threshold=1e-2)
    total_latency = time.time() - start_time
    
    print(f"Neural Accuracy: {acc:.4f}")
    print(f"Neural F1: {f1:.4f}")
    print(f"Total Inference Latency (incl. metrics): {total_latency*1000:.2f} ms")
    print(f"Sparsity: {sparsity:.2f}%")
    
    return {
        "model": "Neural Sparse (SPLADE)",
        "accuracy": acc,
        "f1": f1,
        "latency_ms": total_latency * 1000,
        "sparsity": sparsity
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data')
    parser.add_argument('--model_path', type=str, default='models/model.pth')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    
    results = []
    
    # 1. Run Baseline
    results.append(train_sklearn_baseline(
        os.path.join(args.data_dir, 'movie_reviews_train.txt'),
        os.path.join(args.data_dir, 'movie_reviews_test.txt')
    ))
    
    # 2. Run Ours
    res = evaluate_neural_model(args.model_path, args.data_dir, batch_size=args.batch_size)
    if res:
        results.append(res)
        
    # 3. Print Summary
    print("\n=== Final Benchmark Results ===")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))