# test_imports.py
import torch
import transformers
import sklearn

print("="*50)
print("Dependency Check")
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test DistilBERT imports
from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

print("\n✓ DistilBERT imports successful!")
print(f"Tokenizer loaded: {type(tokenizer)}")
print(f"Model loaded: {type(model)}")