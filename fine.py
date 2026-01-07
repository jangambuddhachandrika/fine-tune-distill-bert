import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ==================== Dataset Class ====================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==================== Model Class ====================
class DistilBertSentimentClassifier(nn.Module):
    def __init__(self, n_classes=2, dropout_prob=0.3):
        super(DistilBertSentimentClassifier, self).__init__()
        
        # Load pre-trained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze DistilBERT layers (optional - can unfreeze later)
        # for param in self.distilbert.parameters():
        #     param.requires_grad = False
        
        # Get hidden size
        hidden_size = self.distilbert.config.hidden_size
        
        # Classification layers
        self.dropout = nn.Dropout(dropout_prob)
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask, labels=None):
        # DistilBERT forward pass
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract hidden states
        hidden_state = distilbert_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Use the [CLS] token for classification
        pooled_output = hidden_state[:, 0]  # (batch_size, hidden_size)
        
        # Classification head
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}

# ==================== Create Sample Data ====================
def create_sample_data(n_samples=200):
    """Create synthetic sentiment analysis data"""
    positive_samples = [
        "I absolutely loved this movie! It was fantastic.",
        "Highly recommended, one of the best films I've seen.",
        "Excellent performance by the actors, very engaging.",
        "Wonderful storyline with great character development.",
        "A masterpiece that deserves all the awards.",
        "I was thoroughly entertained from start to finish.",
        "Brilliant direction and cinematography.",
        "This film exceeded all my expectations.",
        "Heartwarming and inspiring story.",
        "Perfect for family viewing, everyone enjoyed it."
    ]
    
    negative_samples = [
        "Terrible movie, complete waste of time.",
        "I would not recommend this to anyone.",
        "Poor acting and boring storyline.",
        "Very disappointing, had high expectations.",
        "The worst film I've seen this year.",
        "Awful direction and editing.",
        "Couldn't wait for it to end, so boring.",
        "Not worth the money or time.",
        "Painful to watch, terrible dialogue.",
        "I regret watching this film."
    ]
    
    # Create balanced dataset
    texts = []
    labels = []
    
    for i in range(n_samples // 2):
        # Positive samples
        texts.append(positive_samples[i % len(positive_samples)] + f" Sample {i}")
        labels.append(1)
        
        # Negative samples
        texts.append(negative_samples[i % len(negative_samples)] + f" Sample {i}")
        labels.append(0)
    
    return texts, labels

# ==================== Training Functions ====================
def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model.train()
    losses = []
    correct_predictions = 0
    
    progress_bar = tqdm(data_loader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        logits = outputs['logits']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Calculate accuracy
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': np.mean(losses[-10:]),
            'acc': torch.sum(preds == labels).item() / len(labels)
        })
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), all_preds, all_labels

# ==================== Prediction Function ====================
def predict_sentiment(text, model, tokenizer, device, max_len=128):
    model.eval()
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    logits = outputs['logits']
    probs = torch.softmax(logits, dim=1)
    _, prediction = torch.max(probs, dim=1)
    
    sentiment = "positive" if prediction.item() == 1 else "negative"
    confidence = probs[0][prediction.item()].item()
    
    return sentiment, confidence, probs.cpu().numpy()[0]

# ==================== Main Execution ====================
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize tokenizer and model
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    print("Initializing model...")
    model = DistilBertSentimentClassifier(n_classes=2)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Create sample data
    print("\nCreating sample dataset...")
    texts, labels = create_sample_data(n_samples=200)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Create datasets and dataloaders
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 4
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    best_accuracy = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        # Train
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        val_acc, val_loss, val_preds, val_labels = eval_model(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best_distilbert_sentiment.pth')
            print(f"✓ Saved best model with accuracy: {val_acc:.4f}")
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    # Load best model
    model.load_state_dict(torch.load('best_distilbert_sentiment.pth'))
    val_acc, val_loss, val_preds, val_labels = eval_model(model, val_loader, device)
    
    print(f"\nBest Model Performance:")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Loss: {val_loss:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=['negative', 'positive']))
    
    # Test predictions
    print("\n" + "="*50)
    print("Test Predictions")
    print("="*50)
    
    test_texts = [
        "I really loved this movie! It was amazing.",
        "Terrible film, complete waste of time.",
        "It was okay, not great but not bad either.",
        "Absolutely brilliant! Best movie of the year.",
        "Mediocre at best, wouldn't watch again.",
        "Fantastic performances and great storytelling.",
        "Boring and predictable, very disappointing.",
        "A cinematic masterpiece that moved me deeply."
    ]
    
    for i, text in enumerate(test_texts):
        sentiment, confidence, probs = predict_sentiment(text, model, tokenizer, device)
        print(f"\n{i+1}. {text}")
        print(f"   Sentiment: {sentiment} (confidence: {confidence:.2%})")
        print(f"   Probabilities: Negative={probs[0]:.4f}, Positive={probs[1]:.4f}")
    
    # Save final model and tokenizer
    print("\nSaving model assets...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_classes': 2,
            'dropout_prob': 0.3
        }
    }, 'distilbert_sentiment_final.pth')
    
    tokenizer.save_pretrained('./saved_tokenizer/')
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("Model saved as: distilbert_sentiment_final.pth")
    print("Tokenizer saved in: ./saved_tokenizer/")
    print("="*50)

if __name__ == "__main__":
    main()