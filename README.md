PyTorch version: 2.9.1+cu128
CUDA available: True
Loading tokenizer
Initializing model
Using device: cuda

Creating sample dataset...
Training samples: 160
Validation samples: 40


Starting Training
==================================================

Epoch 1/4
------------------------------
Train Loss: 0.6533, Train Acc: 0.6750
Val Loss: 0.5199, Val Acc: 1.0000
✓ Saved best model with accuracy: 1.0000

Epoch 2/4
------------------------------
Train Loss: 0.3819, Train Acc: 0.9813
Val Loss: 0.1939, Val Acc: 1.0000

Epoch 3/4
------------------------------
Train Loss: 0.1236, Train Acc: 1.0000
Val Loss: 0.0497, Val Acc: 1.0000

Epoch 4/4
------------------------------
Train Loss: 0.0367, Train Acc: 1.0000
Val Loss: 0.0165, Val Acc: 1.0000

==================================================
Final Evaluation
==================================================

Best Model Performance:
Accuracy: 1.0000
Loss: 0.5199

Classification Report:
              precision    recall  f1-score   support

    negative       1.00      1.00      1.00        20
    positive       1.00      1.00      1.00        20

    accuracy                           1.00        40
   macro avg       1.00      1.00      1.00        40
weighted avg       1.00      1.00      1.00        40


==================================================
Test Predictions
==================================================

1. I really loved this movie! It was amazing.
   Sentiment: positive (confidence: 56.86%)
   Probabilities: Negative=0.4314, Positive=0.5686

2. Terrible film, complete waste of time.
   Sentiment: negative (confidence: 63.83%)
   Probabilities: Negative=0.6383, Positive=0.3617

3. It was okay, not great but not bad either.
   Sentiment: negative (confidence: 52.66%)
   Probabilities: Negative=0.5266, Positive=0.4734

4. Absolutely brilliant! Best movie of the year.
   Sentiment: positive (confidence: 54.98%)
   Probabilities: Negative=0.4502, Positive=0.5498

5. Mediocre at best, wouldn't watch again.
   Sentiment: negative (confidence: 60.97%)
   Probabilities: Negative=0.6097, Positive=0.3903

6. Fantastic performances and great storytelling.
   Sentiment: positive (confidence: 56.99%)
   Probabilities: Negative=0.4301, Positive=0.5699

7. Boring and predictable, very disappointing.
   Sentiment: negative (confidence: 62.82%)
   Probabilities: Negative=0.6282, Positive=0.3718

8. A cinematic masterpiece that moved me deeply.
   Sentiment: positive (confidence: 53.87%)
   Probabilities: Negative=0.4613, Positive=0.5387

Saving model assets...

==================================================
Training Complete!
Best Accuracy: 1.0000
Model saved as: distilbert_sentiment_final.pth
Tokenizer saved in: ./saved_tokenizer/
==================================================
