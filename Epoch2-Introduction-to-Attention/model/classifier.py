import torch
import torch.nn as nn
from model.pooling import DotAttention

class RNNClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_size=50, rnn_hidden_size=50,
                 classifier_hidden_size=50, classifier_dropout=0.5, out_size=6):
        super(RNNClassifier, self).__init__()
        
        # layers definition 
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, rnn_hidden_size, bidirectional=True, batch_first=True)
        self.attentive_pool = DotAttention(feature_size=rnn_hidden_size * 2, return_attention_weights=True)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, classifier_hidden_size),
            nn.Dropout(classifier_dropout),
            nn.ReLU(),
            nn.Linear(classifier_hidden_size, out_size),
        )
        
        # criterion definition
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, x, labels=None):
        e_x = self.embedding(x)
        h_x, _ = self.rnn(e_x)
        features, attention_weights = self.attentive_pool(h_x)
        logits = self.classifier(features)
        
        if labels is not None:            
            loss = self.criterion(logits, labels)
            return logits, loss, attention_weights
            
        return logits, None, attention_weights