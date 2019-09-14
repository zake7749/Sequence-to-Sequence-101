import torch
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader

class DataHelper(object):
    
    def __init__(self, max_len=100, max_num_words=100000):
        self.max_len = max_len
        self.max_num_words = max_num_words
        self.tokenizer = Tokenizer(num_words=max_num_words)
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.text_column = 'comment_text'
        
    def _get_examples(self, data_path, fit_tokenizer):            
        data = pd.read_csv(data_path, encoding = "utf-8")
        labels = data[self.label_columns].values
        comments = data[self.text_column].values
        
        if fit_tokenizer:
            self.tokenizer.fit_on_texts(comments)
        
        comments = self.tokenizer.texts_to_sequences(comments)
        comments = pad_sequences(comments, self.max_len)  
        
        return comments, labels
    
    def sequences_to_texts(self, sequences):
        return self.tokenizer.sequences_to_texts(sequences)
    
    def get_data_loader(self, data_path, batch_size, 
                        shuffle=True, fit_tokenizer=True, device='cuda:0'):
        comments, labels = self._get_examples(data_path, fit_tokenizer)
        comments, labels = torch.as_tensor(comments).long(), torch.as_tensor(labels).float()
        
        comments = comments.to(device)
        labels = labels.to(device)
                
        dataset = TensorDataset(comments, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader