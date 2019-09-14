import torch
import torch.nn as nn

class DotAttention(nn.Module):
    
    def __init__(self, feature_size, std=0.5, return_attention_weights=False):
        super(DotAttention, self).__init__()
        self.query = nn.Parameter(torch.zeros(feature_size))
        self.query.data.uniform_(-std, std)
        self.return_attention_weights = return_attention_weights
        
    def forward(self, x):
        """
            input shape: (batch, time_step, hidden)
            output shape: (batch, hidden)
        """
        attention_energies = torch.matmul(x, self.query) # (B, T, H) * (H,) -> (B, T)
        attention_weights = torch.nn.functional.softmax(attention_energies) # (B, T)
        
        attented_results = torch.matmul(attention_weights, x) # (B, T) * (B, T, H) -> (B, T, H)
        attented_results = torch.sum(attented_results, dim=1) # (B, T, H) -> (B, H)
        
        if self.return_attention_weights:
            return attented_results, attention_weights
        else:
            return attented_results, None

class GlobalAveragePooling(nn.Module):

    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)


class GlobalMaxPooling(nn.Module):

    def __init__(self):
        super(GlobalMaxPooling, self).__init__()

    def forward(self, x):
        res, _ = torch.max(x, dim=1)
        return res