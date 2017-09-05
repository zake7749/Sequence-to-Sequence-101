import torch.nn as nn


class VanillaDecoder(nn.Module):

    def __init__(self, hidden_size, embedding_size,  output_size):
        """Define layers for a vanilla rnn decoder"""
        super(VanillaDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        batch_size = inputs.size(0)
        embedded = self.embedding(input)
        embedded.view(1, batch_size, self.hidden_size) # S = T(1) x B x N
        rnn_output, hidden = self.gru(embedded, hidden)
        rnn_output = rnn_output.squeeze(0) # squeeze the time dim
        output = nn.LogSoftmax(self.out(rnn_output))
        return output, hidden
