import random
import torch
import torch.nn as nn

from torch.autograd import Variable


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, targets):
        input_vars, input_lengths = inputs
        encoder_outputs, encoder_hidden = self.encoder.forward(input_vars, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder.forward(context_vector=encoder_hidden, targets=targets)
        return decoder_outputs, decoder_hidden

    def evaluation(self, inputs):
        input_vars, input_lengths = inputs
        encoder_outputs, encoder_hidden = self.encoder(input_vars, input_lengths)
        decoded_sentence = self.decoder.evaluation(context_vector=encoder_hidden)
        return decoded_sentence

