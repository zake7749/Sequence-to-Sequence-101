import torch
import torch.nn as nn

from torch.autograd import Variable


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, sos_index, use_cuda):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_index = sos_index
        self.use_cuda = use_cuda

    def forward(self, inputs, targets):

        # data unzip (torch.Variables, list of length)
        input_vars, input_lengths = inputs
        target_vars, target_lengths = targets
        batch_size = input_vars.size(1)

        encoder_outputs, encoder_hidden = self.encoder(input_vars, input_lengths)

        # Prepare variable for decoder on time_step_0
        decoder_input = Variable(torch.LongTensor([[self.sos_index] * batch_size]))

        # Pass the context vector
        decoder_hidden = encoder_hidden

        max_target_length = max(target_lengths)
        decoder_outputs = Variable(torch.zeros(
            max_target_length,
            batch_size,
            self.decoder.output_size
        ))  # (time_steps, batch_size, vocab_size)

        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()

        # Unfold the decoder RNN on the time dimension
        for t in range(max_target_length):
            decoder_outputs_on_t, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[t] = decoder_outputs_on_t
            decoder_input = self.eval_on_char(decoder_outputs_on_t)  # select the former output as input

        return decoder_outputs, decoder_hidden

    def eval_on_char(self, decoder_output):
        """
        evaluate on the decoder output(logits), get the index of top1
        :param decoder_output: S = B x V or T x V
        """
        value, index = torch.topk(decoder_output, 1)
        index = index.transpose(0, 1)  # S = 1 x B, 1 is the index of top1
        if self.use_cuda:
            index = index.cuda()
        return index

    def eval_on_sequence(self, decoder_outputs):
        """
        Evaluate on the decoder output(logits), find the top 1 index.
        Please make sure that the seq2seq model is in evaluation mode when testing
        :param decoder_outputs: the output sequence from decoder, shape = T x B x V 
        """
        decoded_indices = []

        max_length = decoder_outputs.size(0)
        batch_size = decoder_outputs.size(1)

        decoder_outputs = decoder_outputs.transpose(0, 1) # S = B x T x V

        for b in range(batch_size):
            top_ids = self.eval_on_char(decoder_outputs[b])
            decoded_indices.append(top_ids.data[0])
        return decoded_indices

