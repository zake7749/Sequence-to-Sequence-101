import torch
from torch.autograd import Variable

from model.Encoder import VanillaEncoder
from model.Decoder import VanillaDecoder
from model.Seq2Seq import Seq2Seq
from dataset.DataHelper import DataTransformer

# define hyper parameter
use_cuda = True if torch.cuda.is_available() else False

# for training
epochs = 50
batch_size = 128
learning_rate = 1e-3

# model
encoder_embedding_size = 128
encoder_output_size = 256
decoder_hidden_size = encoder_output_size


def train(model, data_transformer, criterion):

    loss = 0
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

    for epoch in range(0, epochs):
        input_batches, target_batches = data_transformer.mini_batches(batch_size=batch_size)
        for input_batch, target_batch in zip(input_batches, target_batches):
            decoder_outputs, decoder_hidden = model(input_batch, target_batch)
            loss += calculate_loss(criterion, decoder_outputs, target_batch[0])


def calculate_loss(criterion, decoder_outputs, targets):
    #To_Fix
    batch_size = decoder_outputs.size(1)
    time_steps = decoder_outputs.size(0)
    targets = targets.transpose(0, 1) # S = B * T
    decoder_outputs = decoder_outputs.view(batch_size * time_steps, -1)  # S = (B*T) x V
    loss = 0
    loss += criterion(decoder_outputs, targets)

def main():
    data_transformer = DataTransformer('dataset/Google-10000-English.txt', use_cuda=False)

    # define our models
    vanilla_encoder = VanillaEncoder(vocab_size=data_transformer.vocab_size,
                                     embedding_size=encoder_embedding_size,
                                     output_size=encoder_output_size)
    vanilla_decoder = VanillaDecoder(hidden_size=decoder_hidden_size,
                                     output_size=data_transformer.vocab_size)
    seq2seq = Seq2Seq(encoder=vanilla_encoder,
                      decoder=vanilla_decoder,
                      sos_index=data_transformer.SOS_ID,
                      use_cuda=use_cuda)

    # define the masked NLLoss
    weight = torch.ones(data_transformer.vocab_size)
    weight[data_transformer.PAD_ID] = 0
    criterion = torch.nn.NLLLoss(weight=weight)

    train(seq2seq, data_transformer, criterion)

if __name__ == "__main__":
    main()
