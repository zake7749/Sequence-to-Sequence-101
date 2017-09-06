import torch
from torch.autograd import Variable

from model.Encoder import VanillaEncoder
from model.Decoder import VanillaDecoder
from model.Seq2Seq import Seq2Seq
from dataset.DataHelper import DataTransformer

######### define hyper parameter #########
use_cuda = True if torch.cuda.is_available() else False

# for training
epoch = 50
batch_size = 128
learning_rate = 1e-3

# model
encoder_embedding_size = 128
encoder_output_size = 256
decoder_hidden_szie = encoder_output_size
#########    end of definiton   #########

def train(model, data_transformer):
    pass


if __name__ == "__main__":

    data_transformer = DataTransformer('dataset/Google-10000-English.txt', use_cuda=False)
    input_batches, target_batches = data_transformer.mini_batches(batch_size=10)

    # define our models
    vanilla_encoder = VanillaEncoder(vocab_size=data_transformer.vocab_size,
                                     embedding_size=encoder_embedding_size,
                                     output_size=encoder_output_size)
    vanilla_decoder = VanillaDecoder(hidden_size=decoder_hidden_szie,
                                     output_size=data_transformer.vocab_size)
    seq2seq = Seq2Seq(encoder=vanilla_decoder,
                      decoder_input=vanilla_decoder)

    train(seq2seq, data_transformer)