import torch
from torch.autograd import Variable

from Model.Encoder import VanillaEncoder
from Model.Decoder import VanillaDecoder
from Dataset.DataHelper import DataTransformer


# Define the hpyer parameter
use_cuda = True if torch.cuda.is_available() else False

# model information
batch_size = 128
encoder_embedding_size = 128
encoder_output_size = 256
decoder_hidden_szie = encoder_output_size


if __name__ == "__main__":
    pass