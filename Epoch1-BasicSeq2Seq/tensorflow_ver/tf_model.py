import numpy as np 
import tensorflow as tf
from DataHelper import DataTransformer
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.layers import core as layers_core

tf.reset_default_graph()
sess = tf.InteractiveSession()

print("Make sure your tf version is the latest 1.3.0 , Your tf version: ",tf.__version__)

class Model():
    def __init__(self, encoder_embedding_size, encoder_output_size, batch_size, vocab_size, forward_only= None):
        # Define the model hyperarameters here
        """
            max_time means the input  length of this batch
            encoder_inputs int32 tensor is shaped [encoder_max_time, batch_size]
            decoder_targets int32 tensor is shaped [decoder_max_time, batch_size]
            decoder_inputs int32 tensor is shaped [decoder_max_time, batch_size]
        """
        self.batch_size = batch_size
        self.encoder_inputs = tf.placeholder(shape=(None, self.batch_size), dtype=tf.int32, name='encoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, self.batch_size), dtype=tf.int32, name='decoder_targets')
        self.decoder_inputs = tf.placeholder(shape=(None, self.batch_size), dtype=tf.int32, name='decoder_inputs')

        self.input_sequence_length = tf.placeholder(shape=([self.batch_size]), dtype=tf.int32, name='input_length')
        self.decoder_sequence_length = tf.placeholder(shape=([self.batch_size]), dtype=tf.int32, name='decoder_inputs_length')
        self.target_sequence_length = tf.placeholder(shape=([self.batch_size]), dtype=tf.float32, name='target_sequence_length')

        self.encoder_hidden_size = encoder_embedding_size
        self.decoder_hidden_size = encoder_output_size 
        
        """Inference placeholder(Not yet finished the custom test part )"""
        # self.encoder_inputs_inf = tf.placeholder(shape=(None, 1), dtype=tf.int32, name='encoder_inputs_inf')
        # self.input_sequence_length_inf =tf.placeholder(shape=([1]), dtype=tf.int32, name='input_length_inf')
        # self.decoder_inputs_inf = tf.placeholder(shape=(None, 1), dtype=tf.int32, name='decoder_inputs')
        # self.forward_only = forward_only
        

        
        # Embedding
        """ 
            We can use the one hot vector in the Datahelper to find the index of the vector in the emdbeddings
        """
        initializer = tf.random_uniform_initializer(-1, 1, dtype=tf.float32)    
        embeddings = tf.get_variable(name='embedding',
            shape=[vocab_size, encoder_embedding_size],
            initializer=initializer, dtype=tf.float32)
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)

        # Define the Cell used in the RNN. We use GRU in this model(Encoder & Decoder)
        encoder_cell = GRUCell(self.encoder_hidden_size)
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                                            encoder_cell, 
                                            encoder_inputs_embedded,
                                            sequence_length = self.input_sequence_length,
                                            dtype=tf.float32, 
                                            time_major=True
                                            )
        
        # Decoder cell
        decoder_cell = GRUCell(self.decoder_hidden_size)
        
        # Training Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_inputs_embedded, self.decoder_sequence_length, time_major=True)
        
        # Project layer (Project the output to this layer(Softmax output). It's shape is [1*vocab_size].)
        output_layer = layers_core.Dense(vocab_size , use_bias=False, name="output_projection")

        # Define the decoder instance and feed it in to dynamic_decode function
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                helper,
                                                encoder_final_state,
                                                output_layer= output_layer
                                                )
        final_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        
        self.decoder_logits = final_outputs.rnn_output
        self.decoder_targets_t = tf.transpose(self.decoder_targets,[1,0])
        print ("decoder_logits_shape:{}, decoder target shape:{},decoder target shape t:{}".format(self.decoder_logits.shape, self.decoder_targets,self.decoder_targets_t))
        self.cross_entrophy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.decoder_targets_t,
                        logits=self.decoder_logits
                        )
        target_weights = tf.sequence_mask( self.target_sequence_length, dtype=tf.float32)
        self.train_loss = (tf.reduce_sum(self.cross_entrophy * target_weights)/batch_size)
        
