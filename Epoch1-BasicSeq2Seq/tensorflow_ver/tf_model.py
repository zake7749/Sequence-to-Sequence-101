import numpy as np 
import tensorflow as tf
from DataHelper import DataTransformer
from tensorflow.contrib.rnn import GRUCell

tf.reset_default_graph()
sess = tf.InteractiveSession()

print("Make sure your tf version is the latest 1.3.0 , Your tf version: ",tf.__version__)

class Model():
	def __init__(self, encoder_embedding_size, encoder_output_size, batch_size, vocab_size):
		#define the model hyperarameters here
		"""
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

		#Embedding:
		"""	
			We can use the one hot vector in the Datahelper to find the index of the vector in the emdbeddings
		"""
		initializer = tf.random_uniform_initializer(-1, 1, dtype=tf.float32)	
		embeddings = tf.get_variable(name='embedding',
			shape=[vocab_size, encoder_embedding_size],
			initializer=initializer, dtype=tf.float32)

		# embeddings = tf.get_variable(tf.random_uniform([vocab_size, encoder_embedding_size], -1.0, 1.0), dtype=tf.float32)

		encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
		decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)
		
		
		encoder_cell = GRUCell(self.encoder_hidden_size)
		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
												encoder_cell, 
												encoder_inputs_embedded,
												sequence_length = self.input_sequence_length,
												dtype=tf.float32, 
												time_major=True
												)
		
		#Decoder
		decoder_cell = GRUCell(self.decoder_hidden_size)
		#helper
		helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_inputs_embedded, self.decoder_sequence_length, time_major=True)
		
		decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
												helper,
												encoder_final_state	
												)
		outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
		

		decoder_logits = outputs.rnn_output
		cross_entrophy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            			labels=self.decoder_targets,
            			logits=decoder_logits
        				)
		target_weights = self.target_sequence_length 
		self.train_loss = (tf.reduce_sum(cross_entrophy * target_weights)/batch_size)