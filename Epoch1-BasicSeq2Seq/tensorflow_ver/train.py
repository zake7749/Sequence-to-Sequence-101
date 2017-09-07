from tf_model import Model
from DataHelper import DataTransformer
import tensorflow as tf 
import numpy as np 



# for training
epochs = 50
batch_size = 100
learning_rate = 1e-10
grad_clip = 1.0
# model
encoder_embedding_size = 128
encoder_output_size = 128
decoder_hidden_size = encoder_output_size




def train(model, data_transformer):

	loss = 0 
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(model.train_loss, tvars), grad_clip)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.apply_gradients(zip(grads, tvars))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(0, epochs):
			input_batches, target_batches = data_transformer.mini_batches(batch_size=batch_size)
			for input_batch, target_batch in zip(input_batches, target_batches):
				input_length = np.array(input_batch[1])
				target_length = np.array(target_batch[1])
				input_batch = np.transpose(np.array(input_batch[0]))
				target_batch = np.transpose(np.array(target_batch[0]))
				if (target_length.shape[0]==100):
					
				# print("B0-0, Inputs")
				# print(input_batch[0],"\n", input_batch[1])
				# print("Targets")
				# print(target_batch[0],"\n",target_batch[1])
					cost,_ = sess.run([model.train_loss,train_op],feed_dict={
							model.encoder_inputs: input_batch,
							model.decoder_targets: target_batch,
							model.decoder_inputs: target_batch,
							model.input_sequence_length: input_length,
							model.decoder_sequence_length: target_length,
							model.target_sequence_length: target_length
						}     	
						)
					print ("cost:",cost)
def main():
	data_transformer = DataTransformer('../dataset/Google-10000-English.txt', use_cuda=False)

	seq2seq_model = Model(encoder_embedding_size, encoder_output_size,batch_size,vocab_size= data_transformer.vocab_size)
	train(seq2seq_model,data_transformer)

if __name__ == "__main__":
	main()
