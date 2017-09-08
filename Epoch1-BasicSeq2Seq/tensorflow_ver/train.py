from tf_model import Model
from DataHelper import DataTransformer
import tensorflow as tf 
import numpy as np 



#hype parameters for training
epochs = 50
batch_size = 100
learning_rate = 1e-3
grad_clip = 0.3
# model parameter 
encoder_embedding_size = 128
decoder_hidden_size = encoder_output_size = 128



class trainer_class():
	def train(self, model, data_transformer):

		n_iter = 0
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
					n_iter +=1
					# print ("input batch:",input_batch)
					# print ("target batch", target_batch)
					# print ("input length",input_length)
					# print ("target length", target_length)

					if target_length.shape[0] == batch_size:
						
					# print("B0-0, Inputs")
					# print(input_batch[0],"\n", input_batch[1])
					# print("Targets")
					# print(target_batch[0],"\n",target_batch[1])
						output, loss, _ = sess.run([model.decoder_logits,model.train_loss,train_op],feed_dict={
								model.encoder_inputs: input_batch,
								model.decoder_targets: target_batch,
								model.decoder_inputs: target_batch,
								model.input_sequence_length: input_length,
								model.decoder_sequence_length: target_length,
								model.target_sequence_length: target_length
							})

						if n_iter % 50 == 0:
							print ("loss:",loss)
							
							Predict_Words =[]
							Input_words = []
							for batch in output:
								word = []
								for prob in batch:
									Max=np.argmax(prob)
									if Max ==1 or Max ==2:
										continue
									char = data_transformer.vocab.idx2char[Max]
									word.append(char)
									tmp_word = ''.join(word)
								Predict_Words.append(tmp_word)
							for batch in np.transpose(input_batch):
								word= data_transformer.vocab.indices_to_sequence(batch)
								Input_words.append(word)

							print("Input",Input_words)
							print("Predict",Predict_Words,"\n")
						
	def evaluate(self, words, data_transformer):	
		
		results = data_transformer.vocab.sequence_to_indices(words)
		length = len(results)
		
		return results, length

def main():
	data_transformer = DataTransformer('../dataset/Google-10000-English.txt', use_cuda=False)
	seq2seq_model = Model(encoder_embedding_size, encoder_output_size,batch_size,vocab_size= data_transformer.vocab_size)
	trainer = trainer_class()
	trainer.train(seq2seq_model, data_transformer)

if __name__ == "__main__":
	main()
