from tf_model import Model
from DataHelper import DataTransformer
import tensorflow as tf 
import numpy as np 


 
# Hype parameters for training
epochs = 50
batch_size = 100
learning_rate = 1e-3
grad_clip = 0.3

# Model parameter 
encoder_embedding_size = 128
decoder_hidden_size = encoder_output_size = 128



class trainer_class():
    def train(self, model, data_transformer):
        #Define some parameter and the optimizer here 
        n_iter = 0
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
                
                    if target_length.shape[0] == batch_size:
                        #Feed the batch into the training model 
                        output, loss, _ = sess.run([model.decoder_logits,model.train_loss,train_op],feed_dict={
                                model.encoder_inputs: input_batch,
                                model.decoder_targets: target_batch,
                                model.decoder_inputs: target_batch,
                                model.input_sequence_length: input_length,
                                model.decoder_sequence_length: target_length,
                                model.target_sequence_length: target_length
                            })
                        #Print loss and result after training 50 epoch
                        if n_iter % 50 == 0:
                            Predict_Words =[]
                            Input_words = []
                            # Two for loops are used to discard the "EOS"
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
                            
                            print("-----{}epochs------------- ".format(n_iter))
                            print("loss: ",loss,"\n")
                            print("Input: ",Input_words,"\n")
                            print("Predict: ",Predict_Words,"")
                            print("--------------------------\n ")

                        

def main():
    data_transformer = DataTransformer('../dataset/Google-10000-English.txt', use_cuda=False)
    seq2seq_model = Model(encoder_embedding_size = encoder_embedding_size, 
                          encoder_output_size = encoder_output_size,
                          batch_size = batch_size,
                          vocab_size= data_transformer.vocab_size)
    trainer = trainer_class()
    trainer.train(seq2seq_model, data_transformer)

if __name__ == "__main__":
    main()
