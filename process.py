import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_lit import Data_Augmentation
import numpy as np

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Initializer:
    def __init__(self, units, train_tokenizer, max_length_train, label_tokenizer, encoder, decoder):
        self.data = Data_Augmentation()
        self.train_tokenizer = train_tokenizer
        self. max_len = max_length_train
        self.units = units
        self.label_tokenizer = label_tokenizer
        self.enc = encoder
        self.dec = decoder

    # Remove the <start> and <end> tags from the sentences
    def Expand(self, sentence):
        return sentence.split("<start>")[-1].split("<end>")[0]

    # proceed for real time prediction.
    '''
    sentence: is the sentence given by the chatbot user
    '''
    def test(self, sentence):
        sentence = self.data.preprocess_sentence(sentence)

        whole = [] # collect the " " split sentence words
        for i in sentence.split(' '):
            # throw an exception if user input word not present in the vocabulary of the train data
            try:
                self.train_tokenizer.word_index[i]
            except Exception as e:
                return('Please say it clearly')

            whole.append(self.train_tokenizer.word_index[i])

        sentence = pad_sequences([whole], maxlen=self.max_len, padding='post')
        sentence = tf.convert_to_tensor(sentence)

        enc_hidden_start = [tf.zeros((1, self.units))] # initial hidden state provide to the encoder
        enc_hidden, enc_output = self.enc(sentence, enc_hidden_start)

        dec_output = enc_output
        dec_input = tf.expand_dims([self.label_tokenizer.word_index['<start>']], 0)

        answer = '' # store the answer string

        # loop for predict word by word from the decoder
        for i in range(1, self.max_len):
            pred, dec_output, attention_weight = self.dec(dec_input, dec_output, enc_hidden)

            answer += self.label_tokenizer.index_word[np.argmax(pred[0])] + " " # add the predicted value after convert index to word

            if self.label_tokenizer.index_word[np.argmax(pred[0])] == '<end>':
                return self.Expand(answer)

            dec_input = tf.expand_dims([np.argmax(pred[0])], 0) # after a loop the decoder input is equal to the index of the previous predected word.

        return self.Expand(answer)

if __name__ == '__main__':
    print('oot sssd proc')