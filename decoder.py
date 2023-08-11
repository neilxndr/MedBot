import tensorflow as tf
from attention import Attention

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units # size of decoder hidden units present in GRU.
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding)
        self.GRU = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.L2(0.001))

        # self.otpt is the final output layer for the prediction of words, units size of vocabulary
        # prediction label would be look like [ 0 0 0 0 0 1 0 0 0...........]
        self.otpt = tf.keras.layers.Dense(vocab_size, kernel_regularizer=tf.keras.regularizers.L2(0.001))

        # initiating the attention layer
        self.attention = Attention(self.dec_units)

    '''
    input: first input given to the decoder GRU -> (<start> index value)
    hidden: final hidden layer from encoder
    enc_output: all intermediate hidden layers from encoder
    '''
    def call(self, inputs, hidden, enc_output):
        attention, attention_weights = self.attention(hidden, enc_output)

        inputs = self.embedding(inputs)
        inputs = tf.concat([tf.expand_dims(attention, 1), inputs], axis=-1) # concatenate or join the input and attention layers.

        output, state = self.GRU(inputs) # provide the input to the GRU (output shape (64, 1, 1024))
        output = tf.reshape(output, (-1, output.shape[2])) # shape (64, 1024)

        output = self.otpt(output)
        return output, state, attention_weights

if __name__ == '__main__':
    print('ood sssd dec')