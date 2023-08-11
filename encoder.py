import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding, enc_units, batch_size):

        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units # Size of the hidden units present in GRU.
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding)
        self.GRU = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.L2(0.001))
        # here GRU (Gated recurrent unit) is used

    """
    inputs: initial input from training data
    hidden: initial hidden state for GRU
    output: all intermediate hidden states of GRU
    state: final hidden state of GRU 
    """

    def call(self, inputs, hidden):
        inputs = self.embedding(inputs)
        output, state = self.GRU(inputs, initial_state = hidden)
        return output, state

if __name__ == '__main__':
    print('oot sssd enc')