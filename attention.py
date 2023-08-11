import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        '''
        self.s: layer for initial hidden unit of decoder
        self.t: layer for all hidden units of encoder
        self.a: layer for predict attention values
        '''

        self.s = tf.keras.layers.Dense(units, kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self.t = tf.keras.layers.Dense(units, kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self.a = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(0.001))

    '''
    query: stands for initial hidden unit of decoder
    values: stand for hidden units of encoder
    '''
    def call(self, query, values):
        query_b = tf.expand_dims(query, 1)

        score = self.a(tf.keras.activations.tanh(self.s(query_b) + self.t(values))) # concatenate the query and values layer
        attention_weights = tf.keras.activations.softmax(score, axis=1) # apply softmax function (64, 24, 1) -> (BATCH_SIZE, max_len_train, 1)
                                                                        # axis provided to train the weights accordingly

        attention = attention_weights * values # calculate the attention provided to each word (64, 24, 1024) -> (BATCH_SIZE, max_len_train, value shape)
        attention = tf.reduce_sum(attention, axis=1) # (64, 1024)

        return attention, attention_weights

if __name__ == '__main__':
    print('ood sssd attn')