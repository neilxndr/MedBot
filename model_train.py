import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # this is just to hide the tensorflow info or warnings
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from train import Train
from data_lit import Data_Augmentation
import pickle

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPUs found.")
''' Install the data '''
data = Data_Augmentation()

question, answer = data.get_data("C:/Users/neilx/Downloads/chatter_bot_3/chatter_bot_3/new2.txt") # get the question answer list individually

pre_questions = [data.preprocess_sentence(str(w)) for w in question] # process the train data
pre_answers = [data.preprocess_sentence(str(w)) for w in answer] # process the label data

train_tensor, train_tokenizer = data.word_to_vec(pre_questions) # get training token vector and train tokenizer
#testing
print(len(train_tensor))


label_tensor, label_tokenizer = data.word_to_vec(pre_answers) # get label token vector and label tokenizer

''' Calculate max_length tokenize data '''
max_length_label, max_length_train = label_tensor.shape[1], train_tensor.shape[1] # length of train features and label features.

''' Save the tokenize data '''
with open('train_tokenizer.pickle', 'wb') as handle:
    pickle.dump(train_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_tokenizer.pickle', 'wb') as handle:
    pickle.dump(label_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

''' Save the maximum length of training feature'''
file = open('max_data.txt', 'w')
file.write(str(max_length_train))
file.close()

''' Initialize the HyperParameters '''
BUFFER_SIZE = len(train_tensor)
BATCH_SIZE = 64 #was 64
steps_per_epoch = len(train_tensor)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_train = len(train_tokenizer.word_index)+1
vocab_label = len(label_tokenizer.word_index)+1


''' Convert dataset into tensor'''
dataset = tf.data.Dataset.from_tensor_slices((train_tensor, label_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


''' Initialize the ENCODER, DECODER and TRAINER '''
encoder = Encoder(vocab_train, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_label, embedding_dim, units, BATCH_SIZE)
trainer = Train()


''' Load the weights '''
#encoder.load_weights('enc/') # path to load the weights
#decoder.load_weights('dec/')

EPOCHS = 40

for epoch in range(1, EPOCHS + 1):
    enc_hidden = tf.zeros((BATCH_SIZE, units)) # initial input for encoder model
    total_loss = 0

    for (batch, (train, label)) in enumerate(dataset.take(steps_per_epoch)): # fetch the dataset with batch size
        batch_loss = trainer.train_step(train, label, enc_hidden, encoder, decoder, BATCH_SIZE, label_tokenizer)
        total_loss += batch_loss

    print('Epoch:{:3d} Loss:{:.4f}'.format(epoch, total_loss / steps_per_epoch))

''' Save The Weights'''
encoder.save_weights('enc/') # path to save the weights
decoder.save_weights('dec/')

