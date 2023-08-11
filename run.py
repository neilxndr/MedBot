from flask import Flask, render_template, request
from encoder import Encoder
from decoder import Decoder
from process import Initializer
import pickle

''' Load the tokenizer data '''
with open('train_tokenizer.pickle', 'rb') as handle: # train_tokenizer path
    train_tokenizer = pickle.load(handle)

with open('label_tokenizer.pickle', 'rb') as handle: # label_tokenizer path
    label_tokenizer = pickle.load(handle)

''' Lode the maximum training feature length '''
file = open('max_data.txt', 'r') # max_data path
max_len = int(file.read())

''' Process the data '''
BATCH_SIZE = 64 #was 64
embedding_dim = 256
units = 1024
vocab_train = len(train_tokenizer.word_index)+1
vocab_label = len(label_tokenizer.word_index)+1

encoder = Encoder(vocab_train, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_label, embedding_dim, units, BATCH_SIZE)

encoder.load_weights('enc/') # enc path for loading the weights
decoder.load_weights('dec/') # dec path for loading the weights


''' Start the sequence '''

seq = Initializer(units, train_tokenizer, max_len, label_tokenizer, encoder, decoder)
#print(seq.test('hi'))


''' Flask for the environment '''
app = Flask('My Chatter',static_folder="C:\\Users\\neilx\\Downloads\\chatter_bot_3\\chatter_bot_3\\templates\\images")

@app.route("/")
def home():
    return render_template("home.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg') # User provide the sentences
    return str(seq.test(userText)) # return the predicted output from the chatbot

if __name__ == "__main__":
    app.run()
