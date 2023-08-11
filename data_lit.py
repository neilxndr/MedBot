from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd

class Data_Augmentation:
    def __init__(self):
       self.hello = None

    def get_data(self, path):
        #file = pd.read_csv(path)
        #questions = file['questions'].values
        #answers = file['answers'].values

        file = open(path, 'r').read()
        lists = [f.split('\t') for f in file.split('\n')]

        questions = [x[0] for x in lists]
        answers = [x[1] for x in lists]

        return questions, answers

    def preprocess_sentence(self, line):
        line = line.lower().strip()

        line = re.sub(r"([?.!,¿])", r" \1 ", line) # create the space between words and [?.!,¿] these signs
        line = re.sub(r'[" "]+', " ", line) # remove the extra space between the words
        line = re.sub(r"[^a-zA-Z?.!,¿]+", " ", line) # allow only alphabets and [?.!,¿] these symbols or remove the digits.
        line = line.strip()
        line = '<start> ' + line + ' <end>' # join the <start> and <end> tags at both ends of the sentence.

        return line

    def word_to_vec(self, inputs):
        tokenizer = Tokenizer(filters='') # this tokenizer will filters nothing.
        tokenizer.fit_on_texts(inputs)

        tensor = tokenizer.texts_to_sequences(inputs)
        tensor = pad_sequences(tensor, padding='post')

        return tensor, tokenizer

if __name__ == '__main__':
    print('oot sssd data')