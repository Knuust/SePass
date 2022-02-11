import pickle
import gensim
import os
from tqdm import tqdm

test_list_path = 'final/wörterbücher/'
password_list_path = 'final/generated_password_lists/'

def load_password_list(file_name):
    file = open(file_name + '.txt', "r", encoding='latin1')
    password_list = [line.rstrip() for line in file.readlines()]
    return password_list

vocabularies = []
for path in os.listdir('vocabs'):
    vocabularies.append(list(pickle.load(open('vocabs/' + path, 'rb'))))

def flatten(t):
    return [item for sublist in t for item in sublist]

full_vocab = flatten(vocabularies)

full_vocab_no_duplicates = set(full_vocab)

vocab_no_numbers = [word for word in full_vocab_no_duplicates if not any(c.isdigit() for c in word)]

vocab_clean = [word for word in vocab_no_numbers if len(word) < 20]

test_set = load_password_list(test_list_path + 'test')
train_set = load_password_list(test_list_path + 'train')

train_string = ' '.join(train_set).lower()
test_string = ' '.join(test_set).lower()

train_words = []
test_words = []

for w in tqdm(range(len(vocab_clean))):
    word = vocab_clean[w].lower()
    if word in train_string:
        train_words.append(word)
    if word in test_string:
        test_words.append(word)

pickle.dump(train_words, open('train_words_real.pkl', 'wb'))
pickle.dump(test_words, open('test_words_real.pkl', 'wb'))