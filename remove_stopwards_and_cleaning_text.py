import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
from bs4 import BeautifulSoup
import string

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'Will it be rainy tomorrow?'
]

imdb_sentences=[]
# It will give us the sentences
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))
for item in train_data:
    # Appending the sentences
    imdb_sentences.append(str(item['text']))

# Now we have sentences hence we are converting to tokenizer means in number format
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)
print(tokenizer.word_index)



stopwords = ["a", "about", "above", "yours", "yourself", "yourselves"]

table = str.maketrans(", ", string.punctuation)

imdb_sentences = []
train_data = tfds. as_numpy(tfds.load('imdb_reviews', split="train"))
for item in train_data:
    sentences = str(item['text'].decode('utf-8').lower())
    # Now we have to remove the small words
    soup = BeautifulSoup(sentences)
    sentences = soup.get_text()
    words = sentences.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
        imdb_sentences.append(filtered_sentence)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(imdb_sentences)
    sequences = tokenizer.texts_to_sequences(imdb_sentences)
    print(tokenizer.word_index)

