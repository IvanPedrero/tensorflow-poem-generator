import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 
import os, sys
import tempfile

#from __future__ import absolute_import, division, print_function, unicode_literals

def getParameters():
  # 1. Create the tokenizer object with the poem words.
  global tokenizer
  tokenizer = Tokenizer()

  global __location__
  __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

  data = open(os.path.join(__location__, 'irish-lyrics-eof.txt')).read()

  corpus = data.lower().split("\n")

  tokenizer.fit_on_texts(corpus)
  total_words = len(tokenizer.word_index) + 1

  #print(tokenizer.word_index)
  #print(total_words)

  # 2. Create the sequences to get the x and y axis to predict the words.

  input_sequences = []
  for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
      n_gram_sequence = token_list[:i+1]
      input_sequences.append(n_gram_sequence)

  # pad sequences
  global max_sequence_len
  max_sequence_len = max([len(x) for x in input_sequences])
  input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

  # create predictors and label
  xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

  ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

  return xs, ys, total_words, max_sequence_len

def create_model(xs, ys, total_words, max_sequence_len, epochs = 150):
  # 3. Define the ml model.

  model = Sequential()
  model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
  model.add(Bidirectional(LSTM(150)))
  model.add(Dense(total_words, activation='softmax'))
  adam = Adam(lr=0.01)

  print("Sequential model done...")

  # 4. Train the model.

  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  history = model.fit(xs, ys, epochs=epochs, verbose=1)
  #print model.summary()
  #print(model)

  print("Finish compiling, returning...")

  return model


def generate_poem(seed_text, model, total_words, max_sequence_len) :
  next_words = 250

  print("Entered generate poem...")
    
  for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
      if index == predicted:
        output_word = word
        break
    seed_text += " " + output_word
  # print(seed_text)
  return seed_text


def getPoem(seed, train):
  print("Entered get poem with: " + seed)

  xs, ys, total_words, max_sequence_len = getParameters()

  # Get the model.
  model = None
  save_path = os.path.join(__location__, "trained_models/poem_generator/1/")

  print("Save path: " + save_path)

  if train :
    model =  create_model(xs, ys, total_words, max_sequence_len, 1)
    tf.keras.experimental.export_saved_model(model, save_path)
  else:
    model = tf.keras.experimental.load_from_saved_model(save_path)
  
  print("Loaded the model...")

  print(model.summary())

  print("Generated model! Proceeding with poem generation...")

  # Generate poem.
  poem = generate_poem(seed, model, total_words, max_sequence_len)

  print("Generated: " + poem)

  return poem

