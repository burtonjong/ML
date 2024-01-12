import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


with open('lyrics3.txt', 'r', encoding='utf-8') as file:
    lyrics_text = file.read()

data = lyrics_text.lower().split('\n')

# To remove duplicates (chorus, etc.)
unique_set = set(data)
songs = list(unique_set)

tokenizer = Tokenizer()

print(songs)

tokenizer.fit_on_texts(songs)
total_words = len(tokenizer.word_index) + 1



input_sequences = []
for line in songs:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

x, labels = input_sequences[:,:-1],input_sequences[:,-1]
y = to_categorical(labels, num_classes=total_words)

model = Sequential([
    Embedding(total_words, 140, input_length=max_sequence_len-1),
    Bidirectional(LSTM(150)),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

r = model.fit(x,y, epochs=40, verbose=1)

seed_text = "Say no and see what happens"
next_words = 60

for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = np.argmax(model.predict(token_list), axis=-1)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)