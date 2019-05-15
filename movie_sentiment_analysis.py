import os
import pickle as pk

imdb_dir = '../datasets/aclImdb/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
labels = list()
texts = list()

# Processing the labels of the raw IMDB data
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


# Tokenizing the data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.optimizers import RMSprop

# cut off reviews after 500 words
max_len = 500 
# train on 10000 samples
training_samples = 10000
 # validate on 10000 samples 
validation_samples = 10000
# consider only the top 10000 words
max_words = 10000 

tokenizer_path = 'tokenizer'
# import tokenizer with the consideration for only the top 500 words
tokenizer = Tokenizer(num_words=max_words) 
# fit the tokenizer on the texts
tokenizer.fit_on_texts(texts) 
# convert the texts to sequences
sequences = tokenizer.texts_to_sequences(texts) 

# save the tokenizer
with open(os.path.join(tokenizer_path, 'tokenizer_m1.pickle'), 'wb') as handle:
    pk.dump(tokenizer, handle, protocol=pk.HIGHEST_PROTOCOL)

word_index = tokenizer.word_index
print('Found %s unique tokens. ' % len(word_index))

 # pad the sequence to the required length to ensure uniformity
data = pad_sequences(sequences, maxlen=max_len)
print('Data Shape: {}'.format(data.shape))

labels = np.asarray(labels)
print("Shape of data tensor: ", data.shape)
print("Shape of label tensor: ", labels.shape)

# split the data into training and validation set but before that shuffle it first
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:training_samples + validation_samples]
y_val = labels[training_samples:training_samples + validation_samples]

# test_data
x_test = data[training_samples+validation_samples:]
y_test = labels[training_samples+validation_samples:]

# model definition
import keras
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import Model, layers
from keras import Input

callback_list = [
    keras.callbacks.EarlyStopping(
        patience=1,
        monitor='acc',
    ),
    
    keras.callbacks.TensorBoard(
        log_dir='log_dir_m1',
        histogram_freq=1,
        embeddings_freq=1,
    ),

    keras.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_best_only=True,
        filepath='model/movie_sentiment_m1.h5',
    ),

    keras.callbacks.ReduceLROnPlateau(
        patience=1,
        factor=0.1,
    )
]

# layer developing
text_input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(max_words, 50)(text_input_layer)
text_layer = Conv1D(256, 3, activation='relu')(embedding_layer)
text_layer = MaxPooling1D(3)(text_layer)
text_layer = Conv1D(256, 3, activation='relu')(text_layer)
text_layer = MaxPooling1D(3)(text_layer)
text_layer = Conv1D(256, 3, activation='relu')(text_layer)
text_layer = MaxPooling1D(3)(text_layer)
text_layer = Conv1D(256, 3, activation='relu')(text_layer)
text_layer = MaxPooling1D(3)(text_layer)
text_layer = Conv1D(256, 3, activation='relu')(text_layer)
text_layer = MaxPooling1D(3)(text_layer)
text_layer = GlobalMaxPooling1D()(text_layer)
text_layer = Dense(256, activation='relu')(text_layer)
output_layer = Dense(1, activation='sigmoid')(text_layer)
model = Model(text_input_layer, output_layer)
model.summary()
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

# multi-input test
history = model.fit(x_train, y_train, epochs=50, batch_size=128, callbacks=callback_list,
                    validation_data=(x_val, y_val))

# plotting the results
import matplotlib.pyplot as plt

acc = history.history.get('acc')
val_acc = history.history.get('val_acc')
loss = history.history.get('loss')
val_loss = history.history.get('val_loss')

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training Acc')
plt.plot(epochs, val_acc, 'b', label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

