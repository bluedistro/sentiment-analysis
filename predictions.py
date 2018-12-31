from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os, pickle
import numpy as np

tokenizer_path = 'tokenizer'
model_path = 'model'
model_file = os.path.join(model_path, 'movie_sentiment_m1.h5')
tokenizer_file = os.path.join(tokenizer_path, 'tokenizer_m1.pickle')
model = load_model(model_file)

# load tokenizer
with open(tokenizer_file, 'rb') as handle:
    tokenizer = pickle.load(handle)

def decode_review(text_list):
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(text_list)
    data = pad_sequences(sequences, maxlen=500)

    # decode the words
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in sequences[0]])
    return decoded_review, data

def review_rating(score, decoded_review):
    if float(score) >= 0.9:
        print('Review: {}\nSentiment: Strongly Positive\nScore: {}'.format(decoded_review, score))
    elif float(score) >= 0.7 and float(score) < 0.9:
        print('Review: {}\nSentiment: Positive\nScore: {}'.format(decoded_review, score))
    elif float(score) >= 0.5 and float(score) < 0.7:
        print('Review: {}\nSentiment: Okay\nScore: {}'.format(decoded_review, score))
    else:
        print('Review: {}\nSentiment: Negative\nScore: {}'.format(decoded_review, score))
    print('\n\n')

def score_review(source=None, file_type=None):
    text_list = list()
    if isinstance(source, str) and file_type is None:
        text_list.append(source)
        decoded_review, data = decode_review(text_list)
        # make prediction
        score = model.predict(data)[0][0]
        review_rating(score, decoded_review)
    
    if isinstance(source, list) and file_type is None:
        for item in source:
            text_list = list()
            text_list.append(item)
            decoded_review, data = decode_review(text_list)
            score = model.predict(data)[0][0]
            review_rating(score, decoded_review)
    
    if isinstance(source, str) and file_type == 'file':
        file_data = open(source).read()
        text_list.append(file_data)
        decoded_review, data = decode_review(text_list)
        # make prediction
        score = model.predict(data)[0][0]
        review_rating(score, decoded_review)
    
    if isinstance(source, str) and file_type == 'dir':
        file_content_holder = list()
        for fname in os.listdir(source):
            if fname[-4:] == '.txt':
                f = open(os.path.join(source, fname))
                file_content_holder.append(f.read())
                f.close()
        for item in file_content_holder:
            text_list = list()
            text_list.append(item)
            decoded_review, data = decode_review(text_list)
            score = model.predict(data)[0][0]
            review_rating(score, decoded_review)

    
score_review('test_dir/test', file_type='dir')