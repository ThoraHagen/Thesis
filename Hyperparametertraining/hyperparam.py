from __future__ import print_function
import sys
sys.path.append('../')
from hatespeech import preprocessing

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout, LSTM
from keras.utils.np_utils import to_categorical

from hyperas import optim
from hyperas.distributions import choice, uniform
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe

import pickle
import numpy as np
from gensim.models import FastText
import json


def data():
    train_path = '../Data/Datasets/train_data.csv'
    dev_path = '../Data/Datasets/dev_data.csv'
    test_path = '../Data/Datasets/test_data.csv'
    # maxwords=4834
    maxlen = 100
    embedding_dim = 300
    texts, labels, cnt = preprocessing.load_datasets(train_path, dev_path, test_path)
    sequences, word_index, mfws, max_words = preprocessing.tokenize_texts_ngrams(texts, ngrams=True, chars=4)
    data_reshaped, labels_reshaped = preprocessing.reshape(sequences, labels, maxlen=maxlen)

    x_train = data_reshaped[:12000]
    y_train = labels_reshaped[:12000]
    x_test = data_reshaped[15000:18000]
    y_test = labels_reshaped[15000:18000]

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def create_LSTM_model(x_train, y_train, x_test, y_test):
    max_words = 4834
    maxlen = 100
    embedding_dim = 300
    embedding_matrix = pickle.load(open("../embeddings_ngrams_small.p", "rb"))

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))

    choice = {{choice(['one', 'two'])}}

    if choice == 'two':
        model.add(LSTM({{choice([5, 8, 16, 32])}}, return_sequences=True, name='LSTM_2_1'))
        model.add(LSTM({{choice([5, 8, 16, 32])}}, name='LSTM_2_2'))

    elif choice == 'one':
        model.add(LSTM({{choice([5, 8, 16, 32])}}, name='LSTM_1'))

    model.add(Dropout({{uniform(0, 1)}}, name='Dropout'))

    model.add(Dense(3, activation='softmax'))

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    result = model.fit(x_train, y_train,
                       batch_size={{choice([32, 64])}},
                       epochs=15,
                       verbose=2,
                       validation_split=0.1)
    # get the highest validation accuracy of the training epochs

    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == "__main__":
    print('Run evaluation.')
    best_run, best_model = optim.minimize(model=create_LSTM_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=70,
                                          eval_space=True,
                                         # notebook_name='Hyperparam',
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    evaluation = best_model.evaluate(X_test, Y_test)
    print(evaluation)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print(best_model.summary())

    with open('output.txt', 'w') as f:
        f.write(json.dumps(best_run))
        f.write('\n')
        best_model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write('\n')
        for e in evaluation:
            f.write(str(e))
            f.write('\n')
