import pandas as pd
import numpy as np
import itertools
import pickle
import regex as re
from nltk.tokenize import TweetTokenizer
from gensim.models import FastText
from collections import Counter
from emoji import UNICODE_EMOJI
import emoji
import html

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout, LSTM
from keras.utils.np_utils import to_categorical




def load_data(csv, p1, p2):
    """
    Loads data from 3 specific paths. 
    Returns a list of texts, a corresponding list of labels (0|1|2), and a labels counter.
    """
    df = pd.read_csv(csv, index_col=0, encoding = 'utf-8')

    texts = []
    labels = []
    for row in df.itertuples(index=True):
        tweet = getattr(row, "tweet")
        classif = getattr(row, "classification")
        texts.append(tweet)
        labels.append(classif)

    more_hateful_tweets = pickle.load(open(p1, "rb" ))
    more_clean_tweets = pickle.load(open(p2, "rb" ))

    hateful_labels = [0] * len(more_hateful_tweets)
    clean_labels = [2] * len(more_clean_tweets)
    print('Additional Hateful:', len(more_hateful_tweets), 'Additional Clean:', len(more_clean_tweets))
    
    texts = texts+more_hateful_tweets+more_clean_tweets
    labels = labels+hateful_labels+clean_labels
    
    texts, labels = drop_duplicates(texts, labels)
    cnt = Counter(labels)
    print(cnt)
    return texts, labels, cnt


def load_datasets(train_path, dev_path, test_path):
    """
    Loads data from 3 specific paths.
    Returns a list of texts, a corresponding list of labels (0|1|2), and a labels counter.
    """
    df_train = pd.read_csv(train_path, encoding='utf-8', sep='\t')
    df_dev = pd.read_csv(dev_path, encoding='utf-8', sep='\t')
    df_test = pd.read_csv(test_path, encoding='utf-8', sep='\t')

    data_df = df_train.append([df_dev, df_test])

    labels = list(data_df['label'])
    texts = list(data_df['text'])

    texts, labels = drop_duplicates(texts, labels)
    cnt = Counter(labels)
    print(cnt)
    return texts, labels, cnt

def drop_duplicates(texts, labels):
    """ Removes duplicate entries."""
    single_texts = []
    single_labels = []

    duplicates = 0
    for i, text in enumerate(texts):
        if text not in single_texts:

            single_texts.append(text)
            single_labels.append(labels[i])
        else:
            duplicates += 1
    return single_texts, single_labels

def convert_html_emojis(corpus):
    new_corpus = []
    for text in corpus:
        text = html.unescape(text)
        new_corpus.append(text)
    return new_corpus

def tokenize_texts(corpus, stopword_path = None):
    """
    Tokenizes a list of texts.
    """
    corpus = convert_html_emojis(corpus)
    
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    
    
    token_doc_list = []
    token_list = []
    
    for text in corpus:
 

        tokens = tknzr.tokenize(text)
        tokens2 = []
        for token in tokens:
            token = re.sub(r'([1234567890!"#$%&()*+,./:@;?[\]^`{|}_~\t\n])', '', token)
            if token != '':
                tokens2.append(token)
        
        
        tmp_tokens = []
        for i, token in enumerate(tokens2):
            if token in UNICODE_EMOJI:
               
                emoji_one = emoji.demojize(token)
                emoji_one = emoji_one[1:-1]
                emoji_all = re.split('_', emoji_one)

                tokens[i] = emoji_all
                token_list.extend(emoji_all)
                tmp_tokens.extend(emoji_all)
            elif token == "i'd":
                token_list.extend(['i', 'would'])
                tmp_tokens.extend(['i', 'would'])
            else: 
                token_list.append(token)
                tmp_tokens.append(token)

            
        token_doc_list.append(tmp_tokens)     
            
    index, mfws, max_words = find_most_common(token_list)
   
    if stopword_path == None: stopwords = []
    else: stopwords = read_stopwords(stopword_path)
    new_texts = fit_index_on_texts(index, token_doc_list, stopwords, mfws) 
    return new_texts, index, mfws, max_words


def tokenize_texts_ngrams(corpus, stopword_path=None, ngrams=False, chars=0):
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    
    token_doc_list = []
    token_list = []

    corpus = convert_html_emojis(corpus)

    for text in corpus:
        tokens = tknzr.tokenize(text)
        tokens2 = []
        for token in tokens:
            token = re.sub(r'([1234567890!"#$%&()*+,./:@;?[\]^`{|}_~\t\n])', '', token)
            if token != '' and token !='charlie' and token !='yankees':
                tokens2.append(token)

        tmp_tokens = []
        for i, token in enumerate(tokens2):
            if token in UNICODE_EMOJI:

                emoji_one = emoji.demojize(token)
                emoji_one = emoji_one[1:-1]
                emoji_all = re.split('_', emoji_one)

                tokens[i] = emoji_all
                token_list.extend(emoji_all)
                tmp_tokens.extend(emoji_all)
            elif token == "i'd":
                token_list.extend(['i', 'would'])
                tmp_tokens.extend(['i', 'would'])
            else:
                token_list.append(token)
                tmp_tokens.append(token)

        token_doc_list.append(tmp_tokens)

    if stopword_path == None:
        stopwords = []
    else:
        stopwords = read_stopwords(stopword_path)

    if ngrams == True:
        ngram_list = []
        ngram_doc_list = []
        for doc in token_doc_list:
            tmp_ngrams = []
            for token in doc:
                ngrams_per_token_list = char_ngrams(token, chars)
                ngram_list.extend(ngrams_per_token_list)
                tmp_ngrams.extend(ngrams_per_token_list)
            ngram_doc_list.append(tmp_ngrams)


        index, mfws, max_words = find_most_common(ngram_list)
        new_texts = fit_index_on_texts(index, ngram_doc_list, stopwords, mfws)
    else:
        index, mfws, max_words = find_most_common(token_list)
        new_texts = fit_index_on_texts(index, token_doc_list, stopwords, mfws)

    return new_texts, index, mfws, max_words

def char_ngrams(token, n):
    if len(token) < n:
        return [token]
    else:
        ngrams = [token[i:i+n] for i in range(len(token)-n+1)]
        return ngrams
	
def tokenize_texts_characters(corpus):
    """
    Tokenizes a list of texts.
    """
	
    corpus = convert_html_emojis(corpus)
    
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        
    token_doc_list = []
    token_list = []
   
    for text in corpus:
        tokens = tknzr.tokenize(text)
        tokens2 = []
        for token in tokens:
            
            if token != '' and token !='charlie' and token !='yankees':
                tokens2.append(token)
        
        tmp_tokens = []
        for i, token in enumerate(tokens2):

            if token == "i'd":
                token_list.extend(['i', 'would'])
                tmp_tokens.extend(['i', 'would'])
            else: 
                token_list.append(token)
                tmp_tokens.append(token)
          
        token_doc_list.append(tmp_tokens)
        
    final_texts = []
    for doc in token_doc_list:
        t = " ".join(doc)
        final_texts.append(t)
    return final_texts

def read_stopwords(path):
    """
    Reads a stopwords file and returns stopwords as a list (not used).
    """
    with open(path, encoding='utf8') as f:
        content = f.readlines()
        stopwords = [x.strip() for x in content] 
    return stopwords

def find_most_common(token_list):
    """
    Expects: a list of tokens.
    Returns:    Index over all words (dict), starting at 2, sorted by frequency 
                Most frequent words (list) - in this case words which appear more often than 10 times
                Max_words (int, counter of mfws+2)
    """
    mfws = []
    cnt = Counter(token_list)
    for key, value in dict(cnt.most_common()).items():
        if value > 10:
            mfws.append(key)
    max_words = len(mfws)+2
    index = {value:index+2 for index, value in enumerate(dict(cnt.most_common()).keys())} 
    return index, mfws, max_words

def fit_index_on_texts(index, texts, stopwords, mfws):
    """
    Expexts a list of texts (which are a list of tokens).
    Substitutes tokens with their corresponding index if the token belongs to the most frequent words
    (and does not belong to the stopwords).
    Tokens which are not in mfws are substituted with the value 1.
    """
    for doc in texts:
        for i, token in enumerate(doc):
            if token in mfws and token not in stopwords and token !='rt' and token !='yankees' and token !='charlie':           
                new = index[token]         
                doc[i] = new
        
            else:             
                doc[i] = 1 #out of mfws = 1, padding = 0
    return texts

def reshape(sequences, labels, maxlen):
    """
    Pads a list of sequences and converts sequences and list of labels to array. Padding is defined by the parameter 'maxlen'.
    Returns new arrays.
    """
    data = pad_sequences(sequences, maxlen=maxlen)
    labels = np.asarray(labels)

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    return data, labels

def reshape_characters(texts, labels, maxlen):
    """
    Pads a list of sequences and converts sequences and list of labels to array. Padding is defined by the parameter 'maxlen'.
    Returns new arrays.
    """
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    #one_hot = tokenizer.texts_to_matrix(texts, mode='binary')
    word_index=tokenizer.word_index

    data = pad_sequences(sequences, maxlen=maxlen)
    labels = np.asarray(labels)

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    return data, labels, word_index

def create_train_and_test_data(data, labels, cnt, training_size=5000, testing_size=1000):
    
    # sort to then pick from classes
    indices  = np.argsort(labels)
    data=data[indices] 
    labels=labels[indices]
    
    class2d = data[-cnt[2]:]
    class1d = data[cnt[0]:cnt[0]+cnt[1]]
    class0d = data[:cnt[0]]
    class2l = labels[-cnt[2]:]
    class1l = labels[cnt[0]:cnt[0]+cnt[1]]
    class0l = labels[:cnt[0]]
    
    d0, l0 = shuffle(class0d, class0l) #shuffle individually
    d1, l1 = shuffle(class1d, class1l)
    d2, l2 = shuffle(class2d, class2l)
    
    x_train=np.concatenate((d0[:training_size], d1[:training_size], d2[:training_size])) #split into sets
    y_train=np.concatenate((l0[:training_size], l1[:training_size], l2[:training_size]), axis = None)
    x_train, y_train = shuffle(x_train, y_train) #shuffle again
    x_test=np.concatenate((d0[training_size:training_size+testing_size], d1[training_size:training_size+testing_size], d2[training_size:training_size+testing_size]))
    y_test=np.concatenate((l0[training_size:training_size+testing_size], l1[training_size:training_size+testing_size], l2[training_size:training_size+testing_size]))
    print('Shape of data tensor:', x_train.shape)
    print('Shape of label tensor:', y_train.shape)
    
    return x_train, y_train, x_test, y_test

def shuffle(data, labels):
    """
    Shuffles a set of corresponding data and labels (np.arrays) and returns them.
    """
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data=data[indices] 
    labels=labels[indices]
    return data, labels


def one_hot_gen(texts, labels, maxlen, max_words, n=10):
    labels_array = []

    result = np.zeros(shape=(n, maxlen, max_words))

    while 1:

        i = 0
        labels_i = 0
        for sample in texts:

            labels_array.append(labels[labels_i])
            labels_i += 1
            for j, character in enumerate(sample):
                index = character

                result[i, j, index] = 1.
            i += 1

            if len(labels_array) >= n:
                labels_array = np.asarray(labels_array)

                yield result, labels_array
                i = 0
                result = np.zeros(shape=(n, maxlen, max_words))
                labels_array = []


def one_hot(texts, maxlen, max_words):
    results = np.zeros((len(texts), maxlen, max_words))
    for i, sample in enumerate(texts):
        # sample = sample.lower()
        for j, character in enumerate(sample):
            index = character

            results[i, j, index] = 1.

    return results