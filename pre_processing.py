'''
    File name         : pre_processing.py
    File Description  : NLP pre processing routines
    Author            : Srini Ananthakrishnan
    Date created      : 12/14/2016
    Date last modified: 12/14/2016
    Python Version    : 2.7
'''

# Standard imports
import numpy as np
import pandas as pd
import re

# NLTK imports
from nltk import clean_html
from nltk import SnowballStemmer
from nltk import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# GraphLab imports
import graphlab as gl

def tokenize(txt):
    """Function computes Tokenizes into sentences, strips punctuation/abbr, 
       converts to lowercase and tokenizes words
    Args:
        txt  : text documents
    Return:
        The return value. Tokenized words
    """
    return [word_tokenize(" ".join(re.findall(r'\w+', t,flags = re.UNICODE | re.LOCALE)).lower()) 
                for t in sent_tokenize(txt.replace("'", ""))]

def remove_stopwords(word_list, lang='english'):
    """Function removes english stopwords
    Args:
        word_list  : list of words
    Return:
        The return value. List of words
    """
    stopwords_list = stopwords.words(lang)
    content = [w for w in word_list if w.lower() not in stopwords_list]
    return content

def stemming(words_list, type="PorterStemmer", lang="english", encoding="utf8"):
    """Function stems all words with stemmer type
    Args:
        word_list  : list of words
    Return:
        The return value. Encoded list of words
    """
    supported_stemmers = ["PorterStemmer","SnowballStemmer","LancasterStemmer","WordNetLemmatizer"]
    if type is False or type not in supported_stemmers:
        return words_list
    else:
        encoded_list = []
        if type == "PorterStemmer":
            stemmer = PorterStemmer()
            for word in words_list:
                encoded_list.append(stemmer.stem(word).encode(encoding))
        if type == "SnowballStemmer":
            stemmer = SnowballStemmer(lang)
            for word in words_list:
                encoded_list.append(stemmer.stem(word).encode(encoding))
        if type == "LancasterStemmer":
            stemmer = LancasterStemmer()
            for word in words_list:
                encoded_list.append(stemmer.stem(word).encode(encoding))
        if type == "WordNetLemmatizer":
            wnl = WordNetLemmatizer()
            for word in words_list:
                encoded_list.append(wnl.lemmatize(word).encode(encoding))
        return encoded_list

def preprocess_pipeline(txt, lang="english", stemmer_type="WordNetLemmatizer"):
    """Function performs preprocess pipeline
    Args:
        txt  : text documents
    Return:
        The return value. Pre-processed word list
    """
    lemmetized = []
    words = []
    # tokenize input text
    sentences = tokenize(txt)
    # for each sentence remove stopwords and perform stemming
    for sentence in sentences:
        words = remove_stopwords(sentence, lang)
        words = stemming(words, stemmer_type)
        # lets's skip short sentences with less than 3 words
        if len(words) < 3:
            continue
        lemmetized.append(" ".join(words))
        return " ".join(lemmetized)
       
def load_json_file_to_sframe(filename):
    """Function loads jason file to graphLab sFrame
    Args:
        filename  : filename to load
    Return:
        The return value. sFrame
    """
    # Read the entire file into a SFrame with one row
    sf = gl.SFrame.read_csv(filename, delimiter='\n', header=False)
    
    # The dictionary can be unpacked to generate the individual columns.
    sf = sf.unpack('X1', column_name_prefix='')
    return sf

def load_jason_file_to_df(filename):
    """Function loads jason file to pandas data frame
    Args:
        filename  : filename to load
    Return:
        The return value. pandas DF
    """
    # read the entire file into a python array
    with open(filename, 'rb') as f:
        data = f.readlines()
    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ','.join(data) + "]"
    return pd.read_json(data_json_str)

def balance_data(reviews,num_to_balance):
    """Function class balance (under sample) reviews using CV
    Args:
        reviews        : reviews document
        num_to_balance : number of documents to balance
    Return:
        The return value. Balanced reviews docuemnts (sFrame)
    """
    indices = []
    # get the indices for each class
    for rating in range(1, 6):
        index = reviews['overall'] == rating
        indices.append(index)
    # using cross validation under sample reviews
    reviews_balanced = gl.toolkits.cross_validation.shuffle(reviews[indices[0]])[:num_to_balance]
    for index in indices[:]:
        sFrame = gl.toolkits.cross_validation.shuffle(reviews[index], random_seed=0)[:num_to_balance]
        reviews_balanced = reviews_balanced.append(sFrame)
    reviews_balanced = gl.toolkits.cross_validation.shuffle(reviews_balanced)
    return reviews_balanced
