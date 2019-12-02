# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:19:03 2019

@author: HP
"""
#Merge D1.csv and D2.csv
import os
import glob
import pandas as pd
os.chdir("C:/Users/HP/Desktop/File")    # Path location to directory

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]  # Add all file names with csv extension
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])  # Merge all files

data = combined_csv
data_text = data[['Tweet']]              # Save tweets into data_text 
data_text['index'] = data_text.index     # Provide index to each tweet
documents = data_text                    # Save tweets along with index to new object
print(len(documents))                    # Check number of tweets for verification
print(documents[:5])                     # Print first 5 tweets with indexes

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
stemmer = SnowballStemmer("english")


#Lemmatize and stem tweets
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

#Preprocess tweets and generate tokens
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        newStopWords = ['vaccine','vaccin','vacc','autism','rabi']
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in newStopWords and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# Sample tweet 
doc_sample = documents[documents['index']==4000].values[0][0]   # Check a sample tweet of index 4000
print('Original document:')               # Print original tweet
words = []
for word in doc_sample.split(' '):        # Split words in tweet
    words.append(word)
print(words)                              # Print words in tweet
print('\n\n tokenized and lemmatized document: ')       # Print processed sample tweet
print(preprocess(doc_sample))

# Apply pre-processing to all tweets
processed_docs = documents['Tweet'].fillna('').astype(str).map(preprocess)
processed_docs[:10]

# Create a dictionary of words in tweets and count occurence of each word
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v, dictionary.dfs[k])
    count += 1
    if count > 10:
        break

# Remove words apprearing in less than 100 times, appearing in more than half of the tweets
# Keep top 100000 words for further processing
dictionary.filter_extremes(no_below = 100, no_above = 0.5, keep_n = 100000)

# Create term document frequency
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Sample tweet to count frequency of words in corpus
bow_doc_4000 = bow_corpus[4000]
print(bow_doc_4000)
for i in range(len(bow_doc_4000)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4000[i][0],
          dictionary[bow_doc_4000[i][0]],
          bow_doc_4000[i][1]))
    
# Fit the LDA model for 2 topics
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics = 2, id2word = dictionary, 
                                       passes = 2, workers = 4)

# Print first 200 terms of LDA equation   
topics = lda_model.print_topics(num_words = 200)
for topic in topics:
    print(topic)

# Save the words with their beta scores for each topic in csv
with open('Topic_Pyfile.csv', 'a', encoding='latin-1') as outfile:
    for idx, topic in lda_model.print_topics(200):
        #outfile.write(str('Topic: {} + str('Words: {}'.format(idx,topic)) + '\n'))
        outfile.write('Topic: {}       Words: {}'.format(idx,topic) + '\n')
        

