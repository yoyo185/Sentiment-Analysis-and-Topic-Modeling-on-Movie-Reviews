# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:04:56 2018

@author: xieyy
"""

import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
from pprint import pprint
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
import itertools
from collections import Counter
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer

import plotly
plotly.tools.set_credentials_file(username='yoyo185', api_key='BDYwvwGcBeAlMKnmfc4t')

import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go

def main():
    #load the train data
    train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3) 
    
    #load the test data
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3) 
    
    # Prepare the stopwords
    stopwords_given = stopwords.words('english')
    stopwords_selected = ['one','would','could','said','may','might',
             'shall','first','also','must','should','instead','year','make',
             'years','take','takes','two','three','without','turn','turns','lives',
             'life','live','story','stories','film','find','finds','set','sets',
             'get','gets','br','movie','even','like','time','see','think','look']
    stopword = stopwords_given + stopwords_selected
    #build the documents dataframe 
    data_text = test[['review']]
    data_text['index'] = data_text.index
    documents = data_text
    
    __spec__ = None
    
    #call the function 
    bow_corpus,dictionary = filter_out(documents)
    #call the function
    documents = lda(bow_corpus,dictionary,documents)
    #call the function
    topic_counts(documents)
    #call the function 
    wcloud(train,stopword)
    
#define the function to lemmatize the text
def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
#define the function to preprocess the text
def preprocess(text):
    result = []
    select = ['like','watch','good','time','people','see','know','great']
    for token in gensim.utils.simple_preprocess(text):
        if token not in select:
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
    return result

#bag of words on the dataset
def filter_out(documents):
    #process the headline text, saving the results as 'processed_docs'
    processed_docs = documents['review'].map(preprocess)
    #processed_docs[:10]
    #bag of words
    dictionary = gensim.corpora.Dictionary(processed_docs)

    #less than 15 documents more than 0.5 documents, the first 100000 most frequent tokens
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    #for each document create how many words how many times appear
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]  
    return bow_corpus,dictionary

############################ Use LDA to do topic modeling
def lda(bow_corpus,dictionary,documents):
    #create tf-idf model object using models
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
        
    ##run LDA using bag of words
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=6, id2word=dictionary, passes=2, workers=2)    
    print('\n#############run LDA using bag of words#################\n')
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
    
    ##run LDA using TF-IDF
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=6, 
                                               id2word=dictionary, passes=2, workers=4)
    print('\n###############run LDA using TF-IDF######################\n')
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {}  \nWord: {}'.format(idx, topic))
        
    #since TF-IDF does not perform well, so we decided to use bag of words 
    #add a new list to save the number
    Category = []
    #testing model on the documents
    for i in range(0,25000):
        level = []
        topic = []
        #testing model on unseen document
        for index, score in lda_model[bow_corpus[i]]:
            #print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
            level.append(score)
            topic.append(index)
        Max = max(level)
        levelcate = level.index(Max)
        c = topic[levelcate]
        Category.append(c)
    documents['Cate'] = Category
    documents.to_csv("level.csv",index=False) 
    return documents

def topic_counts(documents):
    gp = documents.groupby('Cate')
    a = gp.size()
    #build a new data frame
    df1 = pd.DataFrame(a[0:6])
    df1.columns = ['frequency']
    # create trace4 
    trace = go.Bar(
                    x = np.arange(0,6),
                    y = df1['frequency'],
                    name = "movie reviews frequency in each category",
                    marker = dict(color = 'rgba(125, 255, 255, 0.5)',
                                  line=dict(color='rgb(0,0,0)',width=1.5)))
    data = [trace]
    # Add axes and title
    layout = go.Layout(
        	title = "movie reviews frequency in each category",
        	xaxis=dict(
        		title = 'topic'
        	),
        	yaxis=dict(
        		title = 'frequency'
        	)
        )
    fig = go.Figure(data = data, layout = layout)
    py.plot(fig)
#https://plot.ly/~yoyo185/28/movie-reviews-frequency-in-each-category/#/


def wcloud(train,stopword):
    pos = train.loc[train['sentiment']==1,:]
    pos = pos.reset_index(drop=True)
    #pos['review']
    postext = ' '.join(i for i in pos['review'])
   
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                max_words=200,
                stopwords=stopword,
                min_font_size = 10).generate(postext) 
  
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
  
    plt.show()
    # Save the graphs
    wordcloud.to_file('wordcloudpos'+'.png') 
    ###negative
    neg = train.loc[train['sentiment']==0,:]
    neg = neg.reset_index(drop=True)
    #pos['review']
    negtext = ' '.join(i for i in neg['review'])
   
    wordcloud2 = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                max_words=200,
                stopwords=stopword,
                min_font_size = 10).generate(negtext) 
  
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud2) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
  
    plt.show()
    # Save the graphs
    wordcloud2.to_file('wordcloudneg'+'.png') 
  
if __name__ == "__main__":
     main()
