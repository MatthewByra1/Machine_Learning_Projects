#!/usr/bin/env python
# coding: utf-8

# # Extracting Abstract Keywords using NLP
# ### NIPS dataset is used  from kaggle
# ### NLTK for the NLP tools

# ## DATA PREPROCESSING

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv('Downloads/papers.csv')
dataset.head()
df_remove = dataset[dataset['abstract'] == 'Abstract Missing'].index
dataset.drop(df_remove, inplace=True)
dataset.drop(columns = ['event_type'], inplace=True)
dataset.head()


# ### Gather the word count for the paper abstracts

# In[3]:


dataset['word_count'] = dataset['abstract'].apply(lambda x: len(str(x).split(" ")))
dataset[['abstract','word_count']].head()


# In[4]:


dataset.word_count.describe()


# In[5]:


#Word frequency
word_frequency = pd.Series(' '.join(dataset['abstract']).split()).value_counts()
word_frequency


# In[42]:


# Installing and import required NLP libraries
get_ipython().system('pip install nltk')
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')


# In[43]:


# Remove unneccesary and irrelevant words
stop_words = set(stopwords.words("english"))
custom_stopwords = ["using", "show", "result", "large", "also", "over", "one", "two", "new", 
                     "common", "among","mean","look","mostly"]
stop_words = stop_words.union(custom_stopwords)
stop_words


# In[67]:


num_rows = dataset.shape[0]
corpus = []
for i in range(num_rows):
    text = re.sub('[^a-zA-Z]', ' ', str(dataset.iloc[i,4]))
    text = text.lower()
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    text = re.sub("(\\d|\\W)+"," ",text)
    text = text.split()
    ps=PorterStemmer() 
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  
            stop_words] 
    text = " ".join(text)
    corpus.append(text)
corpus[0]


# # Feature Extraction

# In[95]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix


# In[100]:


count_vectors = CountVectorizer(max_df=0.7,stop_words=stop_words, max_features=5000, ngram_range=(1,2))
X = count_vectors.fit_transform(corpus)


# In[101]:


#Most frequently occuring Tri-grams
def top3words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
#Representation using bargraph
top3_words = top3words(corpus, n=20)
top3_df = pd.DataFrame(top3_words)
sns.set(style="whitegrid")
top3_df.columns=["Tri-gram", "Freq"]
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df, palette="Blues_d")
j.set_xticklabels(j.get_xticklabels(), rotation=45)


# ## Refining word counts using TF-IDF to find context relevant keywords

# In[102]:


tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)
feature_names = count_vectors.get_feature_names()
abstract = corpus[-1] 
tf_idf_vector = tfidf_transformer.transform(count_vectors.transform([abstract]))


# In[105]:


#function for sorting the transformations
def coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True) 

def topn(feature_names, sorted_items, topn):    
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        score_vals.append(round(score, 2))
        feature_vals.append(feature_names[idx])
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]  
    return results

sorted_items = coo(tf_idf_vector.tocoo())
keywords = topn(feature_names,sorted_items, 10)


# # Results

# In[106]:


print("\nAbstract:")
print(abstract)
print("\nKeywords:")
for k in keywords:
    print(k,keywords[k])


# In[ ]:





# In[ ]:




