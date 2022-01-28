# Project 3: Web APIs & NLP

### Problem Statement

Fake news is a prevalent and harmful problem in our modern society, often misleading the general public on important topics such as healthcare and defense. Lots of information online especially in social media feeds may appear to be true, often is not. Sometimes it becomes difficult to differentiate the fake news and authentic news. Fake news are more easily to manipulate from the internet and tend to spread faster than true stories ([*source*](https://news.mit.edu/2018/study-twitter-false-news-travels-faster-true-stories-0308)). This can lead to long standing societal issues which are a detriment to nations worldwide.


To tackle the problem, our team aims to develop a model using natural language processing and machine learning models to predict fake news or real news, helping government bodies/regular citizens to identify the fake news, thus creating a secure, and more misinformation-resilient society.

---

### Executive Summary

Fake news are information that are untrue and misleading.
Fake news generally can be categorized into ([*source*](https://firstdraftnews.org/articles/fake-news-complicated/)) :- <br>
1. Satire/Parody
2. Misleading Content
3. Imposter content
4. Fabricated content 
5. False Conection 
6. False Context
7. Manipulated content


For the data collection, we will extract information from two subreddit post, from 'theonion' and 'nottheonion'.
'r/Theonion' (https://www.reddit.com/r/TheOnion/) is a subreddit community that shares satire or parody types of fake news. While 'r/nottheonion' (https://www.reddit.com/r/nottheonion/) is a subreddit community that post true stories that are often easily classified as fake. 

The data was collected using pushsift.io API, containing 5,000 post from each subreddit. The posts was collected from the arcticle before January 01, 2022 00:00:00 UTC. 

Natural language processing technique is used to preprocess the posts collected. The unnecessary special characters such as are removed and sentences are tokenized using Regex tokenizer. We also tried to reduce the word into its root word by using lemmatization process. 


`CountVectorizer` and `TfidfVectorizer` were the two vectorizer used to convert text to numerical data for use as input to the machine learning algorithm. `CountVectorize` is used on model building as it provides better test and train score on the model compared to `TfidfVectorizer`. The best parameters used for `CountVectorizer` after hyperparameter tuning are {'cvec__max_df': 0.9,
 'cvec__max_features': None,
 'cvec__min_df': 0,
 'cvec__ngram_range': (1, 2),
 'cvec__stop_words': None}. <br> 

Data gathered are then splitted into 75% for training model and 25% as the unseen test set to evaluate the model performance. 
Five different types of of Machine Learning Classifiers were used to build the model which includes `RandomForestClassifier`,`MultinomialNB`,`LogisticRegression`, `KNearestNeighbors`, and `Support vector machines model(SVM)`.

---

### Conclusions

We have explore the NLP method to identify fake news and real news. The data collected from r/TheOnion or r/nottheonion has been processed by using methods such as word extration, tokenizing, and lemmatizing. The Support vector machines model(SVM) has the highest test score at 83.44% and F1 scores of 80.87%. LogisticRegression model also performed similarly compared to SVM with test score of 83.23% and F1 scores of 80.73% despite being a simpler model. KNN is the worse model to predict title from r/TheOnion or r/nottheonion with 44.87% f1 score only. The hyperparameter tuning for countvectorizer suggest to include stop word is a better parameter. <br>

Hence, we will choose `CountVectorizer` + `Support vector machines model(SVM)` model as it has the highest F1 scores that minimise false positives and false negatives. Also it is simple and easy to implement in order to provide accurate text classification predictions.



### Recommendations

The model does not include special characters such as emojis, non english words, or any other non alphanumeric characters. There could be some important information being ignored. Further preprocessing could be done on the subreddit title to capture more informations from text. <br> 

Furthermore, we could also try to model with stopword removed. As those common words could appear in both subreddit frequently. Add on more features such as subtext, comments and upvotes can be consider to train the model. We can also experiment with more advanced NLP teahniques such as BERT language model. 

---

### Data

* [`reddit_post.csv`](./datasets/reddit_post.csv): Original Posts collected from r/TheOnion and r/nottheonion.
* [`cleaned_reddit_post.csv`](./datasets/train_cleaned.csv): Cleaned Train dataset.<br>

---
### Data dictionary


|Feature|Type|Dataset|Description|
|---|---|---|---|
|**'title'**|*object*|reddit_post.csv|Title from subreddit post| 
|**selftext**|*object*|reddit_post.csv|Selftext| 
|**subreddit**|*object*|reddit_post.csv|Subreddit|
|**target**|*integer*|cleaned_reddit_post.csv|Target for model|
|**char_count**|*integer*|cleaned_reddit_post.csv|Character count per sentence from title|
|**word_count**|*integer*|cleaned_reddit_post.csv|Word count per sentence from title|
|**processed_title**|*object*|cleaned_reddit_post.csv|Processed title|
|**no_stopword_title**|*object*|cleaned_reddit_post.csv|Title without stopword|
|**processed_wordcount**|*object*|cleaned_reddit_post.csv|Word count per sentence from processed_title|
