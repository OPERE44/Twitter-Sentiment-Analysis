# Twitter-Sentiment-Analysis 

## 1. Problem Statement
The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label ‘1’ denotes the tweet is racist/sexist and label ‘0’ denotes the tweet is not racist/sexist, the objective is to predict the labels on the given test dataset.

## 2. Tweets Preprocessing and Cleaning
The preprocessing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it. If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.


### A) Removing Twitter Handles (@user)
Removed all these twitter handles from the data as they don’t convey much information. For convenience, let’s first combine train and test set. This saves the trouble of performing the same steps twice on test and train.

### B) Removing Punctuations, Numbers, and Special Characters
Punctuations, numbers and special characters do not help much. It is better to remove them from the text just as the twitter handles. Then replace everything except characters and hashtags with spaces.

### C) Removing Short Words
A little careful here in selecting the length of the words to be removed. So, I have decided to remove all the words having length 3 or less. For example, terms like “hmm”, “oh” are of very little use. It is better to get rid of them.


### D) Tokenization
Now tokenize all the cleaned tweets in the dataset. Tokens are individual terms or words, and tokenization is the process of splitting a string of text into tokens.

## E) Stemming
Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word

## 3. Story Generation and Visualization from Tweets
    1. Exploring and visualizing data, no matter whether its text or any other data, is an essential step in gaining insights
    2. What are the most common words in the entire dataset?
    3. What are the most common words in the dataset for negative and positive tweets, respectively?
    4 .How many hashtags are there in a tweet?
    5. Which trends are associated with my dataset?
    6. Which trends are associated with either of the sentiments? Are they compatible with the sentiments?

### A) Understanding the common words used in the tweets: WordCloud
Now see how well the given sentiments are distributed across the train dataset. One way to accomplish this task is by understanding the common words by plotting wordclouds.
##### A wordcloud
is a visualization wherein the most frequent words appear in large size and the less frequent words appear in smaller sizes.

Hence, we plot separate wordclouds for both the classes(racist/sexist or not) in our train data.

### D) Understanding the impact of Hashtags on tweets sentiment
Hashtags in twitter are synonymous with the ongoing trends on twitter at any particular point in time. try to check whether these hashtags add any value to our sentiment analysis task, i.e., they help in distinguishing tweets into the different sentiments

## 4. Extracting Features from Cleaned Tweets
To analyze a preprocessed data, it needs to be converted into features. Depending upon the usage, text features can be constructed using assorted techniques – Bag-of-Words, TF-IDF, and Word Embeddings. In this project, I covered only Bag-of-Words and TF-IDF.

## 5. Model Building: Sentiment Analysis
 Building predictive models on the dataset using the two feature set — Bag-of-Words and TF-IDF
    
