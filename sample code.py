import numpy as np
import pandas as pd


df=pd.read_csv('C:/Talent Battle/6 Weeks Project Challenge/Stress Management using ML/stress.csv')
df.head()


df.describe()


df.isnull()


df.isnull().sum()

#So, this dataset does not have any null values. 
#Now we will prepare the text column of this dataset to clean the 
#text column with stopwords, links, special symbols and language errors:


'''
import nltk
import re
from nltk. corpus import stopwords
import string
nltk. download( 'stopwords' )
stemmer = nltk. SnowballStemmer("english")
stopword=set (stopwords . words ( 'english' ))

def clean(text):
    text = str(text) . lower()  #returns a string where all characters are lower case. Symbols and Numbers are ignored.
    text = re. sub('\[.*?\]',' ',text)  #substring and returns a string with replaced values.
    text = re. sub('https?://\S+/www\. \S+', ' ', text)#whitespace char with pattern
    text = re. sub('<. *?>+', ' ', text)#special char enclosed in square brackets
    text = re. sub(' [%s]' % re. escape(string. punctuation), ' ', text)#eliminate punctuation from string
    text = re. sub(' \n',' ', text)
    text = re. sub(' \w*\d\w*' ,' ', text)#word character ASCII punctuation
    text = [word for word in text. split(' ') if word not in stopword]  #removing stopwords
    text =" ". join(text)
    text = [stemmer . stem(word) for word in text. split(' ') ]#remove morphological affixes from words
    text = " ". join(text)
    return text
df [ "text"] = df["text"]. apply(clean)


'''



'''
import matplotlib. pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ". join(i for i in df. text)
stopwords = set (STOPWORDS)
wordcloud = WordCloud( stopwords=stopwords,background_color="white") . generate(text)
plt. figure(figsize=(15, 10) )
plt. imshow(wordcloud, interpolation='bilinear' )
plt. axis("off")
plt. show( )

'''


from sklearn. feature_extraction. text import CountVectorizer
from sklearn. model_selection import train_test_split

x = np.array (df["text"])
y = np.array (df["label"])

cv = CountVectorizer ()
X = cv. fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33,random_state=42)



#As this task is based on the problem of binary classification, 
#We will be using the Bernoulli Naive Bayes algorithm, 
#which is one of the best algorithms for binary classification problems.


from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtrain,ytrain)



#let’s test the performance of our model on some 
#random sentences based on mental health

user=input("Enter the text")
data=cv.transform([user]).toarray()
output=model.predict(data)
if (output==0):
    print("Not in Stress")
else:
    print("You are in Stress")



