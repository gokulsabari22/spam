import pandas as pd

df = pd.read_csv('spam_ham_dataset.csv')
df.head()
df = df.drop(['Unnamed: 0', 'label'], axis=1)
c = df.isnull().sum()
print(c)
import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []
for i in range(len(df)):
    sentence = re.sub('[^A-Za-z]', ' ', df['text'][i])
    sentence = sentence.lower()
    sentence = sentence.split()
    sentence = [ps.stem(words) for words in sentence if words not in stopwords.words('english')]
    sentence = ' '.join(sentence)
    corpus.append(sentence)

from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer()
X = vector.fit_transform(corpus)
y = df['label_num']
X_df = pd.DataFrame(X.todense()) # Total number of columns 37890. Rows 5171
print(X_df)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train, y_train)
predict = clf.predict(X_test)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(predict, y_test))
print(confusion_matrix(predict, y_test))






