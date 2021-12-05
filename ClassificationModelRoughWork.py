from ast import literal_eval
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB

vec = CountVectorizer()
model = MultinomialNB()
gnb = GaussianNB()
ber = BernoulliNB()


df_train = pd.read_csv('cleaned_datasets/cleaned_training_data.csv', header='infer', low_memory=False, converters={'Text_As_Word_List' : literal_eval, 'Text_As_Freq_Dict' : literal_eval})
df_test = pd.read_csv('cleaned_datasets/cleaned_test_data.csv', header='infer', low_memory=False, converters={'Text_As_Word_List' : literal_eval, 'Text_As_Freq_Dict' : literal_eval})

x = df_train['Text']
y = df_train['Class']

x, x_test, y, y_test= train_test_split(x, y, stratify=y, test_size=0.15, random_state=33)

x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

model.fit(x, y)

print(model.score(x_test, y_test))
