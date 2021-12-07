from ast import literal_eval
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import csv

# building the sklearn classes
vec = CountVectorizer()
model = MultinomialNB()

# reading training and testing data from CSV files
df_train = pd.read_csv('cleaned_datasets/cleaned_training_data.csv', header='infer', low_memory=False, converters={'Text_As_Word_List' : literal_eval, 'Text_As_Freq_Dict' : literal_eval})
df_test = pd.read_csv('cleaned_datasets/cleaned_test_data.csv', header='infer', low_memory=False, converters={'Text_As_Word_List' : literal_eval, 'Text_As_Freq_Dict' : literal_eval})

# establishing our X and y columns for the model
x = df_train['Text']
y = df_train['Class']

# splitting our training data into training and predicting
x, x_test, y, y_test= train_test_split(x, y, stratify=y, test_size=0.15, random_state=33)

# vectorizing the text class
# https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

# building our model and outputting its accuracy
model.fit(x, y)
print(model.score(x_test, y_test))

# adding header for CSV file
# https://www.pythontutorial.net/python-basics/python-write-csv-file/
f = open('prediction.csv', 'w')
writer = csv.writer(f)
writer.writerow(['REVIEW-ID', 'CLASS'])

# predicting classes from the test data, outputing the results to a CSV file
for id, text in zip(df_test['ID'], df_test['Text_As_Word_List']):
    sentiment_count={
        'positive':0,
        'neutral':0,
        'negative':0
    }
    text = vec.transform(text).toarray()
    list_of_sentiment = model.predict(text)
    for word in list_of_sentiment:
        if word == 'positive':
            sentiment_count['positive'] += 1
        elif word == 'negative':
            sentiment_count['negative'] += 1
        elif word == 'neutral':
            sentiment_count['neutral'] += 1
    predicted_class = max(sentiment_count, key=sentiment_count.get)
    writer.writerow([id, predicted_class])
f.close()