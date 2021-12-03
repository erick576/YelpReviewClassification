from ast import literal_eval
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('cleaned_datasets/cleaned_training_data.csv', header='infer', low_memory=False, converters={'Text_As_Word_List' : literal_eval, 'Text_As_Freq_Dict' : literal_eval})
df_test = pd.read_csv('cleaned_datasets/cleaned_test_data.csv', header='infer', low_memory=False, converters={'Text_As_Word_List' : literal_eval, 'Text_As_Freq_Dict' : literal_eval})

x = df_train['Text_As_Freq_Dict']
y = df_train['Class']
# train_text_counts = cv.fit_transform(df_train)
# test_text_counts = cv.fit_transform(df_test)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)

mnb = MultinomialNB()
mnb.fit(x_train, y_train)

predmnb = mnb.predict(x_test)

print("Score:",round(accuracy_score(y_test,predmnb)*100,2))
print("Classification Report:",classification_report(y_test,predmnb))
print("Accuracy:",metrics.accuracy_score(X_test, Y_test))

# print(df_train['Text_As_Word_List'][0])
# print(df_train['Text_As_Word_List'][0][3])
# print('\n')
# print(df_train['Text_As_Freq_Dict'][0])
# print(df_train['Text_As_Freq_Dict'][0]['place'])
