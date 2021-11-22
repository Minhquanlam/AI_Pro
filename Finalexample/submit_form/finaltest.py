# -*- coding: utf-8 -*-
import pandas as pd
import sys
data = pd.read_csv('C:/Users/lmqua/Finalexample/submit_form/index.php', encoding="latin-1") 
# Ông copy path riêng đường link máy ông để đọc data vào spam.csv

import numpy as np
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

le = preprocessing.LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

x=data['EmailText']
y=data['Label']

cv = CountVectorizer()
x = cv.fit_transform(x).toarray()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.naive_bayes import GaussianNB
model1 = GaussianNB()
model1.fit(x_train, y_train)
y_predicted1 = model1.predict(x_test)
accuracy_score(y_test, y_predicted1)
print("Gaussian Naive Bayes accuracy score:\t", accuracy_score(y_test, y_predicted1) * 100, "%")

#Chay ouput
input=sys.argv[1]

input = cv.fit_transform(input).toarray()
prediction = model1.predict(input)

result=prediction[1]
if result==1:
    print('<h3>This is a spam mail</h3>')

else:
    print('<h3>This is a ham mail</h3>')





#from sklearn.naive_bayes import MultinomialNB
#model2 = MultinomialNB()
#model2.fit(x_train, y_train)
#y_predicted2 = model2.predict(x_test)
#print("Multinomial Naive Bayes accuracy score:\t", accuracy_score(y_test, y_predicted2) * 100, "%")


#from sklearn.naive_bayes import BernoulliNB
#model3 = BernoulliNB()
#model3.fit(x_train, y_train)
#y_predicted3 = model3.predict(x_test)
#print("Bernoulli Naive Bayes accuracy score:\t", accuracy_score(y_test, y_predicted3) * 100, "%")

