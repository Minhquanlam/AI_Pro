# -*- coding: utf-8 -*-
Import pickle

model=pickle.load(open("spam.pkl","rb")
cv=pickle.load(open("vectorizer.pkl","rb")
               
               
msg=" "
data=[msg]
vect=cv.transform(data).toarray()
prediction=model.predict(vect)
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

