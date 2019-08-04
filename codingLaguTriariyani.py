import pandas as pd
import numpy as np
import sys  

spam_data = pd.read_csv('/resources/data/datalaguTriariyani.csv')
spam_data['label'] = np.where(spam_data['label']=='positif',1,0)
print(spam_data.shape)
spam_data.head(20)


import matplotlib.pyplot as plt
import seaborn as sns
spam_data["label"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Positif vs Negatif")
plt.legend(["positif", "negatif"])
plt.show()


from sklearn.model_selection import train_test_split
#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(spam_data['text'],spam_data['label'],random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
#Train and evaluate the model
vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)
clfrNB = MultinomialNB(alpha = 0.1)
clfrNB.fit(X_train_vectorized, y_train)
preds = clfrNB.predict(vect.transform(X_test))
score = roc_auc_score(y_test, preds)
print(score)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, preds))


from sklearn.metrics import confusion_matrix
import numpy as np
target = np.array([1,0,1,0])
prediksi =  np.array([1, 1, 1, 0])

tn, fp, fn, tp = confusion_matrix(target, prediksi).ravel()
tn = float(tn)
fp = float(fp)
fn = float(fn)
tp = float(tp)
print (tn, fp, fn, tp)

akurasi = ((tp+tn)/(tp+tn+fp+fn))*100
presisi = (tp/(fp+tp))*100
recall = (tp/(fn+tp))*100
print (akurasi,presisi, recall)