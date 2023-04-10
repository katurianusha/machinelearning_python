Data set creation
--------------
import pandas as pd
import numpy as np
from google.colab import files

df = pd.read_csv ('https://raw.githubusercontent.com/katurianusha/machinelearning_python/main/Tatabinge-trainmodel-new.csv')
df -- to print 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import FileLink

train_labels = df['Label'][:210]
test_data = df['Review'][210:]
test_labels = df['Label'][210:]
train_data = df['Review'][:210]
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, train_labels)
predicted_labels_test = nb_classifier.predict(X_test)
predicted_labels_train = nb_classifier.predict(X_train)
print(predicted_labels_test)

all_labels = np.concatenate((predicted_labels_train, predicted_labels_test))
df1 = pd.DataFrame({'Review': test_data, 'Predicted Label': predicted_labels_test})
df1.to_csv('output.csv', index=True)
files.download('output.csv')

