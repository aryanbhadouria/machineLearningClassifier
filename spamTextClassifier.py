
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from predict import train_and_save_model

data_url = "https://raw.githubusercontent.com/ogozuacik/sms_spam_classification/main/SMSSpamCollection"
data = pd.read_csv(data_url, sep='\t', names=["label", "message"])

# Preprocess data
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

train_and_save_model(data)


X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
