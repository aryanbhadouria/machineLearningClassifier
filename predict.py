import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_and_save_model(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['message'])
    y = data['label']

    clf = MultinomialNB()
    clf.fit(X, y)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)

def predict_spam(text):
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("model.pkl", "rb") as f:
        clf = pickle.load(f)

    text_vectorized = vectorizer.transform([text])
    prediction = clf.predict(text_vectorized)

    return "spam" if prediction[0] == 1 else "ham"
