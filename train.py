import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    print('Loading dataset...')
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    print('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    print('Training Multinomial Naive Bayes model...')
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
