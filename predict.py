import joblib
import sys

def main():
    try:
        model = joblib.load('spam_classifier_model.pkl')
    except FileNotFoundError:
        print('Model file not found. Please run train.py first.')
        sys.exit(1)

    print('--- Email/SMS Spam Classifier ---')
    print('Type your message below and press Enter to classify.')
    print('Type exit or quit to stop.')

