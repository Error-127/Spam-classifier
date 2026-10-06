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

    while True:
        try:
            text = input('\nEnter message: ')
            if text.lower() in ['exit', 'quit']:
                break
            if not text.strip():
                continue
            prediction = model.predict([text])[0]
            if prediction.lower() == 'spam':
                result_text = "NON REPLYABLE"
            else:
                result_text = "REPLYABLE"
            print(f'Prediction: [{result_text}]')
        except KeyboardInterrupt:
            print('\nExiting...')
            break
        except Exception as e:
            print(f'An error occurred: {e}')

if __name__ == '__main__':
    main()
