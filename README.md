# Email/SMS Spam Classifier

A replayable machine learning project to classify messages as `spam` or `ham` using a Multinomial Naive Bayes model.

## Setup

1. Make sure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the model
Run the `train.py` script. It will read data from `spam.csv`, train a Multinomial Naive Bayes model using scikit-learn, print the accuracy metrics, and save the model to `spam_classifier_model.pkl`.

```bash
python train.py
```

### 2. Predict new messages (Replayable)
Run the `predict.py` script. It will load the trained model and open an interactive prompt where you can type or paste new messages to see if they are classified as spam or ham.

```bash
python predict.py
```
Type `exit` or `quit` to stop the interactive prediction script.

## Project Structure
- `spam.csv`: The dataset containing labeled messages (ham/spam).
- `train.py`: The script to train the model and save it.
- `predict.py`: The interactive script to classify new messages.
- `requirements.txt`: Python dependencies.
- `spam_classifier_model.pkl`: The saved model (generated after running `train.py`).
