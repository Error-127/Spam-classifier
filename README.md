# 📧 Spam Classifier

A machine learning web app that classifies a message as **Spam** or **Not Spam** (Ham) in real time.

🔗 **Live Demo:** [https://spam-classifier-qfvu.onrender.com/](https://spam-classifier-qfvu.onrender.com/)

> ⚠️ Hosted on Render's free plan, so the first load may take 30-60 seconds if the app has been idle.

---

## 📸 Screenshot

![App Screenshot](screenshot.png)

---

## 🚀 Features

- Enter any email or SMS text and get an instant prediction
- Text preprocessing (lowercasing, tokenization, stopword removal, stemming)
- TF-IDF feature extraction
- Simple and clean web interface built with Streamlit

---

## 🧠 How It Works

1. **Input:** the user enters a message
2. **Preprocessing:** the text is cleaned and normalized
3. **Vectorization:** the text is converted to numbers using TF-IDF
4. **Prediction:** the trained model classifies it as Spam or Not Spam

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | [97%] |
| Precision | [100%] |
| Recall | [85%] |
| F1-Score | [92%] |

**Algorithm used:** [Multinomial Naive Bayes]
**Dataset:** [SMS Spam Collection Dataset from Kaggle / UCI]

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- NLTK
- Pandas, NumPy
- Streamlit
- Render (deployment)

---

## 📁 Project Structure

```
spam-classifier/
├── app.py              # Streamlit web app
├── model.pkl           # Trained model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # Dependencies
└── README.md
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Error-127/Spam-classifier.git

# 2. Go to the project folder
cd Spam-classifier

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🔮 Future Improvements

- Try other models (SVM, Random Forest, deep learning)
- Add support for more languages
- Show the prediction confidence score

---

## 👤 Author

**Saurabh**
GitHub: [@Error-127](https://github.com/Error-127)
LinkedIn: [your-linkedin-link]

⭐ If you found this project useful, please give it a star!
