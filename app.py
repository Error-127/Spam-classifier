import gradio as gr
import joblib

# Load the trained model
try:
    model = joblib.load('spam_classifier_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'spam_classifier_model.pkl' exists and run train.py if it doesn't.")
    exit(1)

def classify_email(text):
    if not text.strip():
        return "Please enter some text."
    
    prediction = model.predict([text])[0]
    
    if prediction.lower() == 'spam':
        return "🚨 This looks like **SPAM**!"
    else:
        return "✅ This looks like **HAM** (Safe)."

# Create the Gradio interface
interface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=7, placeholder="Paste your email or SMS message here...", label="Email / SMS Content"),
    outputs=gr.Markdown(label="Prediction Result"),
    title="Email & SMS Spam Classifier",
    description="A machine learning model trained with Multinomial Naive Bayes. Paste an email or text message below to check if it's spam or not.",
    theme="default",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()
