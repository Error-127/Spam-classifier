import gradio as gr
import joblib

import os

# Load the trained model or train if not present
MODEL_FILE = 'spam_classifier_model.pkl'

def get_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    if os.path.exists('spam.csv'):
        import pandas as pd
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import make_pipeline

        print("Training model from spam.csv...")
        df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
        df.columns = ['label', 'text']
        trained_model = make_pipeline(CountVectorizer(), MultinomialNB())
        trained_model.fit(df['text'], df['label'])
        joblib.dump(trained_model, MODEL_FILE)
        return trained_model
    raise FileNotFoundError("Could not find spam_classifier_model.pkl or spam.csv")

model = get_model()

def classify_email(text):
    if not text.strip():
        return "<h3 style='color: gray;'>Please enter some text to classify.</h3>"
    
    prediction = model.predict([text])[0]
    
    if prediction.lower() == 'spam':
        return "<h2 style='color: #e74c3c;'>🚨 Non Replyable</h2><p style='color: #c0392b;'>This message seems suspicious or automated and likely does not require a reply.</p>"
    else:
        return "<h2 style='color: #2ecc71;'>✅ Replyable</h2><p style='color: #27ae60;'>This is a legitimate message that you should consider replying to.</p>"

# Create a custom, better-looking UI using Gradio Blocks
with gr.Blocks() as interface:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1 style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem; color: #2c3e50;">Email Replyable & Non Replyable</h1>
            <p style="font-size: 1.1rem; color: #7f8c8d;">Paste your email or text message below to check if it's worth your time replying.</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                lines=8, 
                placeholder="Paste the email or SMS content here...", 
                label="Message Content",
                show_label=False
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Analyze Message", variant="primary")
                
        with gr.Column(scale=1):
            gr.Markdown("### Analysis Result")
            output_box = gr.HTML(
                value="<div style='text-align: center; padding: 2rem; color: #95a5a6; border: 2px dashed #bdc3c7; border-radius: 8px;'>Result will appear here</div>",
                label="Status"
            )

    submit_btn.click(classify_email, inputs=input_box, outputs=output_box)
    clear_btn.click(lambda: ("", "<div style='text-align: center; padding: 2rem; color: #95a5a6; border: 2px dashed #bdc3c7; border-radius: 8px;'>Result will appear here</div>"), inputs=None, outputs=[input_box, output_box])

if __name__ == "__main__":
    interface.launch(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"))
