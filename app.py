import streamlit as st
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import numpy as np
import re

# Load model (change to your own local model if needed)
model_name = "./fake_news_model"
classifier = pipeline("text-classification", model=model_name, return_all_scores=True)

# Class label mapping
label_map = {
    "LABEL_0": "Real",
    "LABEL_1": "Fake",
    "ham": "Real",
    "spam": "Fake"
}

# Lime explainability
explainer = LimeTextExplainer(class_names=list(set(label_map.values())))

# Predict function for LIME
def predict_proba(texts):
    results = classifier(texts)
    return np.array([[r['score'] for r in result] for result in results])

# Streamlit App UI
st.title("📰 Fake News Detector with Explainability")
st.markdown("**Model Label Legend:**")
st.markdown("- 🟩 Positive weight (supports prediction)")
st.markdown("- 🟥 Negative weight (opposes prediction)")
st.markdown("- 🎯 Final prediction shown below")

input_text = st.text_area("Enter a news statement to verify:")

if st.button("Predict"):
    if input_text.strip() != "":
        # Run classifier
        all_scores = classifier(input_text)[0]
        top_pred = max(all_scores, key=lambda x: x['score'])
        label = top_pred['label']
        readable_label = label_map.get(label, label)
        confidence = top_pred['score']

        st.markdown(f"### 🎯 Prediction: **{readable_label.upper()}** (Confidence: `{confidence:.2f}`)")

        # LIME Explanation
        exp = explainer.explain_instance(input_text, predict_proba, num_features=10)

        # Extract weights for heatmap
        weights = dict(exp.as_list())
        tokens = re.findall(r"\w+|[^\w\s]", input_text, re.UNICODE)

        def color_token(token):
            raw = token.lower()
            weight = weights.get(raw, 0)
            red = int(min(255, max(0, 255 - (weight * 500 if weight < 0 else 0))))
            green = int(min(255, max(0, 255 - (-weight * 500 if weight > 0 else 0))))
            blue = 180  # keep it consistent
            return f"<span style='background-color: rgba({red}, {green}, {blue}, 0.6); padding: 2px; border-radius: 4px'>{token}</span>"

        st.markdown("### 🔥 Heatmap Explanation:")
        highlighted_text = " ".join(color_token(tok) for tok in tokens)
        st.markdown(f"<p>{highlighted_text}</p>", unsafe_allow_html=True)

        # Optional: Print raw word weights
        st.markdown("### 🧠 Top Influential Words:")
        for word, weight in exp.as_list():
            bar = "🟩" if weight > 0 else "🟥"
            st.markdown(f"{bar} **{word}** → *{weight:.3f}*")

    else:
        st.warning("Please enter a news statement.")
