# streamlit_sentiment_app.py
# A simple, accessible Streamlit front-end for the sentiment model.
# Designed with large controls, icons and CSV batch support for non-readers/low-literacy users.

import streamlit as st
import pandas as pd
import re
import joblib
import io
import numpy as np

# -------------------------
# Helper functions
# -------------------------

def clean_text(text):
    """Minimal cleaning that matches training preprocessing."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(model, texts):
    """Return labels and (optional) probability/confidence for a list of texts."""
    clean_texts = [clean_text(t) for t in texts]
    try:
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(clean_texts)
            preds = model.predict(clean_texts)
        else:
            preds = model.predict(clean_texts)
        return preds, probs
    except Exception as e:
        st.error(f"Error while predicting: {e}")
        return None, None


# -------------------------
# UI layout & style
# -------------------------

st.set_page_config(page_title="Sentiment Studio", layout="wide")

# Large header for easy recognition
st.markdown("# 🌟 Sentiment Studio")
st.markdown("## Big buttons · Simple icons · CSV upload")
st.caption("A friendly app to test sentiment — designed for quick use and low-literacy accessibility.")

# Two-column layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Try one review")
    # large text area
    user_review = st.text_area("", placeholder="Type or paste a review here...", height=150, key="txt_input")

    # big predict button
    if st.button("🔍 Predict Sentiment", key="predict_btn"):
        if not user_review.strip():
            st.warning("Please write or paste a review to predict.")
        else:
            # load model
            try:
                model = joblib.load("sentiment_model.joblib")
            except FileNotFoundError:
                st.error("Model file 'sentiment_model.joblib' not found. Put it in the same folder as this app.")
                st.stop()

            preds, probs = predict_sentiment(model, [user_review])
            if preds is None:
                st.stop()

            label = preds[0]
            st.markdown("---")

            # Display large visual feedback for low-literacy users
            if label == "positive":
                st.markdown("<h1 style='color:green;font-size:64px'>👍 Positive</h1>", unsafe_allow_html=True)
            elif label == "negative":
                st.markdown("<h1 style='color:red;font-size:64px'>👎 Negative</h1>", unsafe_allow_html=True)
            else:
                st.markdown("<h1 style='color:orange;font-size:64px'>😐 Neutral</h1>", unsafe_allow_html=True)

            # show confidence if available
            if probs is not None:
                # get probability of predicted label
                classes = model.classes_
                pred_idx = list(classes).index(label)
                conf = probs[0][pred_idx]
                st.write(f"Confidence: {conf:.2%}")

with col2:
    st.subheader("Batch mode — Upload CSV")
    st.markdown("Upload a CSV with a column named `Text` containing reviews. A new file with predictions will be returned.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"]) 
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        if "Text" not in df_upload.columns:
            st.error("CSV must contain a 'Text' column. If your column is named differently, rename it to 'Text' or use the single review box.")
        else:
            st.write("Preview:")
            st.dataframe(df_upload.head(5))

            if st.button("Predict CSV", key="predict_csv"):
                try:
                    model = joblib.load("sentiment_model.joblib")
                except FileNotFoundError:
                    st.error("Model file 'sentiment_model.joblib' not found. Put it in the same folder as this app.")
                    st.stop()

                texts = df_upload["Text"].astype(str).tolist()
                preds, probs = predict_sentiment(model, texts)
                if preds is None:
                    st.stop()

                df_upload["Sentiment"] = preds

                # optionally add confidence column
                if probs is not None:
                    # pick probability of predicted class
                    classes = model.classes_
                    confidences = []
                    for i, p in enumerate(preds):
                        idx = list(classes).index(p)
                        confidences.append(probs[i][idx])
                    df_upload["Confidence"] = confidences

                # Show results and allow download
                st.success("Prediction complete — download results below")
                st.dataframe(df_upload.head(10))

                # convert to CSV
                csv_bytes = df_upload.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download predictions (CSV)", data=csv_bytes, file_name="predictions.csv", mime="text/csv")


# -------------------------
# Footer / Tips
# -------------------------

st.markdown("---")
st.markdown("### Tips for use")
st.markdown("- For best results, paste the full review text (not just a few words).\n- If you trained a different model (e.g. transformer), place `sentiment_model.joblib` in this folder or update the filename in the code.")
st.markdown("\nMade for quick demos and teaching — accessible UI with large text and clear icons.")
