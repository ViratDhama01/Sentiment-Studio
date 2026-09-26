# streamlit_sentiment_app.py
# A professional, high-end hybrid sentiment analysis app.
# Combines Fast Logistic Regression with a powerful Multi-lingual DistilBERT Transformer.

import streamlit as st
import pandas as pd
import re
import joblib
import io
import numpy as np
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
import os

# -------------------------
# Configuration & Styling
# -------------------------

st.set_page_config(
    page_title="Sentiment Studio",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def apply_custom_style():
    """Injects custom CSS to remove the 'basic' Streamlit look and add a premium 'Studio' feel."""
    st.markdown("""
        <style>
        .stApp {
            background-color: #F8FAFC;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        .main-header {
            font-size: 3rem !important;
            font-weight: 800 !important;
            color: #1E293B;
            text-align: center;
            margin-bottom: 0rem !important;
            letter-spacing: -1px;
        }
        .sub-header {
            font-size: 1.2rem !important;
            color: #64748B;
            text-align: center;
            margin-bottom: 3rem !important;
        }
        div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
            background-color: #FFFFFF;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 1.5rem;
            border: 1px solid #E2E8F0;
        }
        .stButton > button {
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            border: none !important;
            color: white !important;
            background-color: #4F46E5 !important;
        }
        .stButton > button:hover {
            background-color: #4338CA !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }
        .stTextArea textarea {
            border-radius: 0.75rem !important;
            border: 1px solid #CBD5E1 !important;
            padding: 1rem !important;
        }
        .source-badge {
            background-color: #F1F5F9;
            color: #475569;
            padding: 0.2rem 0.6rem;
            border-radius: 0.4rem;
            font-family: monospace;
            font-size: 0.85rem;
            border: 1px solid #E2E8F0;
        }
        header {visibility: hidden;}
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# -------------------------
# Model Loading & Setup
# -------------------------

@st.cache_resource
def load_models():
    """Load both models into memory once and cache them."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "sentiment_model.joblib")
        lr_model = joblib.load(model_path)
    except Exception:
        lr_model = None

    try:
        bert_pipeline = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    except Exception:
        bert_pipeline = None

    return lr_model, bert_pipeline

lr_model, bert_model = load_models()

# -------------------------
# Helper functions
# -------------------------

def find_text_column(df):
    keywords = ["text", "review", "comment", "body", "message", "content", "feedback", "desc"]
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    max_len = 0
    best_col = None
    for col in df.columns:
        avg_len = df[col].astype(str).str.len().mean()
        if avg_len > max_len:
            max_len = avg_len
            best_col = col
    return best_col

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_hybrid(text):
    clean_t = clean_text(text)
    lr_result = None
    lr_conf = 0.0

    if lr_model:
        try:
            probs = lr_model.predict_proba([clean_t])[0]
            label = lr_model.predict([clean_t])[0]
            classes = lr_model.classes_
            pred_idx = list(classes).index(label)
            lr_conf = probs[pred_idx]
            lr_result = label
        except Exception:
            lr_result = None

    if lr_result and lr_conf > 0.85:
        return lr_result, lr_conf, "⚡ Fast Match (LR)"

    if bert_model:
        try:
            bert_out = bert_model(text)[0]
            label_raw = bert_out['label'].lower()
            bert_conf = bert_out['score']

            if "negative" in label_raw:
                bert_label = "negative"
            elif "neutral" in label_raw:
                bert_label = "neutral"
            elif "positive" in label_raw:
                bert_label = "positive"
            else:
                bert_label = "neutral"

            return bert_label, bert_conf, "🌐 Global AI (DistilBERT)"
        except Exception:
            pass

    return lr_result if lr_result else "neutral", 0.0, "Unknown"

def render_visuals(df):
    st.markdown("### 📊 Sentiment Analysis Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        color_map = {"positive": "#10B981", "neutral": "#F59E0B", "negative": "#F43F5E"}
        fig_pie = px.pie(
            sentiment_counts,
            values="Count",
            names="Sentiment",
            color="Sentiment",
            color_discrete_map=color_map,
            hole=0.5,
            title="Sentiment Distribution"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            color_discrete_map=color_map,
            title="Total Review Volume"
        )
        fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### AI Confidence Analysis")
    fig_hist = px.histogram(
        df,
        x="Confidence",
        nbins=20,
        title="Model Confidence Distribution",
        labels={"Confidence": "Score"},
        color_discrete_sequence=["#6366F1"]
    )
    fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------
# Main App Interface
# -------------------------

apply_custom_style()

st.markdown('<p class="main-header">🌟 Sentiment Studio Hybrid</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">High-Precision English & Hinglish Intelligence</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["✨ Single Analysis", "📦 Batch Processor"])

with tab1:
    st.markdown("### Analyze a Single Review")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        user_review = st.text_area("", placeholder="Paste a customer review here...", height=150, key="txt_input")
        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            if st.button("🗑️ Clear", key="clear_btn"):
                st.session_state.txt_input = ""
                st.rerun()
        with btn_col2:
            predict_clicked = st.button("🔍 Predict Sentiment", key="predict_btn", use_container_width=True)

        if predict_clicked:
            if not user_review.strip():
                st.warning("Please enter some text to analyze.")
            else:
                label, conf, source = predict_hybrid(user_review)
                st.markdown("---")
                st.markdown(f"**Source:** `<span class='source-badge'>{source}</span>`", unsafe_allow_html=True)

                if label == "positive":
                    st.markdown("<h1 style='color:#10B981;font-size:64px;margin:0'>👍 Positive</h1>", unsafe_allow_html=True)
                    gauge_color = "#10B981"
                elif label == "negative":
                    st.markdown("<h1 style='color:#F43F5E;font-size:64px;margin:0'>👎 Negative</h1>", unsafe_allow_html=True)
                    gauge_color = "#F43F5E"
                else:
                    st.markdown("<h1 style='color:#F59E0B;font-size:64px;margin:0'>😐 Neutral</h1>", unsafe_allow_html=True)
                    gauge_color = "#F59E0B"

                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = conf * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Score %", 'font': {'size': 14}},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 50], 'color': "#FEE2E2"},
                            {'range': [50, 80], 'color': "#F8FAFC"},
                            {'range': [80, 100], 'color': "#DCFCE7"}
                        ],
                        'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': conf * 100}
                    }
                ))
                fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.write(f"Final Confidence: {conf:.2%}")

with tab2:
    st.markdown("### Process Bulk Reviews")
    with st.expander("📖 Data Preparation Guide"):
        st.markdown("To ensure a perfect run, your CSV should ideally have a column named **`Text`**. If it doesn't, our **Universal Detector** will try to find the review column automatically.")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df_upload = None
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                df_upload = pd.read_csv(uploaded_file, encoding=encoding)
                break
            except: continue

        if df_upload is not None:
            detected_col = find_text_column(df_upload)
            col_a, col_b = st.columns([1, 2])
            with col_a:
                selected_col = st.selectbox(
                    "Verify Text Column:",
                    options=df_upload.columns.tolist(),
                    index=df_upload.columns.tolist().index(detected_col) if detected_col in df_upload.columns else 0
                )
            with col_b:
                st.dataframe(df_upload.head(5), use_container_width=True)

            if st.button("🚀 Run Batch Analysis", key="predict_csv", use_container_width=True):
                with st.spinner("Analyzing reviews..."):
                    texts = df_upload[selected_col].fillna("").astype(str).tolist()
                    results, confs = [], []
                    for t in texts:
                        l, c, s = predict_hybrid(t)
                        results.append(l)
                        confs.append(c)

                    df_upload["Sentiment"] = results
                    df_upload["Confidence"] = confs

                    st.success("Analysis Complete!")
                    st.dataframe(df_upload.head(10), use_container_width=True)
                    render_visuals(df_upload)
                    csv_bytes = df_upload.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Export Results", data=csv_bytes, file_name="sentiment_results.csv", mime="text/csv")

st.markdown("---")
st.markdown('<div style="text-align: center; color: #94A3B8; font-size: 0.9rem;">Powered by Logistic Regression & Multi-lingual DistilBERT Transformer</div>', unsafe_allow_html=True)
