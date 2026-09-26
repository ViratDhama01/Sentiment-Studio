import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_balanced_model():
    print("Loading dataset...")
    # Load the dataset (assumes Reviews.csv is in the same folder)
    try:
        df = pd.read_csv("Reviews.csv")
    except FileNotFoundError:
        print("Error: Reviews.csv not found. Please ensure it's in the project folder.")
        return

    # 1. Labeling (from 1-5 stars to sentiment)
    def score_to_sentiment(score):
        if score >= 4: return "positive"
        elif score <= 2: return "negative"
        else: return "neutral"

    df["Sentiment"] = df["Score"].apply(score_to_sentiment)

    print("\nClass Distribution:")
    print(df["Sentiment"].value_counts())

    # 2. Cleaning
    print("\nCleaning text...")
    df["Text_clean"] = df["Text"].apply(clean_text)

    # 3. Splitting
    # We keep all classes this time (no removing 'neutral')
    train, temp = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Sentiment"],
        random_state=42
    )
    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp["Sentiment"],
        random_state=42
    )

    # 4. Pipeline with Balanced Class Weights
    # class_weight='balanced' tells LogisticRegression to give more importance to minority classes
    print("\nTraining Balanced Model...")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=-1, class_weight='balanced'))
    ])

    pipe.fit(train["Text_clean"], train["Sentiment"])

    # 5. Evaluation
    test_preds = pipe.predict(test["Text_clean"])
    print("\n--- Balanced Model Test Results ---")
    print(classification_report(test["Sentiment"], test_preds))

    # 6. Save the model
    joblib.dump(pipe, "sentiment_model_balanced.joblib")
    print("\n✅ Balanced model saved as 'sentiment_model_balanced.joblib'")

if __name__ == "__main__":
    train_balanced_model()
