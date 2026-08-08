import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. Load the dataset
data = pd.read_csv("dataset/emotions.csv")

print("Dataset loaded successfully!")
print(data.head())


# 2. Separate text and emotion
X = data["text"]
y = data["emotion"]


# 3. Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)


# 4. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Create the machine learning model
model = LogisticRegression(max_iter=1000)


# 6. Train the model
model.fit(X_train, y_train)

print("Model training completed!")


# 7. Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


# 8. Save the trained model
joblib.dump(model, "model/emotion_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("Model saved successfully!")