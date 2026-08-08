from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained emotion model
model = joblib.load("model/emotion_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Emotion prediction
@app.route("/predict", methods=["POST"])
def predict():

    # Get the sentence entered by the user
    text = request.form["text"]

    # Convert the sentence into TF-IDF numbers
    text_vector = vectorizer.transform([text])

    # Predict emotion
    prediction = model.predict(text_vector)[0]

    # Emoji for each emotion
    emojis = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😡",
        "anxiety": "😰",
        "depression": "😔",
        "neutral": "😐"
    }

    emoji = emojis.get(prediction, "🙂")

    # Send result back to webpage
    return render_template(
        "index.html",
        emotion=prediction,
        emoji=emoji,
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)