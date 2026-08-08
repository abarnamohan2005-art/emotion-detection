# AI-Based Text Emotion Detection System

A simple machine learning web application that detects emotions from text.

## Technologies

- Python
- Flask
- HTML
- CSS
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression

## Emotions

- Happy
- Sad
- Angry
- Anxiety
- Depression
- Neutral

## How it works

User enters a sentence through the web interface.

The Flask backend receives the sentence, converts it into TF-IDF features, and sends it to the trained Logistic Regression model.

The predicted emotion is then displayed on the webpage.