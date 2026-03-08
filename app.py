from flask import Flask, render_template, request, redirect
import pickle
import re
import string

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

history = []
real_count = 0
fake_count = 0

# ---------------- TEXT CLEANING ----------------

def clean_text(text):
    text = text.lower()

    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("home.html")


# ---------------- CHECK NEWS PAGE ----------------

@app.route("/check", methods=["GET","POST"])
def check():

    global real_count, fake_count

    result = ""
    confidence = 0

    if request.method == "POST":

        news = request.form["news"]

        cleaned = clean_text(news)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)

        prob = model.predict_proba(vector)

        confidence = round(max(prob[0]) * 100,2)

        if prediction[0] == 1:
            result = "Real News"
            real_count += 1
        else:
            result = "Fake News"
            fake_count += 1

        history.append((news,result,confidence))

    return render_template(
        "check_news.html",
        result=result,
        confidence=confidence
    )


# ---------------- HISTORY PAGE ----------------

@app.route("/history")
def history_page():

    return render_template(
        "history.html",
        history=history
    )


# ---------------- ACCURACY DASHBOARD ----------------

@app.route("/accuracy")
def accuracy():

    total = real_count + fake_count

    accuracy_value = 0

    if total != 0:
        accuracy_value = round((real_count/total)*100,2)

    return render_template(
        "accuracy.html",
        accuracy=accuracy_value,
        real_count=real_count,
        fake_count=fake_count
    )


# ---------------- CLEAR HISTORY ----------------

@app.route("/clear", methods=["POST"])
def clear():

    global history, real_count, fake_count

    history = []
    real_count = 0
    fake_count = 0

    return redirect("/history")


# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run(debug=True)