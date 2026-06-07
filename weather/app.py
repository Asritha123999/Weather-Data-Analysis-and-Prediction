from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(
    open("models/weather_model.pkl", "rb")
)

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        day = int(request.form["day"])

        result = model.predict([[day]])

        prediction = round(result[0], 2)

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
