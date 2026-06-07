import pandas as pd
import pickle
import os

from sklearn.linear_model import LinearRegression

df = pd.read_csv("weather.csv")

df["Day"] = range(len(df))

X = df[["Day"]]
y = df["Temperature"]

model = LinearRegression()
model.fit(X, y)

os.makedirs("models", exist_ok=True)

with open("models/weather_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")
print("Saved as models/weather_model.pkl")
