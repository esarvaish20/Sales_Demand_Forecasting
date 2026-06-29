import os
import joblib
import pandas as pd
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    flash
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

app = Flask(__name__)
app.secret_key = "sales_forecasting_secret"

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_FOLDER, "sales_forecasting_model.pkl")
PREDICTION_PATH = os.path.join(OUTPUT_FOLDER, "sales_predictions.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():

    if "dataset" not in request.files:
        flash("No file uploaded.")
        return redirect(url_for("home"))

    file = request.files["dataset"]

    if file.filename == "":
        flash("Please choose a CSV file.")
        return redirect(url_for("home"))

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:

        data = pd.read_csv(file_path)

        
        # Data Cleaning
       

        data.drop_duplicates(inplace=True)

        for col in data.columns:

            if data[col].dtype == "object":
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)

        if "Date" in data.columns:

            data["Date"] = pd.to_datetime(data["Date"])

            data["Year"] = data["Date"].dt.year
            data["Month"] = data["Date"].dt.month
            data["Day"] = data["Date"].dt.day
            data["Weekday"] = data["Date"].dt.day_name()

        if "Sales" not in data.columns:
            flash("Dataset must contain a 'Sales' column.")
            return redirect(url_for("home"))

        data["Sales Growth"] = data["Sales"].pct_change()
        data["Sales Growth"].fillna(0, inplace=True)

        # Encode categorical columns
        

        categorical = data.select_dtypes(include="object").columns

        for col in categorical:
            data[col] = data[col].astype("category").cat.codes

      
        # Features & Target
        

        drop_columns = ["Sales"]

        if "Date" in data.columns:
            drop_columns.append("Date")

        X = data.drop(columns=drop_columns)

        y = data["Sales"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        
        # Train Model
        

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        
        # Metrics
       

        mae = mean_absolute_error(y_test, predictions)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        r2 = r2_score(y_test, predictions)

       
        # Save Model
        

        joblib.dump(model, MODEL_PATH)

      
        # Save Predictions
        

        comparison = pd.DataFrame({

            "Actual": y_test.values,
            "Predicted": predictions

        })

        comparison.to_csv(PREDICTION_PATH, index=False)

        return render_template(
            "result.html",
            mae=round(mae, 2),
            rmse=round(rmse, 2),
            r2=round(r2, 4),
            tables=[comparison.head(20).to_html(
                classes="table table-bordered",
                index=False
            )]
        )

    except Exception as e:
        flash(str(e))
        return redirect(url_for("home"))


@app.route("/download")
def download():

    return send_file(
        PREDICTION_PATH,
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0", port=5000,debug=True
    )