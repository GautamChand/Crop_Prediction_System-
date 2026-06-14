import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    PolynomialFeatures
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    layout="wide"
)

st.title("🌾 Crop Recommendation System")
st.write("Crop Prediction using Multiple Machine Learning Algorithms")

# ---------------------------
# LOAD DATASET
# ---------------------------
df = pd.read_csv("Crop_recommendationV2.csv")

st.subheader("Dataset Information")

st.write("Shape:", df.shape)

st.write("Missing Values")
st.dataframe(df.isnull().sum())

# ---------------------------
# TARGET
# ---------------------------
target = "label"

X = df.drop(columns=[target])
y = df[target]

# ---------------------------
# ENCODE TARGET
# ---------------------------
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# ---------------------------
# COLUMN TYPES
# ---------------------------
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# ---------------------------
# PREPROCESSOR
# ---------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder",
                 Pipeline([
                     ("label", SimpleImputer(strategy="most_frequent"))
                 ]))
            ]),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Manual encoding of categorical columns
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# ---------------------------
# FILL MISSING NUMERIC VALUES
# ---------------------------
num_imputer = SimpleImputer(strategy="mean")
X[numeric_features] = num_imputer.fit_transform(
    X[numeric_features]
)

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# ---------------------------
# MODELS
# ---------------------------
models = {

    "Linear Regression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

    "Logistic Regression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000))
        ]),

    "Polynomial Regression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(
                degree=2,
                include_bias=False
            )),
            ("model", LinearRegression())
        ]),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
}

# ---------------------------
# TRAIN MODELS
# ---------------------------
results = {}

st.header("Model Performance")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if name in [
        "Linear Regression",
        "Polynomial Regression"
    ]:
        predictions = np.clip(
            np.round(predictions),
            0,
            len(target_encoder.classes_) - 1
        ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "predictions": predictions
    }

    st.subheader(name)

    st.write(
        f"Accuracy: **{accuracy:.4f}**"
    )

    sample_preds = target_encoder.inverse_transform(
        predictions[:10]
    )

    st.write(
        "Sample Predictions:",
        sample_preds
    )

# ---------------------------
# BEST MODEL
# ---------------------------
best_model_name = max(
    results,
    key=lambda x: results[x]["accuracy"]
)

best_model = results[best_model_name]["model"]

st.success(
    f"Best Model: {best_model_name} "
    f"(Accuracy = {results[best_model_name]['accuracy']:.4f})"
)

# ---------------------------
# CLASSIFICATION REPORT
# ---------------------------
best_predictions = results[
    best_model_name
]["predictions"]

st.subheader("Classification Report")

report = classification_report(
    y_test,
    best_predictions,
    target_names=target_encoder.classes_,
    zero_division=0
)

st.text(report)

# ---------------------------
# USER INPUT
# ---------------------------
st.header("Predict Crop")

user_input = {}

for column in X.columns:

    if column in numeric_features:

        user_input[column] = st.number_input(
            column,
            value=float(X[column].mean())
        )

    else:

        unique_values = sorted(
            df[column].astype(str).unique()
        )

        selected = st.selectbox(
            column,
            unique_values
        )

        encoder = LabelEncoder()
        encoder.fit(
            df[column].astype(str)
        )

        user_input[column] = encoder.transform(
            [selected]
        )[0]

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Predict Crop"):

    sample = pd.DataFrame(
        [user_input]
    )

    st.subheader("Predictions")

    for name, model_info in results.items():

        model = model_info["model"]

        pred = model.predict(sample)[0]

        if name in [
            "Linear Regression",
            "Polynomial Regression"
        ]:
            pred = int(
                np.clip(
                    round(float(pred)),
                    0,
                    len(target_encoder.classes_) - 1
                )
            )

        crop = target_encoder.inverse_transform(
            [int(pred)]
        )[0]

        st.write(
            f"**{name}:** {crop}"
        )