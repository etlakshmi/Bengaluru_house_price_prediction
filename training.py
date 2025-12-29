import pandas as pd
import numpy as np
import pickle
import json
import os

from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso


# -----------------------------
# Load and clean dataset
# -----------------------------
def load_data(data):
    df = pd.read_csv(data)

    df.drop(['area_type', 'society', 'balcony', 'availability'], axis=1, inplace=True)
    df.dropna(inplace=True)

    # Extract BHK
    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
    df.drop('size', axis=1, inplace=True)

    # Convert total_sqft
    def convert_sqft_to_num(x):
        try:
            if '-' in x:
                a, b = x.split('-')
                return (float(a) + float(b)) / 2
            return float(x)
        except:
            return None

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    df.dropna(inplace=True)

    return df


# -----------------------------
# Remove outliers
# -----------------------------
def remove_location_outliers(df):
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

    df_out = pd.DataFrame()
    for _, subdf in df.groupby('location'):
        mean = subdf.price_per_sqft.mean()
        std = subdf.price_per_sqft.std()
        reduced_df = subdf[
            (subdf.price_per_sqft > (mean - std)) &
            (subdf.price_per_sqft < (mean + std))
        ]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)

    return df_out


# -----------------------------
# Feature engineering
# -----------------------------
def prepare_features(df):
    df = remove_location_outliers(df)

    # Handle rare locations
    location_stats = df.groupby('location')['location'].agg('count')
    location_stats_less_than_10 = location_stats[location_stats <= 10]

    df.location = df.location.apply(
        lambda x: 'other' if x in location_stats_less_than_10 else x
    )

    # One-hot encoding
    dummies = pd.get_dummies(df['location'], drop_first=True)
    df = pd.concat([df.drop('location', axis=1), dummies], axis=1)

    # Target & features
    X = df.drop('price', axis=1)
    y = np.log1p(df['price'])  # Log transform target

    return X, y


# -----------------------------
# Train and save model
# -----------------------------
def train_and_save_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "data", "Bengaluru_House_Data.csv")

    df = load_data(csv_path)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001)
    }

    best_model = None
    best_score = 0

    print("\nModel Accuracy Comparison:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name}: {score * 100:.2f}%")

        if score > best_score:
            best_score = score
            best_model = model

    # Cross-validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=cv)

    print(f"\nBest Model: {type(best_model).__name__}")
    print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")

    # Save model & columns
    os.makedirs("model", exist_ok=True)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("model/columns.json", "w") as f:
        json.dump(list(X.columns), f)

    print("\n✅ Model and columns saved successfully!")


# -----------------------------
# Run training
# -----------------------------
if __name__ == "__main__":
    train_and_save_model()
