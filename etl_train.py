import os
import sqlite3
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("--- URUCHAMIANIE PIPELINE ETL ---")
if not os.path.exists("train.csv"):
    print("BŁĄD: Brak pliku 'train.csv' w bieżącym folderze!")
    exit()

# 1. Wczytanie i transformacja danych
df = pd.read_csv("train.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna("S")
df["Sex_encoded"] = df["Sex"].map({"male": 0, "female": 1})
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 2. Ładowanie danych do lokalnej bazy danych SQLite
conn = sqlite3.connect("titanic.db")
df.to_sql("passengers", conn, if_exists="replace", index=False)
conn.close()
print("Sukces: Dane oczyszczone i zapisane w bazie 'titanic.db'\n")

print("--- TRENOWANIE I OPTYMALIZACJA MODELU ML ---")
features = ["Pclass", "Sex_encoded", "Age", "Fare", "FamilySize"]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Przeszukiwanie siatki hiperparametrów
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Ocena modelu
y_pred_best = best_model.predict(X_test)
print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Dokładność na danych testowych: {accuracy_score(y_test, y_pred_best):.2%}")

# 3. Eksport modelu do pliku
joblib.dump(best_model, 'titanic_model.pkl')
print("Sukces: Model został zapisany do pliku 'titanic_model.pkl'!")