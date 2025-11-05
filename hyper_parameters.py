import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from collections import Counter

df = pd.read_csv("features.csv")
X = df.drop(columns=["label", "file", "start", "end"])
y = df["label"]

count_class_0 = sum(y == 0)
count_class_1 = sum(y == 1)

df_class_0 = df[df["label"] == 0].sample(count_class_1, random_state=42)
df_class_1 = df[df["label"] == 1]
df_balanced = pd.concat([df_class_0, df_class_1]).sample(frac=1, random_state=42)

X = df_balanced.drop(columns=["label", "file", "start", "end"])
y = df_balanced["label"]

print("Баланс после undersampling:", Counter(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

params = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0, 0.1, 0.3],
    "min_child_weight": [1, 3, 5],
    # L2-регуляризация
    "reg_lambda": [0.5, 1, 2],
    # L1-регуляризация
    "reg_alpha": [0, 0.1, 0.5]
}

grid = GridSearchCV(
    XGBClassifier(
        random_state=42,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist"
    ),
    param_grid=params,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

print("\nПодбираем гиперпараметры...")
grid.fit(X_train, y_train)

print("\nЛучшие параметры:", grid.best_params_)

best_model = grid.best_estimator_

print("\nОбучаем модель с лучшими параметрами...")
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print("\n=== Отчёт по классификации ===")
print(classification_report(y_test, y_pred, digits=3))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели: {accuracy:.3f}")
