import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from collections import Counter

# === 1. Загрузка данных ===
df = pd.read_csv("features.csv")  # путь к CSV
X = df.drop(columns=["label", "file", "start", "end"])
y = df["label"]

# === 2. Разделение на train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 3. Масштабирование ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4. Балансировка классов ===
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print("Баланс классов после SMOTE:", Counter(y_train_res))

# === 5. Подбор гиперпараметров с GPU ===
params = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 8],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1, 2]
}

# Создаем базовый классификатор с поддержкой GPU
base_classifier = XGBClassifier(
    random_state=42,
    n_jobs=-1,  # Используем все CPU ядра для предобработки
    tree_method='gpu_hist',  # Используем GPU для построения деревьев
    predictor='gpu_predictor',  # Используем GPU для предсказаний
    gpu_id=0,  # ID GPU устройства
    objective="binary:logistic",
    eval_metric="logloss"
)

grid = GridSearchCV(
    estimator=base_classifier,
    param_grid=params,
    cv=3,
    scoring="f1_macro",
    n_jobs=1,  # Важно: устанавливаем 1 при использовании GPU
    verbose=1
)

print("Начинаем подбор гиперпараметров с использованием GPU...")
grid.fit(X_train_res, y_train_res)

# === 6. Результаты ===
print("\nЛучшие параметры:", grid.best_params_)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test_scaled)

print("\n=== Отчёт по классификации ===")
print(classification_report(y_test, y_pred, digits=3))

accuracy = (y_test == y_pred).mean()
print(f"\nТочность модели: {accuracy:.3f}")

# Дополнительная информация о использовании GPU
print(f"\n=== Информация о GPU ===")
print(f"Используемый метод деревьев: {best_model.get_params()['tree_method']}")
print(f"Используемый метод предсказания: {best_model.get_params()['predictor']}")