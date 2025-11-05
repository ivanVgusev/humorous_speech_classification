import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from collections import Counter

df = pd.read_csv("features.csv")
X = df.drop(columns=["label", "file", "start", "end"])
y = df["label"]

count_class_0 = sum(y == 0)
count_class_1 = sum(y == 1)

# Случайно выбираем столько же 0, сколько есть 1
df_class_0 = df[df["label"] == 0].sample(count_class_1, random_state=42)
df_class_1 = df[df["label"] == 1]

# Объединяем и перемешиваем
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

model = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    objective="binary:logistic",
    eval_metric="aucpr",
    n_estimators=300,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=1.0,
    tree_method="hist"
)

print("\nОбучаем модель...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n=== Отчёт по классификации ===")
print(classification_report(y_test, y_pred, digits=3))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели: {accuracy:.3f}")
