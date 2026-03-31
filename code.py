import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. CREATE REALISTIC DATASET
# -----------------------------
data = []

for _ in range(60):
    study_hours = random.randint(1, 8)
    screen_time = random.randint(1, 8)
    breaks = random.randint(1, 8)
    sleep_hours = random.randint(4, 8)
    deadline_days = random.randint(1, 7)

    # Simple logic to generate label
    if screen_time > 5 or breaks > 5 or study_hours < 3:
        procrastination = 1
    else:
        procrastination = 0

    data.append([study_hours, screen_time, breaks, sleep_hours, deadline_days, procrastination])

df = pd.DataFrame(data, columns=[
    'study_hours', 'screen_time', 'breaks', 'sleep_hours', 'deadline_days', 'procrastination'
])

print("Sample Data:\n", df.head())

# -----------------------------
# 2. SPLIT DATA
# -----------------------------
X = df.drop('procrastination', axis=1)
y = df['procrastination']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 3. SCALE DATA
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 4. TRAIN MODEL
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. TEST MODEL
# -----------------------------
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# 6. USER INPUT PREDICTION
# -----------------------------
print("\n--- Enter Your Details ---")

study_hours = float(input("Study hours per day: "))
screen_time = float(input("Screen time (hours): "))
breaks = float(input("Number of breaks: "))
sleep_hours = float(input("Sleep hours: "))
deadline_days = float(input("Days left for deadline: "))

user_data = pd.DataFrame([[study_hours, screen_time, breaks, sleep_hours, deadline_days]],
                         columns=X.columns)

user_data = scaler.transform(user_data)

prediction = model.predict(user_data)

print("\nPrediction Result:")
if prediction[0] == 1:
    print("⚠️ You are likely to PROCRASTINATE")
else:
    print("✅ You are NOT likely to procrastinate")