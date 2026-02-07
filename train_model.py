import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ==============================
# 1️⃣ CHECK CURRENT DIRECTORY
# ==============================
print("Current Working Directory:", os.getcwd())

# ==============================
# 2️⃣ LOAD DATASET (UPDATED FULL PATH)
# ==============================
file_path = r"C:\Users\JIJAU\OneDrive\Desktop\codsoft\archive\Genre Classification Dataset\train_data.txt"

if not os.path.exists(file_path):
    print("Dataset file not found. Check the path.")
    exit()

df = pd.read_csv(
    file_path,
    sep=":::",
    engine="python",
    names=["id", "title", "genre", "plot"],
    encoding="latin-1"
)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

# ==============================
# 3️⃣ CLEAN DATA
# ==============================
df = df.dropna()

if len(df) == 0:
    print("Dataset is empty after cleaning.")
    exit()

X = df["plot"]
y = df["genre"]

# ==============================
# 4️⃣ TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==============================
# 5️⃣ ML PIPELINE
# ==============================
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", MultinomialNB())
])

# ==============================
# 6️⃣ TRAIN MODEL
# ==============================
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# ==============================
# 7️⃣ SAVE MODEL
# ==============================
joblib.dump(model, "model.pkl")
print("Model saved successfully as model.pkl")
