import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("Harisathi-master/Data-processed/crop_recommendation.csv")

# Features and target
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Save in models folder
with open("Harisathi-master/app/models/RandomForest.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Crop Recommendation Model retrained and saved!")
