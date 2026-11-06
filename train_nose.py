import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sensor import simulate_sensor_reading

print("Training the AI Nose...")

# Load our synthetic chemical data
with open('samples.json', 'r') as f:
    samples = json.load(f)

X = []
y = []

# Sniff each smell 200 times with random noise to build a training dataset
for label, molecules in samples.items():
    for _ in range(200):
        # Generate a simulated 8-channel sensor reading
        reading = simulate_sensor_reading(molecules, noise_level=0.05)
        X.append(reading)
        y.append(label)

# Train a Random Forest classifier to recognize the sensor patterns
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save the trained model
with open('nose_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Training complete! Model saved as 'nose_model.pkl'.")