import joblib
import numpy as np

# Load trained model
model = joblib.load("../models/random_forest_model.pkl")

# Example input (Iris features)
# Format: [sepal_length, sepal_width, petal_length, petal_width]
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = model.predict(sample)

print("Predicted Class:", prediction[0]) 
