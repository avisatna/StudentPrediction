# student_predictor.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


print("hello")
class StudentPredictor:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()
        
        # Initialize with some basic rules for prediction
        # We'll use this until we have real training data
        self.initialize_basic_model()
        
    def initialize_basic_model(self):
        # Create a simple rule-based model
        # Generate a small dataset with clear rules
        X = np.array([
            # attendance, prev_marks, seminar, assignment, study_hrs
            [95, 85, 9, 18, 6],  # Good student - Pass
            [75, 65, 7, 15, 4],  # Average student - Pass
            [60, 45, 5, 10, 2],  # Poor student - Fail
            [40, 35, 4, 8, 1],   # Poor student - Fail
        ])
        
        # Define results (1 = Pass, 0 = Fail)
        y = np.array([1, 1, 0, 0])
        
        # Fit the scaler and model
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
    
    def predict_result(self, attendance, prev_marks, seminar, assignment, study_hrs):
        # Prepare input data
        features = np.array([[attendance, prev_marks, seminar, assignment, study_hrs]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Get prediction probability
        prob = self.model.predict_proba(features_scaled)[0]
        
        return prediction, max(prob) * 100

