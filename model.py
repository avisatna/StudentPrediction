from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    # Hardcoded training data: [attendance, previous_marks, assignments, study_hours]
    X = [
        [85, 78, 80, 3],
        [60, 55, 50, 1],
        [75, 70, 65, 2],
        [40, 45, 40, 0],
        [90, 88, 85, 4],
        [50, 60, 45, 1],
        [95, 92, 90, 5],
        [55, 58, 60, 1],
        [30, 35, 25, 0],
        [70, 68, 70, 3]
    ]
    y = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]  # 1 = Pass, 0 = Fail

    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save model
    joblib.dump(model, 'result_predictor.pkl')
