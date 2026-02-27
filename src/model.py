from sklearn.ensemble import RandomForestClassifier
from config import MODEL_PARAMS


def get_model():
    return RandomForestClassifier(**MODEL_PARAMS)