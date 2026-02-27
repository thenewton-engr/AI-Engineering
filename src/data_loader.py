from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE


def load_data():
    data = load_iris()
    X = data.data
    y = data.target

    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )