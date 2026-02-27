import logging
from data_loader import load_data
from model import get_model
from train import train_and_evaluate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    logging.info("Loading dataset...")
    X_train, X_test, y_train, y_test = load_data()

    logging.info("Initializing model...")
    model = get_model()

    logging.info("Training model...")
    accuracy = train_and_evaluate(model, X_train, X_test, y_train, y_test)

    logging.info(f"Model Accuracy: {accuracy}")


if __name__ == "__main__":
    main()