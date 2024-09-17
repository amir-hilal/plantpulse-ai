import sys
from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py [train|evaluate]")
        sys.exit(1)

    if sys.argv[1] == "train":
        train_model()
    elif sys.argv[1] == "evaluate":
        evaluate_model()
    else:
        print("Invalid argument. Use 'train' or 'evaluate'.")
