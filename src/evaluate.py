import tensorflow as tf
from data_loader import load_data

def evaluate_model():
    model = tf.keras.models.load_model('models/best_model.keras')

    _, _, test_gen = load_data('data')

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()
