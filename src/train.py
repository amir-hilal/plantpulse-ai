from model import create_model
from data_loader import load_data
import tensorflow as tf

def train_model():
    train_gen, val_gen, test_gen = load_data('data', batch_size=32)

    # Explicitly set `num_classes=15` for your 15 classes
    model = create_model(num_classes=15)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint('models/best_model.keras', monitor='val_loss', save_best_only=True, mode='min')

    # Train the model
    model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=[checkpoint])

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    train_model()
