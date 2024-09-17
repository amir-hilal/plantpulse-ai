import tensorflow as tf
import numpy as np

def predict_image(model_path, img_path, target_size=(224, 224)):

    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image using the updated method
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_idx, confidence

if __name__ == "__main__":

    img_path = 'path_to_image_to_predict'  # Update with actual image path
    class_idx, confidence = predict_image('models/best_model.keras', img_path)
    print(f"Predicted class: {class_idx}, Confidence: {confidence}")
