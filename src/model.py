import tensorflow as tf

def create_model(input_shape=(224, 224, 3), num_classes=15):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model


    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),  # Global pooling after the base model
        tf.keras.layers.Dense(1024, activation='relu'),  # Add a dense layer for more abstraction
        tf.keras.layers.Dropout(0.5),  # Dropout to reduce overfitting
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Final layer with `num_classes` units
    ])

    return model
