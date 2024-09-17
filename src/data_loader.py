import tensorflow as tf

def load_data(data_dir, batch_size=32, image_size=(224, 224)):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + '/train',
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'  # Make sure labels are categorical
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + '/val',
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + '/test',
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    return train_ds, val_ds, test_ds
