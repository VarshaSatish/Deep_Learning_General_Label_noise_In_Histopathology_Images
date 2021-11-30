import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator()

def make_generator():
    directory_train = './BACH_dataset/train'
    train_generator = train_datagen.flow_from_directory(
        directory=directory_train,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=8,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    return train_generator

val_datagen = ImageDataGenerator()

directory_val = './BACH_dataset/validation'
val_generator= val_datagen.flow_from_directory(
    directory=directory_val,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=8,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

train_dataset = tf.data.Dataset.from_generator(make_generator,(tf.float32, tf.float32))

print(train_dataset.load_data())



