"""
train_and_save.py
─────────────────
Run this ONCE on a machine that has the dataset to train the model
and save it as fruit_model.h5.

Usage:
    python train_and_save.py --data_dir dataset/train --epochs 10
"""

import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def build_model(num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/train", help="Path to training directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", default="fruit_model.h5")
    args = parser.parse_args()

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_data = datagen.flow_from_directory(
        args.data_dir,
        target_size=(100, 100),
        batch_size=32,
        class_mode="categorical",
        subset="training",
    )

    val_data = datagen.flow_from_directory(
        args.data_dir,
        target_size=(100, 100),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
    )

    num_classes = len(train_data.class_indices)
    print("Classes detected:", train_data.class_indices)

    model = build_model(num_classes)
    model.fit(train_data, epochs=args.epochs, validation_data=val_data)

    model.save(args.output)
    print(f"\n✅ Model saved to {args.output}")
    print("Copy this file to your server next to bot.py")


if __name__ == "__main__":
    main()
