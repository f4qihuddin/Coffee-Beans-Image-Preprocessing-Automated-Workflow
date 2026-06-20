import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

def preprocess_image(TRAIN_DIR, VAL_DIR):
    checkpoint_dir_train = os.path.join(TRAIN_DIR, '.ipynb_checkpoints')
    checkpoint_dir_val = os.path.join(VAL_DIR, '.ipynb_checkpoints')

    if os.path.exists(checkpoint_dir_train):
        shutil.rmtree(checkpoint_dir_train)
        print(f"Removed: {checkpoint_dir_train}")

    if os.path.exists(checkpoint_dir_val):
        shutil.rmtree(checkpoint_dir_val)
        print(f"Removed: {checkpoint_dir_val}")

    # Memastikan direktori ada sebelum digunakan oleh generator
    # Catatan: Kode augmentasi sebelumnya harus sudah memindahkan file ke folder kategori di sini
    if not os.path.exists(TRAIN_DIR):
        print(f"Peringatan: Folder {TRAIN_DIR} tidak ditemukan. Pastikan proses augmentasi di sel sebelumnya sudah selesai.")

    # ImageDataGenerator dengan augmentasi sederhana
    datagen = ImageDataGenerator(
        rescale=1/255.,
        validation_split=0.3
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_generator = datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(150, 150),
            batch_size=128,
            color_mode='grayscale',
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        validation_generator = datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(150, 150),
            batch_size=128,
            color_mode='grayscale',
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        test_generator = test_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(150, 150),
            batch_size=128,
            color_mode='grayscale', # Disamakan dengan training (grayscale)
            class_mode='categorical',
            shuffle=False
        )

        # Define output directories for the preprocessed images
        output_base_dir = 'coffeebeans_preprocessing'
        output_train_dir = os.path.join(output_base_dir, train_generator)
        output_validation_dir = os.path.join(output_base_dir, validation_generator)
        output_test_dir = os.path.join(output_base_dir, test_generator)

        os.makedirs(output_train_dir, exist_ok=True)
        os.makedirs(output_validation_dir, exist_ok=True)
        os.makedirs(output_test_dir, exist_ok=True)

    except Exception as e:
        print(f"Error saat memuat generator: {e}\nPastikan data augmentasi sudah tersusun dalam folder kategori (misal: train_augmented/cats/)")

    return train_generator, validation_generator, test_generator

preprocess_image('coffeebeans_raw/train', 'coffeebeans_raw/test')