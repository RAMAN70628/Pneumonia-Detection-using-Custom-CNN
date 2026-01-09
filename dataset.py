import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(224,224), batch_size=32):

    train_path = os.path.join(data_dir, "train")
    test_path  = os.path.join(data_dir, "test")

    train_normal = glob.glob(train_path + "/NORMAL/*.jpeg")
    train_pneumonia = glob.glob(train_path + "/PNEUMONIA/*.jpeg")

    test_normal = glob.glob(test_path + "/NORMAL/*.jpeg")
    test_pneumonia = glob.glob(test_path + "/PNEUMONIA/*.jpeg")

    train_list = train_normal + train_pneumonia
    test_list  = test_normal + test_pneumonia

    df_train = pd.DataFrame({
        "image": train_list,
        "class": ["Normal"] * len(train_normal) + ["Pneumonia"] * len(train_pneumonia)
    })

    df_test = pd.DataFrame({
        "image": test_list,
        "class": ["Normal"] * len(test_normal) + ["Pneumonia"] * len(test_pneumonia)
    })

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.1
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        df_train,
        x_col="image",
        y_col="class",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )

    val_generator = train_datagen.flow_from_dataframe(
        df_train,
        x_col="image",
        y_col="class",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )

    test_generator = test_datagen.flow_from_dataframe(
        df_test,
        x_col="image",
        y_col="class",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_generator, val_generator, test_generator
