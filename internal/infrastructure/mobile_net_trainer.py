import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import math
from internal.domain.training.model_trainer import ModelTrainer
import matplotlib.pyplot as plt

DATASET_PREFIX = "v3"
BATCH_SIZE = 160
EPOCHS = 20
INITIAL_EPOCH = 0
LEARNING_RATE = 1e-4
IMG_SIZE = (224, 224)
OUTPUT_NORMALIZATION = 655.35

def configure_gpu(memory_fraction=0.75):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_fraction * 1024 * 4)])
        except RuntimeError as e:
            print("Error configurando GPU:", e)

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    return preprocess_input(arr)

def data_generator(df, batch_size, img_dir, should_shuffle=True):
    if should_shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    img_list = df['front'].values
    wheel_axis = df['wheel-axis'].values
    num_samples = len(img_list)
    index = 0
    while True:
        batch_img = np.zeros((batch_size, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        for i in range(batch_size):
            if index >= num_samples:
                index = 0
                df = df.sample(frac=1).reset_index(drop=True)
                img_list = df['front'].values
                wheel_axis = df['wheel-axis'].values
            img_path = os.path.join(img_dir, img_list[index])
            arr = load_and_preprocess_image(img_path)
            batch_img[i] = arr
            batch_label[i] = wheel_axis[index] / OUTPUT_NORMALIZATION
            index += 1
        yield batch_img, batch_label

def build_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)
    return model

def plot_and_save_history(history, filename="training_history.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title('Curvas de Pérdida')
    plt.xlabel('Epoch')
    plt.ylabel('Error Cuadrático Medio')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Gráfica guardada como {filename}")

class MobileNetTrainer(ModelTrainer):

    def train(self) -> None:
        configure_gpu()
        parent_path = os.path.dirname(os.getcwd())
        data_path = os.path.join(parent_path, 'data')
        img_front_dir_path = os.path.join(data_path, 'img', 'front')
        csv_dir_path = os.path.join(data_path, 'csv', 'final')
        model_path = os.path.join(parent_path, 'model')
        log_path = os.path.join(model_path, 'log')

        train_csv = os.path.join(csv_dir_path, DATASET_PREFIX + '_train.csv')
        valid_csv = os.path.join(csv_dir_path, DATASET_PREFIX + '_valid.csv')

        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(valid_csv)
        print(f"Train samples: {df_train.shape[0]}, Validation samples: {df_val.shape[0]}")

        train_steps = math.ceil(df_train.shape[0] / BATCH_SIZE)
        val_steps = math.ceil(df_val.shape[0] / BATCH_SIZE)

        train_gen = data_generator(df_train, BATCH_SIZE, img_front_dir_path, should_shuffle=True)
        val_gen = data_generator(df_val, BATCH_SIZE, img_front_dir_path, should_shuffle=False)

        model = build_model()
        optimizer = Adam(lr=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss="mse")
        model.summary()

        cur_model_name = f"{DATASET_PREFIX}-MobileNetV2_Reg"
        csv_logger = CSVLogger(os.path.join(log_path, cur_model_name + '.log'))
        checkpoint_filepath = os.path.join(model_path, cur_model_name + '-{epoch:03d}-{val_loss:.5f}.h5')
        checkpoint = ModelCheckpoint(checkpoint_filepath, verbose=1, save_best_only=True)

        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=[csv_logger, checkpoint],
            initial_epoch=INITIAL_EPOCH
        )

        plot_and_save_history(history, filename="training_history.png")
        final_model_path = os.path.join(model_path, cur_model_name + '_final.h5')
        model.save(final_model_path)
        print(f"Modelo final guardado en {final_model_path}")


