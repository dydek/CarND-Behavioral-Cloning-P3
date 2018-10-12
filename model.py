import csv
import json
import os

import cv2
import numpy as np
from keras.backend import name_scope

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Conv2D, Cropping2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

STEERING_CORRECTION = 0.15


def get_samples():
    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append([d.strip() for d in line])
    return lines


def create_model():
    """
    Based on https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    model = Sequential()
    # cropping
    with name_scope('preparing'):
        model.add(Cropping2D(cropping=((25, 10), (0, 0)), input_shape=(80, 160, 3)))
        # normalization
        model.add(Lambda(lambda x: x / 255.0 - 0.5))
    with name_scope('convolution'):
        model.add(
            Conv2D(24, (5, 5), activation='relu', name='Conv_stage_1', strides=(2, 2), padding='same')
        )
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(
            Conv2D(36, (5, 5), activation='relu', strides=(2, 2), padding='same')
        )
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(
            Conv2D(48, (5, 5), activation='relu', strides=(2, 2), padding='same')
        )
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(
            Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        )
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(
            Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        )
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    with name_scope('dense'):
        model.add(Flatten())
        model.add(Dense(1164, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))

    model.summary()
    model.compile(loss='mse', optimizer='adam')
    return model


def process_image(original_image):
    """
    - resize the image
    - change colot back to RGB
    :param img:
    :return:
    """
    if original_image is not None:
        height, width = original_image.shape[:2]

        image = cv2.resize(original_image, (width // 2, height // 2), interpolation=cv2.INTER_CUBIC)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_model(model):
    model_json = model.to_json()
    with open('./model.json', 'w') as json_file:
        json.dump(model_json, json_file)
    model.save('./model.h5')
    model.save_weights('./model.weights')
    print('--- Saved model to disk ----')


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image_center = process_image(cv2.imread(
                    os.path.join('data', batch_sample[0])
                ))

                image_flip = cv2.flip(image_center, 1)

                image_left = process_image(cv2.imread(
                    os.path.join('data', batch_sample[1])
                ))

                image_right = process_image(cv2.imread(
                    os.path.join('data', batch_sample[2])
                ))

                # run next loop if any of images would be None or float conversion fails
                # if None in [image_center, image_flip, image_left, image_right]:
                #     continue
                #
                try:
                    steering = float(batch_sample[3])
                except ValueError:
                    continue

                images.extend(
                    [image_center, image_flip, image_left, image_right]
                )
                angles.extend(
                    [steering, -steering, steering + STEERING_CORRECTION, steering - STEERING_CORRECTION]
                )

            yield shuffle(np.array(images), np.array(angles))


def save_training_history(history):
    with open('./history.json', 'w') as json_file:
        json.dump(history, json_file)


if __name__ == '__main__':

    train_samples, validation_samples = train_test_split(get_samples(), test_size=0.2)

    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = create_model()
    # updates args for keras 2.0 api
    history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
                                         validation_data=validation_generator,
                                         validation_steps=len(validation_samples), epochs=3, verbose=1)

    save_model(model)
    save_training_history(history_object.history)
