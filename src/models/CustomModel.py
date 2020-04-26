from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import GlobalMaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras import optimizers
from src.config import app_config
import os
import numpy as np


class Classifier:

    def __init__(self, number_of_classes=15, learning_rate=0.00005, activation="softmax",
                 loss="categorical_crossentropy"):
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.activation = activation
        self.loss = loss

        self.base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        for layer in self.base_model.layers[:170]:
            layer.trainable = False

        x = self.base_model.output
        x = Flatten()(x)

        self.prediction = Dense(self.number_of_classes, activation=self.activation)(x)

        self.model = Model(inputs=self.base_model.input, outputs=self.prediction)

        self.model.compile(loss=self.loss,
                           optimizer=optimizers.SGD(lr=self.learning_rate),
                           metrics=['accuracy'])

    def get_model(self):
        return self.model

    def save_model_to_disk(self, model_name):
        self.model.save(os.path.join(app_config['MODEL_WEIGHT_PATH'], model_name ))


class ImageEmbedding:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.base_model = ResNet50(weights='imagenet',
                                   include_top=False,
                                   input_shape=self.input_shape)
        self.base_model.trainable = False

        self.model = Sequential([
            self.base_model,
            GlobalMaxPooling2D()
        ])

    def get_model(self):
        return self.model

    def save_model_to_disk(self, model_name):
        self.model.save(os.path.join(app_config['MODEL_WEIGHT_PATH'], model_name))

    def get_embedding(self, img_name):
        image_path = os.path.join(app_config['DATA_IMAGE_ROOT'], img_name)
        # Reshape
        if os.path.exists(image_path):
            img = load_img(image_path, target_size=(self.input_shape[0], self.input_shape[1]))
            # img to Array
            x = img_to_array(img)
            # Expand Dim (1, w, h)
            x = np.expand_dims(x, axis=0)
            # Pre process Input
            x = preprocess_input(x)
            return self.model.predict(x).reshape(-1)
        else:
            return [0]*self.model.output_shape[1]


def load_model_from_disk(self, model_name):
    try:
        model = load_model(os.path.join(app_config['MODEL_WEIGHT_PATH'], model_name))
        return model
    except ImportError as err:
        print('Model not available, please train first')
        return err
    except IOError as err:
        print('Invalid file')
        return err
