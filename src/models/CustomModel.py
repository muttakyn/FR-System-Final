import keras
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import GlobalMaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras import optimizers
from src.config import app_config
import os
import numpy as np
import tensorflow as tf

# Input Shape
img_width, img_height, _ = 224, 224, 3  # load_image(df.iloc[0].image).shape

graph = tf.get_default_graph()

sess = tf.Session(graph=graph)
keras.backend.set_session(sess)

embedding_model = load_model(app_config['MODEL_WEIGHT_PATH'] + '/embedding-calculator.h5')
classifier_model = load_model(app_config['MODEL_WEIGHT_PATH'] + '/classifier.h5')


def classify_image(name_labels, other_labels, path):
    """
    Classify the given image into 15 sub-categories

    :param name_labels: labels of the 15 sub-categories
    :param other_labels: labels of all the other categories
    :param path: path of the query image
    :return: list of all predicted sub-classes
    """

    if not os.path.exists(path):
        print('Invalid image path')
        return

    keras.backend.clear_session()
    # Reshape
    img = load_img(path, target_size=(img_width, img_height))
    # img to Array
    x = img_to_array(img)
    # Expand Dim (1, w, h)
    x = np.expand_dims(x, axis=0)
    # Pre process Input
    x = preprocess_input(x)

    global sess
    global graph

    with graph.as_default():
        keras.backend.set_session(sess)
        x = classifier_model.predict(x).reshape(-1)
        threshold_flag = False

        # result = np.where(x == np.amax(x))
        max_ind = 0
        max_val = -1

        second_max_ind = 0
        second_max_val = -1

        ind = 0
        for prediction in x:
            if prediction > max_val:
                second_max_ind = max_ind
                second_max_val = max_val
                max_val = prediction
                max_ind = ind
            else:
                if prediction > second_max_val:
                    second_max_val = prediction
                    second_max_ind = ind

            if prediction >= app_config['CLASSIFIER_THRESHOLD']:
                threshold_flag = True
            ind = ind + 1

        print(x)
        if threshold_flag:
            return [name_labels[max_ind]]
        else:
            other_labels.append(name_labels[max_ind])

            if second_max_val > 0.2:
                other_labels.append(name_labels[second_max_ind])
            print(other_labels)
            return other_labels


def get_embeddings(image_path):
    """
    Calculate image feature map or embedding of a given image

    :param image_path: path of the image
    :return: embedding of the image if successfully extracted the feature, empty array otherwise
    """
    keras.backend.clear_session()
    if os.path.exists(image_path):
        img = keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
        # img to Array
        x = keras.preprocessing.image.img_to_array(img)
        # Expand Dim (1, w, h)
        x = np.expand_dims(x, axis=0)
        # Pre process Input
        x = preprocess_input(x)

        global sess
        global graph

        with graph.as_default():
            keras.backend.set_session(sess)
            y = embedding_model.predict(x).reshape(-1).tolist()
            return y
    else:
        return []


def load_model_from_disk(model_name):
    """
    Load a model from the disk

    :param model_name: name of the model to be loaded
    :return: loaded model
    """
    try:
        model = load_model(os.path.join(app_config['MODEL_WEIGHT_PATH'], model_name))
        return model
    except ImportError as err:
        print('Model not available, please train first')
        return err
    except IOError as err:
        print('Invalid file')
        return err


class Classifier:
    """Image classifier"""

    def __init__(self, number_of_classes=15, learning_rate=0.00005, activation="softmax",
                 loss="categorical_crossentropy"):
        """
        Initiate an image classifier

        :param number_of_classes: number of classes of the model
        :param learning_rate: learning rate of the classifier model
        :param activation: activation function of the prediction layer
        :param loss: loss function of the classifier model
        """
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
        """
        Get the classifier model

        :return: compiled classifier
        """
        return self.model

    def save_model_to_disk(self, model_name):
        """
        Save model to disk

        :param model_name: name of the model will be
        :return:
        """
        self.model.save(os.path.join(app_config['MODEL_WEIGHT_PATH'], model_name))


class ImageEmbedding:
    """Image Embedding"""
    def __init__(self):
        """
        Initiate the embedding class
        """
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
        """
         Get the embedding model

        :return: compiled feature extractor model
        """
        return self.model

    def save_model_to_disk(self, model_name):
        """
        Save model to disk

        :param model_name: name of the model will be
        :return:
        """
        self.model.save(os.path.join(app_config['MODEL_WEIGHT_PATH'], model_name))

    def get_embedding(self, img_name):
        """
        Extract feature from the image.

        :param img_name: name of the image
        :return: feature map of the image
        """
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
            return [0] * self.model.output_shape[1]
