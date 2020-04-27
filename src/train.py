import matplotlib.pyplot as plt
from src.data.dataset import ApparelDataset
from src.models.CustomModel import Classifier
from src.config import app_config
from src.data.preprocessing import Preprocessing
from src.test import ModelTesting


def plot_train_history(history):
    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 4.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
    except:
        print('Invalid history')
        return


class ModelTraining:
    def __init__(self, learning_rate=0.00005, activation="softmax", loss="categorical_crossentropy",
                 min_value_count=500, per_class=527):
        self.learning_rate = learning_rate
        self.activation = activation
        self.loss = loss
        self.number_of_classes = 15
        self.min_value_count = min_value_count
        self.per_class = per_class

        self.classifier = Classifier(self.number_of_classes, self.learning_rate, self.activation, self.loss)
        self.apparel_data = ApparelDataset(app_config['DATA_LABEL_PATH'], app_config['DATA_IMAGE_ROOT'])

        self.model = self.classifier.get_model()

        self.candidate_meta = self.apparel_data.get_candidate_meta(min_value_count, per_class)
        self.data_processor = Preprocessing(self.candidate_meta)
        self.data_processor.move_image()
        self.validation_data_generator = self.data_processor.get_data_generator('validation')
        self.train_data_generator = self.data_processor.get_data_generator('train')
        self.test_data_generator = self.data_processor.get_data_generator('test')

    def train_model(self, epoch=10):
        train_history = self.model.fit_generator(self.train_data_generator,
                                                 steps_per_epoch=None,
                                                 epochs=epoch,
                                                 verbose=1,
                                                 validation_data=self.validation_data_generator)

        self.classifier.save_model_to_disk('classifier.h5')

        return train_history

    def evaluate_model(self):
        class_indices = self.validation_data_generator.class_indices
        tester = ModelTesting(self.test_data_generator, self.model)
        return tester.test_model(class_indices)

    def reset_data_processor(self):
        self.data_processor.return_image_to_inventory()


