from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from src.utils.utils import ImageFileUtils
from src.config import app_config
import os


class Preprocessing:
    def __init__(self, image_meta, image_width=224, image_height=224):
        self.input_dim = (image_height, image_width)
        self.image_meta = image_meta

        self.image_meta.sample(frac=1)
        self.train_image_meta, self.test_image_meta = train_test_split(self.image_meta, test_size=0.2)
        self.train_image_meta, self.val_image_meta = train_test_split(self.train_image_meta, test_size=0.2)

        self.test_file_util = ImageFileUtils(self.test_image_meta, 'subCategory')
        self.train_file_util = ImageFileUtils(self.train_image_meta, 'subCategory')
        self.validation_file_util = ImageFileUtils(self.val_image_meta, 'subCategory')

        self.train_datagen = ImageDataGenerator(rotation_range=30,
                                                zoom_range=0.15,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                shear_range=0.15,
                                                horizontal_flip=True,
                                                fill_mode="nearest")
        self.val_datagen = ImageDataGenerator()
        self.test_datagen = ImageDataGenerator()
        return

    def move_image(self):
        self.train_file_util.move_image_to_dir('train')
        self.test_file_util.move_image_to_dir('test')
        self.validation_file_util.move_image_to_dir('validation')
        return

    def return_image_to_inventory(self):
        self.train_file_util.move_images_to_inventory('train')
        self.test_file_util.move_images_to_inventory('test')
        self.validation_file_util.move_images_to_inventory('validation')
        return

    def get_data_generator(self, generator_for, batch_size=10):
        if generator_for == 'train':
            return self.train_datagen.flow_from_directory(directory=os.path.join(app_config['DATA_IMAGE_ROOT'], 'train'),
                                                          target_size=self.input_dim,
                                                          color_mode='rgb',
                                                          batch_size=batch_size,
                                                          class_mode="categorical",
                                                          shuffle=True,
                                                          seed=42)
        if generator_for == 'test':
            return self.test_datagen.flow_from_directory(directory=os.path.join(app_config['DATA_IMAGE_ROOT'], 'test'),
                                                         target_size=self.input_dim,
                                                         color_mode='rgb',
                                                         batch_size=1,
                                                         class_mode="categorical",
                                                         shuffle=False)
        if generator_for == 'validation':
            return self.train_datagen.flow_from_directory(directory=os.path.join(app_config['DATA_IMAGE_ROOT'],
                                                                                 'validation'),
                                                          target_size=self.input_dim,
                                                          color_mode='rgb',
                                                          batch_size=batch_size,
                                                          class_mode="categorical",
                                                          shuffle=True)
        return
