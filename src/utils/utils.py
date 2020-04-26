import os
from src.config import app_config
import shutil
import matplotlib.pyplot as plt
import cv2


def plot_figures(figures, nrows=1, ncols=1, figsize=(5, 5)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind in range(len(figures)):
        fig = figures[ind]
        img = read_image_by_id(fig[0])
        axeslist.ravel()[ind].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(str(fig[0])+'-'+fig[2])
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    return


def read_image_by_id(image_id):
    path = os.path.join(app_config['DATA_IMAGE_ROOT'], str(image_id) + '.jpg')
    if os.path.exists(path):
        img = cv2.imread(path)
        return cv2.resize(img, (224, 224))
    print('No image found!')


class InvalidValueError(Exception):
    def __init__(self, message, errors):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class ImageFileUtils:
    def __init__(self, meta, column_name):
        self.meta = meta
        self.column_name = column_name

    # move all train/test/validation images to train/test/validation folder
    # train/test/validation folder contains same number of sub-folder for each class(column_name)
    def move_image_to_dir(self, dir_name):
        actual_moved = {}
        image_path = app_config['DATA_IMAGE_ROOT']
        for i, img in self.meta.iterrows():

            if not os.path.exists(os.path.join(image_path, dir_name + '/' + img[self.column_name])):
                os.makedirs(os.path.join(image_path, dir_name + '/' + img[self.column_name]))

            if os.path.isfile(os.path.join(image_path, img['image'])):
                os.rename(os.path.join(image_path, img['image']),
                          os.path.join(image_path, dir_name + '/' + img[self.column_name] + '/' + img['image']))

                if img[self.column_name] in actual_moved:
                    actual_moved[img[self.column_name]] = actual_moved[img[self.column_name]] + 1
                else:
                    actual_moved[img[self.column_name]] = 1
        return actual_moved

    # move train/test/validation images in single directory(inventory)

    def move_images_to_inventory(self, dir_name):
        image_path = app_config['DATA_IMAGE_ROOT']

        for column_name in self.meta[self.column_name].unique():
            source = os.path.join(image_path, dir_name + '/' + column_name)

            if os.path.isdir(source):
                files = os.listdir(source)
                for f in files:
                    shutil.move(source + '/' + f, image_path)
                os.rmdir(source)
        return
