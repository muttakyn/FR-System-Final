import os
from src.config import app_config
import shutil
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics.pairwise import pairwise_distances
from scipy import spatial
import pickle


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
        axeslist.ravel()[ind].set_title(str(fig[0]) + '-' + fig[2])
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    return


def read_image_by_id(image_id):
    path = os.path.join(app_config['DATA_IMAGE_ROOT'], str(image_id) + '.jpg')
    if os.path.exists(path):
        img = cv2.imread(path)
        return cv2.resize(img, (224, 224))
    print('No image found!')


def get_article_type(emb_new_item, candidate_image_meta):
    if candidate_image_meta is None:
        print('No candidate image found!')
        return

    if len(candidate_image_meta) <= 0:
        print('No candidate found')
        return

    tmp = 0
    most_similar = None
    for ind, row in candidate_image_meta.iterrows():
        sim_score = 1 - spatial.distance.cosine(emb_new_item, row.emb)
        if sim_score > tmp:
            most_similar = row
            tmp = sim_score

    return most_similar['gender'], most_similar['articleType']


def generate_candidates(apparel_data, save_as_pickle=True):
    candidates = []
    unique_types = set(apparel_data['genderArticle'])
    print('Unique types ' + str(len(unique_types)))
    for group in unique_types:
        apparel_data_group = apparel_data[apparel_data['genderArticle'] == group]
        print('Group ' + group)
        print(len(apparel_data_group))
        score_map = {}
        embeddings_group = []
        for ind, row in apparel_data_group.iterrows():
            embeddings_group.append(row.emb)

        sim_scores = 1 - pairwise_distances(embeddings_group, metric='cosine')

        i = 0
        for inner_ind, data in apparel_data_group.iterrows():
            score_map[data.id] = sum(sim_scores[i])
            i += 1

        candidates.append(max(score_map, key=score_map.get))

    print(len(candidates))
    if save_as_pickle:
        save_to_pickle(candidates, 'candidate_images')

    return


def save_to_pickle(data, file_name):
    try:
        candidate_meta_file = open(os.path.join(app_config['PROCESSED_FILE_PATH'], file_name), 'wb')
        pickle.dump(data, candidate_meta_file)
        candidate_meta_file.close()
        print('Saved...')
        return True
    except Exception as error:
        print('Error occurred while saving in file')
        print(error)
        return False
    finally:
        print('Done...')


def load_from_pickle(file_name):
    try:
        file = open(os.path.join(app_config['PROCESSED_FILE_PATH'], file_name), 'rb')
        return pickle.load(file)
    except Exception as error:
        print('Error reading pickle file')
        print(error)
        return []
    finally:
        print('Done...')


def get_top_10_similar_product(query_image_embedding, filtered_meta):
    filtered_image_embeddings = []
    for ind, meta in filtered_meta.iterrows():
        if len(meta.emb) != 2048:
            print(meta.id)
            meta.emb = [0]*2048
        filtered_image_embeddings.append(meta.emb)

    query_image_embedding = [query_image_embedding]

    sim_scores = 1 - pairwise_distances(query_image_embedding, filtered_image_embeddings, metric='cosine')
    sim_scores = sim_scores[0]

    recommendations = []
    i = 0
    for meta in filtered_meta.iterrows():
        recommendations.append(
            (meta[1]['id'], sim_scores[i], meta[1]['articleType']))
        i = i + 1

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    recommendations = recommendations[: 10]
    return recommendations


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
