from src.data.dataset import ApparelDataset
from src.config import app_config
from src.models.CustomModel import ImageEmbedding
from sklearn.metrics.pairwise import pairwise_distances
import src.utils.utils as utils


class Inference:

    def __init__(self):
        self.apparel_meta = ApparelDataset(app_config['DATA_LABEL_PATH'], app_config['DATA_IMAGE_ROOT'])
        self.embedding_model = ImageEmbedding()
        self.embedding_map = {}
        self.recommendations = []

    def calculate_all_embeddings(self):
        for meta in self.apparel_meta.iterrows():
            self.embedding_map[meta[1]['id']] = self.embedding_model.get_embedding(meta[1]['image'])

    def recommend_by_id(self, image_id):
        filtered_meta = self.apparel_meta.filter_by_id(image_id)
        if filtered_meta.shape[0] <= 0:
            print('No image found')
            return

        filtered_image_embeddings = []
        for meta in filtered_meta.iterrows():
            if meta[1]['id'] not in self.embedding_map:
                self.embedding_map[meta[1]['id']] = \
                    self.embedding_model.get_embedding(meta[1]['image'])

            filtered_image_embeddings.append(self.embedding_map[meta[1]['id']])

        query_image_embedding = [self.embedding_map[image_id]]

        sim_scores = 1 - pairwise_distances(query_image_embedding, filtered_image_embeddings, metric='cosine')
        sim_scores = sim_scores[0]

        self.recommendations = []
        i = 0
        for meta in filtered_meta.iterrows():
            self.recommendations.append(
                (meta[1]['id'], sim_scores[i], meta[1]['articleType']))
            i = i + 1

        self.recommendations = sorted(self.recommendations, key=lambda x: x[1], reverse=True)
        self.recommendations = self.recommendations[: 10]

        return self.recommendations

    def recommend_by_image(self, image_path):
        filtered_meta = self.apparel_meta.filter_by_sub_categories(['Topwear'])
        print(filtered_meta.head(5))

    def show_recommendation(self):
        utils.plot_figures(self.recommendations, nrows=2, ncols=5)


def print_ok():
    print('Ok')
