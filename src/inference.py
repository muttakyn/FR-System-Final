import src.models.CustomModel as custom_model
import src.models.CustomModel as CustomModel
from src.data.dataset import ApparelDataset
from src.config import app_config
from src.models.CustomModel import ImageEmbedding
import src.utils.utils as utils


class Inference:

    # COnstructor to initialize and load data, embedding and recommendation table of items
    def __init__(self):
        self.apparel_meta = ApparelDataset(app_config['DATA_LABEL_PATH'], app_config['DATA_IMAGE_ROOT'])
        self.embedding_model = ImageEmbedding()
        self.embedding_map = utils.load_from_pickle('embeddings')
        self.candidate_images_ids = utils.load_from_pickle('candidate_images')
        self.candidate_images = self.apparel_meta.filter_by_ids(self.candidate_images_ids)
        self.candidate_images['emb'] = self.candidate_images.apply(lambda row: self.embedding_map[row['id']], axis=1)
        self.recommendations = []

    # Calculate embedding of all items
    def calculate_all_embeddings(self):
        all_metadata = self.apparel_meta.get_all_meta()
        for meta in all_metadata.iterrows():
            if meta[1]['id'] not in self.embedding_map:
                self.embedding_map[meta[1]['id']] = self.embedding_model.get_embedding(meta[1]['image'])

    # Recommend items for an specific item by id
    def recommend_by_id(self, image_id):
        filtered_meta = self.apparel_meta.filter_by_ids([image_id])
        if filtered_meta.shape[0] <= 0:
            print('No image found')
            return

        filtered_meta['emb'] = filtered_meta.apply(lambda row: self.embedding_map[row['id']], axis=1)
        self.recommendations = utils.get_top_10_similar_product(self.embedding_map[image_id], filtered_meta)

        return self.recommendations

    # Recommend items for an specific item by image
    def recommend_by_image(self, image_path):
        all_metadata = self.apparel_meta.get_all_meta()

        try:
            new_img_embedding = custom_model.get_embeddings(image_path)
            labels = ['Bags', 'Belts', 'Bottomwear', 'Eyewear', 'Flip Flops', 'Fragrance', 'Innerwear', 'Jewellery',
                      'Lips', 'Sandal', 'Shoes', 'Socks', 'Topwear', 'Wallets', 'Watches']
            other_labels = ["Dress", "Loungewear and Nightwear", "Saree", "Nails", "Makeup", "Headwear", "Ties",
                            "Accessories", "Scarves", "Cufflinks", "Apparel Set", "Free Gifts", "Stoles", "Skin Care",
                            "Skin", "Eyes", "Mufflers", "Shoe Accessories", "Sports Equipment", "Gloves", "Hair",
                            "Bath and Body", "Water Bottle", "Perfumes", "Umbrellas", "Wristbands",
                            "Beauty Accessories", "Sports Accessories", "Vouchers", "Home Furnishing"]
            print('Classifying')

            sub_categories = CustomModel.classify_image(labels, other_labels, image_path)
            print(sub_categories)
            modified_metadata = all_metadata[all_metadata['subCategory'].isin(sub_categories)]
            modified_metadata['emb'] = modified_metadata.apply(lambda row: self.embedding_map[row['id']], axis=1)

            gender, article_type = utils.get_article_type(new_img_embedding, self.candidate_images)
            modified_candidate_metadata = modified_metadata[modified_metadata['gender'] == gender]
            modified_candidate_metadata = modified_candidate_metadata[modified_metadata['articleType'] == article_type]

            if len(modified_candidate_metadata) > 0:
                modified_metadata = modified_candidate_metadata

            print(article_type)
            self.recommendations = utils.get_top_10_similar_product(new_img_embedding, modified_metadata)

            # self.apparel_meta.filter_by_sub_categories(sub_categories)
            return self.recommendations

        except IOError as io_error:
            print('Invalid model file, train and save a valid model')
            return
        except ImportError as import_error:
            print('No model found, train and save the model first')
            return
        except Exception as exception:
            print('Unknown error')
            print(exception)
            return
        finally:
            print('Done...')
            return

    # SHow recommendations through plotting figure
    def show_recommendation(self):
        utils.plot_figures(self.recommendations, nrows=2, ncols=5)
        return

    def save_embeddings_to_pickle(self):
        utils.save_to_pickle(self.embedding_map, 'embeddings')

    def load_embeddings_from_pickle(self):
        self.embedding_map = utils.load_from_pickle('embeddings')

    def generate_candidate_products(self):
        all_meta = self.apparel_meta.get_all_meta()
        all_meta['emb'] = all_meta.apply(lambda row: self.embedding_map[row['id']], axis=1)
        utils.generate_candidates(all_meta, save_as_pickle=True)
        return True


def print_ok():
    print('Ok')
