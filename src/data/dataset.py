import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.utils.utils import InvalidValueError

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class ApparelDataset(Dataset):
    """Apparel dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.apparel_meta = pd.read_csv(csv_file, error_bad_lines=False)
        self.apparel_meta['image'] = self.apparel_meta.apply(lambda row: str(row['id']) + ".jpg", axis=1)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.apparel_meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.apparel_meta.iloc[idx]

    def get_image(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                str() + ".jpg")
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image

    def get_candidate_meta(self, min_count=500, per_class=1000):
        modified_image_meta = self.apparel_meta.groupby('subCategory').filter(lambda x: len(x) >= min_count)
        return modified_image_meta.groupby('subCategory').head(per_class)

    def filter_by_id(self, image_id):
        try:
            product_meta = self.apparel_meta[self.apparel_meta['id'] == image_id]
            if len(product_meta) != 1:
                raise InvalidValueError('Invalid value', 'Unknown image id')

            filtered_product_meta = self.apparel_meta[self.apparel_meta['gender'] == product_meta['gender'].iloc[0]]
            filtered_product_meta = filtered_product_meta[
                filtered_product_meta['articleType'] == product_meta['articleType'].iloc[0]]

        except InvalidValueError as invalid_id_except:
            print(invalid_id_except.errors)
            return pd.DataFrame()

        except Exception as exception:
            print(exception)
            return pd.DataFrame()

        finally:
            print('Done....')

        return filtered_product_meta

    def filter_by_sub_categories(self, sub_categories):

        try:
            if type(sub_categories) is not list:
                raise InvalidValueError('Invalid value', 'Sub-categories should be a list')

            filtered_product_meta = self.apparel_meta[self.apparel_meta['subCategory'].isin(sub_categories)]

        except InvalidValueError as invalid_value_exception:
            print(invalid_value_exception.errors)
            return pd.DataFrame()

        finally:
            print('Done...')

        return filtered_product_meta
