import numpy as np
import pandas as pd
from src.config import app_config
import os
import src.utils.metric as metric


class ModelTesting:
    def __init__(self, test_generator, model):
        self.test_generator = test_generator
        self.model = model

    def test_model(self, class_indices, save_as_csv=True, output_file_name='prediction.csv'):
        filenames = self.test_generator.filenames
        nb_samples = len(filenames)
        predict = self.model.predict_generator(self.test_generator, steps=nb_samples, verbose=1)
        predicted_class_indices = np.argmax(predict, axis=1)

        labels = dict((v, k) for k, v in class_indices.items())
        predictions = [labels[k] for k in predicted_class_indices]

        results = pd.DataFrame({"Filename": filenames,
                                })

        for k, v in labels.items():
            results[v] = predict[:, k]

        results['Actual'] = results.apply(lambda row: row['Filename'].split('/')[0], axis=1)
        results["Predictions"] = predictions

        if save_as_csv:
            results.to_csv(os.path.join(app_config['MODEL_LOG_PATH'], output_file_name), index=False)

        return metric.calculate_accuracy(filenames, predictions)
