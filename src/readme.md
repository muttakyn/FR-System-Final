## Fashion Recommender(FR) System

Fashion Recommender engine recommends fashion products based on visual similarity. You can run the code following the next steps:
1. Get the dataset in your specified path.
2. You can run the entire system running run.sh file. If you want to run it step by step then follow step 3 and onward.
3. Run functions for data loading.
4. Run functions for data preprocessing.
5. Then load the keras model for training and image embedding running the required functions.
6. Run embedding calculation for every image.
7. Run cosine similarity functions for getting similarity matrix.
8. for training the model for classification, run functions for training.
9. for candidate based classification, run required functions.
10. Call the recommender function with image id to get visually similar fashion products.

Commands
-
1. ##### To train the model
    ```train
    from src.train import ModelTraining
    model_training = ModelTraining()
    model_training.train_model()
    model_training.reset_data_processor()
    ```
2. ##### Save embedding calculator model to disk
    ```save embedding model
       from src.models.CustomModel import ImageEmbedding;
       embedding_calculator = ImageEmbedding();
       embedding_calculator.save_model_to_disk("embedding-calculator.h5")
   ```
3. ##### Calculate image embeddings and generate candidate products from each gender-articleType group
    ```Calculate image embeddings
       from src.models.CustomModel import EmbeddingCalculator
       embedding_calculator = EmbeddingCalculator()
       embedding_calculator.calculate_all_embeddings()
       embedding_calculator.save_embeddings_to_pickle()
       embedding_calculator.generate_candidate_products()
   ```
 4. ##### Make recommendation by image id (Known recommendation)
    ```Make recommendation
       from src.inference import Inference
       inferrer = Inference()
       inferrer.recommend_by_id(1542)
       inferrer.show_recommendation()
    ```
 5. ##### Make recommendation by image path with no filtering(Unknown recommendation 1)
    ```Make recommendation 2
       from src.inference import Inference
       inferrer = Inference()
       inferrer.recommend_by_image("data/inference/brush.jpg")
       inferrer.show_recommendation()
    ```
 6. ##### Make recommendation by image path with filtering(Unknown recommendation 2)
    ```Make recommendation 2
       from src.inference import Inference
       inferrer = Inference()
       inferrer.recommend_by_image("data/inference/brush.jpg", article_type="Shoe Accessories")
       inferrer.show_recommendation()
    ```
    
---
---
---
---