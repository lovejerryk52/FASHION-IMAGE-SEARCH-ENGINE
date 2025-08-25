from features_extractor import FeatureExtractor
import pickle
from PIL import Image
import numpy as np
from io import BytesIO
import pandas as pd
import time

FEATURES_DATABASE = "./features_database/image_features_"
DATASET_PATH = "./dataset"
MODEL_NAMES = ["VGG16", "ResNet50", "Xception"]

def load_listings():
   df = pd.read_csv(DATASET_PATH + "/current_farfetch_listings.csv")
   return df

def search_similar_images(query_image):

    listing_data = load_listings()

    for model_name in MODEL_NAMES:
      t_start = time.time()
      # extract features of query image
      extractor = FeatureExtractor(arch=model_name)
      query_features = extractor.extract_features(query_image)

      # read features in database
      with open(FEATURES_DATABASE + model_name.lower() + ".pkl", "rb") as file:
          extracted_features = pickle.load(file)


      distance_scores = {}
      for idx, feat in extracted_features.items():
          distance_scores[idx] = np.sum((query_features - feat)**2) ** 0.5

      sorted_distance_scores = sorted(distance_scores.items(), key = lambda x : x[1], reverse=False)

      top_10_distance_scores = [score for _, score in sorted_distance_scores][ : 10]
      top_10_indexes = [idx for idx, _ in sorted_distance_scores][ : 10]

      similar_images_paths = listing_data.iloc[top_10_indexes]['images.model']
      similar_images_brands = listing_data.iloc[top_10_indexes]['brand.name']

      t_end = time.time()

      yield {
          "name": model_name,
          "paths": similar_images_paths, 
          "brands": similar_images_brands,
          "scores": top_10_distance_scores,
          "time": t_end-t_start
      }