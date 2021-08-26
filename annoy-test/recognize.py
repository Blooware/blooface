import cv2
import time
import pandas as pd
from annoy import AnnoyIndex
from deepface import DeepFace
from deepface.commons import functions

embedding_size = 128
model = DeepFace.build_model("Facenet")

# Load data from train cycle
embeddings = pd.read_pickle("embeddings.pkl")
t = AnnoyIndex(embedding_size, 'euclidean')
t.load('result.ann')

# Generate test image embedding
test_image = cv2.imread('dataset/test.jpg')
test_image = functions.preprocess_face(test_image, target_size=(160, 160))
test_image = model.predict(test_image)[0]

k = 2 # number of nearest neighbors
neighbors = t.get_nns_by_vector(test_image, k)
file_name = embeddings.iloc[neighbors[1]]["img_name"]

# Declare results
print("Nearest neighbors: ", neighbors)
print("File name: ", file_name)