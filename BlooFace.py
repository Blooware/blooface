# This is the main file for Blooface.
import os
import random
from deepface import DeepFace
import pandas as pd
from annoy import AnnoyIndex
from deepface.commons import functions
from deepface.basemodels import Facenet


class Blooface:
    def __init__(self, database_path="dataset/", file_type=".jpg", embedding_size=128, train=True):
        self.database_path = database_path
        self.file_type = file_type
        self.embedding_size = embedding_size
        self.model = Facenet.loadModel()
        if (train):
            self.train()

    def train(self, synthesize=0):
        files = []
        for r, d, f in os.walk(self.database_path):
            for file in f:
                if (self.file_type in file):
                    exact_path = r + file
                    files.append(exact_path)

        representations = []
        for img_path in files:
            img = functions.preprocess_face(img=img_path, target_size=(160, 160), enforce_detection=False, detector_backend='mtcnn')
            embedding = self.model.predict(img)[0,:]
            
            representation = []
            representation.append(img_path)
            representation.append(embedding)
            representations.append(representation)

        if (synthesize > 0):
            for i in range(0, synthesize):
                key = 'dataset/synth_%d.jpg' % (i)
                vector = [random.gauss(-0.35, 0.48) for z in range(self.embedding_size)]
                dummy_item = []
                dummy_item.append(key)
                dummy_item.append(vector)
                representations.append(dummy_item)

        df = pd.DataFrame(representations, columns = ["img_name", "embedding"])
        df.to_pickle("embeddings.pkl")

        t = AnnoyIndex(self.embedding_size, 'euclidean')
 
        for i in range(0, len(representations)):
            representation = representations[i]
            img_path = representation[0]
            embedding = representation[1]
            t.add_item(i, embedding)

        t.build(10)
        t.save('result.ann')

    def query_embedding(self, embedding, num_results=2):
        embeddings = pd.read_pickle("embeddings.pkl")
        t = AnnoyIndex(self.embedding_size, 'euclidean')
        t.load('result.ann')
        neighbors = t.get_nns_by_vector(embedding, num_results)
        img_paths = []
        for neighbor in neighbors:
            img_paths.append(embeddings.iloc[neighbor]["img_name"])
        
        return img_paths

    def query_image(self, img_path, num_results=2):
        img, region = functions.preprocess_face(img=img_path, target_size=(160, 160), enforce_detection=False, return_region=True, detector_backend='mtcnn')
        test_image = self.model.predict(img)[0,:]
        return self.query_embedding(test_image, num_results), region

    def embedding(self, img_path):
        img = functions.preprocess_face(img=img_path, target_size=(160, 160))
        embedding = self.model.predict(img)[0,:]
        return embedding

    def detect(self, frame, enforce=False, model='mtcnn'):
        return functions.detect_face(frame, detector_backend=model, enforce_detection=enforce)

    def verify(self, img_1, img_2):
        return DeepFace.verify(img_1, img_2, distance_metric='euclidean', detector_backend='mtcnn')