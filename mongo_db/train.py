from pymongo import MongoClient
connection = "mongodb://localhost:27017"
 
client = MongoClient(connection)
 
database = 'deepface'; collection = 'deepface'
 
db = client[database]

#!pip install deepface
from deepface import DeepFace
model = DeepFace.build_model("Facenet")

import os 
facial_img_paths = []
for root, directory, files in os.walk("./../dataset"):
    for file in files:
        if '.jpg' in file:
            facial_img_paths.append(root+"/"+file)

from deepface.commons import functions
from tqdm import tqdm
 
instances = []
 
for i in tqdm(range(0, len(facial_img_paths))):
    facial_img_path = facial_img_paths[i]
    print(facial_img_path)
    facial_img = functions.preprocess_face(facial_img_path, target_size = (160, 160))
     
    embedding = model.predict(facial_img)[0]
     
    instance = []
    instance.append(facial_img_path)
    instance.append(embedding)
    instances.append(instance)