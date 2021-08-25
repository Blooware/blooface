# Large scale face recognition with spotify annoy
import re
import os
import cv2

embedding_size = 128
# Get every JPG image in the dataset and store it's path
print("Loading images from dataset...")
import os
files = []
for r, d, f in os.walk("dataset/"):
    for file in f:
        if ('.jpg' in file):
            exact_path = r + file
            files.append(exact_path)


# files = []
# raw_files = os.listdir("dataset/")
# for file in raw_files:
#     if file.endswith(".jpg"):
#         # remove all letters
#         file = re.sub("[^0-9]", "", file)
#         files.append(int(file.replace(".jpg", "")))

# files = sorted(files)
# new_files = []
# for file in files:
#     new_files.append("dataset/" + str(file) + ".jpg")

# files = new_files


from deepface.commons import functions
from deepface.basemodels import Facenet

print("Loading model...")
model = Facenet.loadModel()

print("Generating embeddings...")
representations = []
for img_path in files:
    
    img = functions.preprocess_face(img=img_path, target_size=(160, 160))
    embedding = model.predict(img)[0,:]
     
    representation = []
    representation.append(img_path)
    representation.append(embedding)
    representations.append(representation)

print(representations)
# Generate sythetic data
print("Generating synthetic data...")
import random
for i in range(70, 100000): #1M instances
    key = 'dataset/img_%d.jpg' % (i)
    vector = [random.gauss(-0.35, 0.48) for z in range(embedding_size)]
 
    dummy_item = []
    dummy_item.append(key)
    dummy_item.append(vector)
    representations.append(dummy_item)

# save as datframe
import pandas as pd
df = pd.DataFrame(representations, columns = ["img_name", "embedding"])
df.to_pickle("embeddings.pkl")

# Spotify annoy
print("Training annoy...")
from annoy import AnnoyIndex
import time
t = AnnoyIndex(embedding_size, 'euclidean')
 
for i in range(0, len(representations)):
   representation = representations[i]
   img_path = representation[0]
   embedding = representation[1]
    
   t.add_item(i, embedding)
 
t.build(10) #3 trees

#save the built model
t.save('result.ann')

print("Done!")