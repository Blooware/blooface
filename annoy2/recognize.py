from annoy import AnnoyIndex
import time
import re

embedding_size = 128
# Get every JPG image in the dataset and store it's path
print("Loading images from dataset...")
import os

files = []
raw_files = os.listdir("dataset/")
for file in raw_files:
    if file.endswith(".jpg"):
        # remove all letters
        file = re.sub("[^0-9]", "", file)
        files.append(int(file.replace(".jpg", "")))

files = sorted(files)
new_files = []
for file in files:
    new_files.append("dataset/" + str(file) + ".jpg")

files = new_files
# for r, d, f in os.walk("dataset/"):
#     for file in f:
#         if ('.jpg' in file):
#             exact_path = r + file
#             files.append(exact_path)


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

#restore the built model
t = AnnoyIndex(embedding_size, 'euclidean')
t.load('result.ann')

idx = 60
k = 2
tic = time.time()
neighbors = t.get_nns_by_item(idx, k)
toc = time.time()
print("Time to find %d nearest neighbors: %f" % (k, toc - tic))
print(representations[neighbors[0]][0], " is highly correlated with "
   ,representations[neighbors[1]][0])

# for i in neighbors:
#     print(i)
#     file_name = representations[i][0]
#     print(i, ": ", file_name)

#     functions.preprocess_face(file_name)

#     cv2.imshow("face", file_name)