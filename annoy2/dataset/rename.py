import os
import uuid


# rename every file in a directory
def rename(path):
    for i, filename in enumerate(os.listdir(path)):
        if filename.endswith('.jpg'):
            print(i)
            id = str(uuid.uuid4())
            os.rename(os.path.join(path, filename), os.path.join(path, id + '.jpg'))

rename(".")