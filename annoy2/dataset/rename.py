import os

# rename every file in a directory
def rename(path):
    for i, filename in enumerate(os.listdir(path)):
        if filename.endswith('.jpg'):
            os.rename(os.path.join(path, filename), os.path.join(path, filename.replace('img', '')))

rename(".")