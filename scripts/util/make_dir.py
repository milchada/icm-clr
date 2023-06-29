import os

def make_dir(path):
    path = os.path.split(path)[0]
    os.makedirs(path, exist_ok=True)