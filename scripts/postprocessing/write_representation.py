import torch
import scripts.data.data as data
from scripts.model.load import load_resnet_model
from scripts.util.make_dir import make_dir
from torch.cuda.amp import autocast
import config as c
from tqdm import tqdm
import numpy as np
import os

def write_representation(in_path, out_path, params={}, obs_path = None, load_set='test'):
    
    #Prepare data
    if obs_path:
        test_loader = data.get_loader(batch_size=128, labels=False, augmentation=None, n_views=1, shuffle=False, drop_last=False, meta_path = image_path_csv, label_path=None)
    elif load_set == 'test':
        test_loader = data.get_test_loader(batch_size=128, labels=False, augmentation=None, n_views=1, shuffle=False, drop_last=False)
    elif load_set == 'train':
        test_loader = data.get_train_loader(batch_size=128, labels=False, augmentation=None, n_views=1, shuffle=False, drop_last=False)
    elif load_set == 'val':
        test_loader = data.get_val_loader(batch_size=128, labels=False, augmentation=None, n_views=1, shuffle=False, drop_last=False)

    #Load model
    model = load_resnet_model(in_path, params)
    model.eval()

    features_list = []

    #Get representations for all test galaxies
    for images in tqdm(test_loader):
        images = torch.cat(images, dim=0)
        images = images.to(c.device)

        with torch.no_grad(), autocast(enabled=True):
            features = model(images, projection_head=False).detach().cpu().numpy()
            features_list.append(features)

    features = np.concatenate(features_list, axis=0)

    #Save
    make_dir(out_path)
    with open(out_path, 'wb') as f:
        np.save(f, features)


if __name__ == "__main__":
    write_representation(c.resnet_path, c.representation_path)
    write_representation(c.resnet_path, c.representation_path.replace('.npy', '_train.npy'), load_set='train')
    # write_representation(c.resnet_path, c.representation_path.replace('.npy', '_val.npy'), load_set='val')    
    # write_representation(c.resnet_path, c.representation_path.replace('rep','obs_rep'), "obs_files.csv")
