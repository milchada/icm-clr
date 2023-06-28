import torch
import scripts.data.data as data
from scripts.model.load import load_resnet_model
from torch.cuda.amp import autocast
import config as c
from tqdm import tqdm
import numpy as np
import os

def write_representation(in_path, out_path):
    
    #Prepare data
    test_loader = data.get_test_loader(batch_size=128, labels=False, augmentation=None, n_views=1, shuffle=False, drop_last=False)

    #Load model
    model = load_resnet_model(in_path)
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
    with open(out_path, 'wb') as f:
        np.save(f, features)
      
    
if __name__ == "__main__":
    #Write representations for all models in the model folder
    for in_path in os.listdir(path=c.model_path):
        print('Get representation for model ' + in_path)
        out_path = os.path.splitext(in_path)[0] + '.npy'
        write_representation(c.model_path + in_path, c.representation_path + out_path)
        
    