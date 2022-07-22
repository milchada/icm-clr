import torch
import scripts.data.data as data
from scripts.model.load import load_resnet_model
from torch.cuda.amp import autocast
import config as c
from tqdm import tqdm
import numpy as np

#Prepare data
test_loader = data.get_test_loader(batch_size=128, labels=False, augmentation=None, n_views=1, shuffle=False, drop_last=False)

#Load model
model = load_resnet_model()
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
with open(c.postprocessing_path + 'representation.npy', 'wb') as f:
    np.save(f, features)
        
    

