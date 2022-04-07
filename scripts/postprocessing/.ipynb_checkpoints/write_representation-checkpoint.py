import torch
from scripts.data.SimClrDataset import SimClrDataset
from scripts.data.transforms import get_transforms
from scripts.data.transforms import get_default_transforms
from scripts.model.resnet_simclr import ResNetSimCLR
from scripts.model.train import parser
from torch.cuda.amp import autocast
import config as c
from tqdm import tqdm
import numpy as np

args = parser.parse_args()

transforms = get_default_transforms(args.image_size)
test_dataset = SimClrDataset('~/simclr/dataset/m_test.csv', transform=transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.workers, pin_memory=True, drop_last=False)

model = ResNetSimCLR(args.depth,
                     args.widen_factor,
                     args.dropout_rate,
                     args.representation_dim,
                     args.projection_dim,
                     args.num_channels)

checkpoint = torch.load("./runs/Apr06_20-03-22_ravg1001/checkpoint_0050.pth.tar")
model.load_state_dict(checkpoint['state_dict'])

features = []

for images in tqdm(test_loader):
    images = torch.cat(images, dim=0)
    with autocast(enabled=args.fp16_precision):
        features.append(model(images, projection_head=False).detach().numpy())
        
features = np.concatenate(features, axis=0)
with open(c.postprocessing_path + 'representation.npy', 'wb') as f:
    np.save(f, features)
        
    

