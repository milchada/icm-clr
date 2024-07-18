from PIL import Image
import pandas as pd
from astropy.io import fits 
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, csv_file, pred_labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pred_labels = pred_labels
        self.transform = transform

        #normalise the labels for training
        #this is for the training data so we will know the max and min to reverse this operation later

        data = pd.read_csv(csv_file)
        for label in pred_labels:
            lmax = max(abs(data[label]))
            if lmax > 20:
                data[label] = np.log10(data[label])
            data[label] -= min(data[label])
            data[label] /= max(data[label])
            # print(label, 'normalised')
        
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data["image_path"].iloc[idx]
        image = fits.getdata(img_name)[:, np.newaxis]
        image = image.byteswap().newbyteorder()
        # image = Image.fromarray(array.astype('uint8'))
        label = [self.data[label].iloc[idx] for label in self.pred_labels]
        for i in range(len(label)):
            if abs(label[i]) > 20:
                label[i] = np.log10(label[i])

        if self.transform:
            image = self.transform(image)

        return image, label
