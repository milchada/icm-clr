import illustris_python as il
import glob, os
import numpy as np
import pandas as pd
from scripts.util.str2None import str2None
from scripts.preprocessing.TNG.Subhalos import Subhalos

import yaml

params = yaml.safe_load(open('params.yaml'))
extract_params = params['extract']
FIELDS = [str2None(i) for i in extract_params["FIELDS"]]
MIN_MASS = float(extract_params["MIN_MASS"])
MAX_MASS = float(extract_params["MAX_MASS"])
DATASETS = extract_params["DATASETS"]
NUM_PROJECTIONS = extract_params['NUM_PROJECTIONS']

simulation = '/virgotng/mpia/TNG-Cluster/'
image_path = 'dataset_raw/Xray-TNG-Cluster/images/'

files = glob.glob(image_path+'*_halo*fits')
snaps = np.array([file.split('snap')[1].split('_')[0] for file in files]).astype(int)
snaps.sort()


def get_labels(name=None):
    for snap in np.unique(snaps):
        out = []
        print(snap)
        subfiles = [f for f in files if 'snap%d' % snap in f]
        hnum = np.array([file.split('_halo')[1].split('_')[0] for file in subfiles]).astype(int)
        firstsub = il.groupcat.loadHalos(simulation+'TNG-Cluster/output/', snap, fields = ['GroupFirstSub'])[hnum]
        print("FirstSub IDs collected")
        halos = Subhalos(firstsub, snap, 'TNG-Cluster', simulation)
        labels = []
        for field in FIELDS[0]:
            field_values = getattr(halos, field)
            labels.append(field_values)
            print(field, 'logged')
            print(np.array(labels).shape)
        labels.append(hnum)
        labels.append(np.array([snap]*len(hnum)))
        labels.append(firstsub)
        labels.append(np.array([0] * len(hnum)))
        
        for projection in range(NUM_PROJECTIONS):
            labels[-1] = np.array([projection]*len(hnum))
            out.append(np.transpose(labels))
            print(np.array(out).shape)

        dfnew = pd.DataFrame(np.concatenate(out), columns=FIELDS[0]+['halo_num', 'snapshot_id', 'subhalo_id', 'projection'])
        if snap == snaps.min():
            df = dfnew
        else:
            df = pd.concat((df, dfnew))
        if name:
            df.to_csv(name, index=False)
    return df.drop_duplicates()

def split_filenames(filelist):
        #~/simclr-gas/dataset_raw/Xray-TNG-Cluster/images/snap59_halo110011010_1.fits
        splitlist = list(map(lambda x: os.path.split(x)[1], filelist))
        # snap59_halo110011010_1.fits
        snapnums = list(map(lambda x: x.split("snap")[1].split("_")[0], splitlist))
        projections = list(map(lambda x: x.split("_")[2].split('.')[0], splitlist))
        halo_ids  = list(map(lambda x: x.split("halo")[1].split("_")[0], splitlist))

        snapnums = np.array(snapnums, dtype=np.int32)
        halo_ids  = np.array(halo_ids, dtype=np.int32)
        projections  = np.array(projections, dtype=np.int32)
        
        return snapnums, halo_ids, projections

def add_image_path(df):
    '''Add the path to the respective image for each entry in df. Multiply entrys if there are multiple images for the same galaxy and delete if there is no image'''

    snapshot_ids = df["snapshot_id"].to_numpy(dtype=np.int32)
    halo_ids = df["halo_num"].to_numpy(dtype=np.int32)
    projection_ids = df["projection"].to_numpy(dtype=np.int32)

    filelist = glob.glob(image_path + '**/*_halo*.fits', recursive=True) 

    if len(filelist) == 0:
        logger.info("No images available, extract images first!")

    #Get snapshot and id
    snapnums, halonums, projections = split_filenames(filelist)

    logger.info("Assign Images to Labels")
    #Match the images with the data read from the csv (contained in df)
    origin = df.to_numpy()
    target = []
    mask = [] #Mask to sort out images with not avail data

    #Loop over all images
    for j, (snap, i, p, filepath) in enumerate(zip(snapnums, halonums, projections, filelist)):
        #Get matched df index for the image
        index = np.argwhere(np.logical_and(np.logical_and(snapshot_ids==snap, halo_ids==i), projection_ids==p))
        assert len(index)<=1, ("Multiple Data for one Image", index)

        #Check if there is data in df available for the given image
        if len(index) == 1:
            index = index[0]
            target.append(origin[index])
            mask.append(True)
        else:
            mask.append(False)

    logger.info(str(np.sum(mask)) + " images assigned to simulation data.")
    logger.info(str(np.sum(np.logical_not(mask))) + " images dropped.")
    df_matched = pd.DataFrame(np.array(target)[:,0,:], columns=df.columns)
    df_matched['image_path'] = np.array(filelist)[mask]
    df_matched['projection'] = np.array(projections)[mask]

    return df_matched

if __name__ == "__main__":
    df = get_labels()
    df = add_image_path(df)
    df.to_csv(image_path.replace('images/','label.csv', index=False)

#add more labels