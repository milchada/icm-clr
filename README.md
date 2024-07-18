# Unsupervised Contrastive Learning of Simulated and Real Galaxy+ Images #
Contrastive learning is an unsupervised machine learning technique with powerful applications in computer vision and data discovery. In essence, a Convolutional Neural Network (CNN) is fed sets of images of the same object - in different viewing angles, colours, crop and zoom levels, for example - called positive sampleas, and images of other objects, called negative samples. It is then told to find a representation that minimises the distance between the positive sampes while maximising the distance to the negatives. In other words, it learns that the first set all represent the same object, in contrast to the rest. This mimics the way humans gain object permanence, and can tell that a chair from any angle or distance, viewed through any filtered lens, is still a chair; it also allows us to tell that two objects are different from each other, even if we do not know what one or either of them is. 

This project builds off of the `simclr` package built by the team at Google [https://github.com/google-research/simclr]. We are interested in learning from large catalogues of images of galaxies and conglomerations thereof, namely galaxy groups and clusters. These are all highly complex, non-linear systems with many moving and poorly understood parts, resulting in a rich diversity of observable properties. Contrastive learning can be used on images of galaxies and galaxy groups to:
- Cluster the images into categories in latent, or embedding, spaces of parameters that are not obvious summary statistics. Analysing these clusters aids data discovery, by highlighting patterns that may not be immediately obvious in a very large and heterogeneous dataset.
- Find analogues to observed systems. Once the model has learned to sort images in the representation space, it can also place a new input image in this representation space and find the nearest neighbours. This assumes, of course, that the image is inherently similar to the training sample - for example, that we are comparing images of stellar light in the same colours, or of the X-ray emitting gas in the same energy bands, with similar resolution and exposure times. 
- Assess the realism of simulations of galaxy evolution. Each simulated sample of galaxies/galaxy groups forms a certain shape, or distribution, in the embedding/representation space. Models that are effectively similar will produce populations that have similar distributions. If a model is wildly different from reality, on the other hand, its representative distribution will have a very different shape from the observed sample. 
- Identifying physical drivers of observed features. Once objects have been sorted in the representation space, we can explore the distribution of various physical properties throughout this space. While not necessary and sufficient proof of causality, this can inspire and/or validate theoretical models of the causal mechanisms at play.

## How to customize to your dataset ##
### 1. Modify config.py
`illustris_path`: path to directory with simulation catalogues
`label_dict`: for all the `FIELDS` in `params.yaml` (see below), add a human-friendly label that will be used in plots.

### 2. Modify params.yaml
**neptune parameters**:

These allow you to log the CLR training process on Neptune.ai. Optional but very helpful.

**extract**:
- `DATASETS`: Name the dataset that you will be running CLR on. This is used as a key in `extractors.py` and `TNG/Subhalos.py`. The CLR can be run simultaneously on multiple datasets - for example, one can be images from TNG50, another from TNG100, another from an observed survey like HSC. Name each DATASET on a separate row. Each of the following parameters should then be specified the same number of times, i.e. if you have three datasets, you should have three rows each for FIELDS, LOAD, and FRACTION.
- `FIELDS`: These are the properties that will be pulled from the simulation group catalogues and appended to each image. After the images have been clustered in the representation space, you can see how whether the ordering correlates with these different labels.
- `LOAD`: True if the input DataFrames have not been compiled yet, i.e. if `dataset/` is empty.
- `FRACTION`: What fraction of the available images to use. This is especially helpful if you are training multiple datasets with inherently very different sizes.
- `IMAGE_SIZE`: 
- `NUM_PROJECTIONS`: How many angles is the same galaxuy/galaxy cluster viewed from? The DataFrames will have to have this many rows per object, with each row pointing to a different image but carrying the same labels for the intrinsic properties in FIELDS.
- `FILTERS`: Your images must be FITS files with one HDU per filter that you want to include in your training. FILTERS are the names of these HDUs. E.g. if your FILTERS are ['G','R','I'], and you open one of your image files as `f = astropy.io.fits.open("image.fits")`, then `f['G']` should be the G-band image, and likewise for 'R' and 'I'.
- `PETRO_RADIUS_FACTOR`: Only used if you are working with stellar photometry. In this case, images are cropped using the `PetrosianCropper` in `scripts/preprocessing/croppers.py`. You can modify `cropper.py` to create a new object that crops in multiples of some other characteristic radius, like $R_{200c}$.
- `USE_CACHE`: True if dataset has already been loaded but you are updating *only* the images.
- `NUM_WORKERS`: To parallelize the script.
- `SIZE_CHUNKS`: How many images are loaded in per batch of CLR training.

**prepare:**
`SETS`: for each dataset in DATASETS, specify a training/validation/testing split
`MATCHING_SOURCE_SETS`: if you have more than one dataset, and you want to create subsamples that are matched in certain `FIELDS`, name here the reference `DATASET` to which the others should be matched
`MATCHING_TARGET_SETS`: which `DATASET`s should be matched to `MATCHING_SOURCE_SET`.
`MATCHING_FIELDS`: what `FIELDS` must be matched between the `DATASET`s.
`MATCHING_UNCERTAINTIES`: margin of error in each `FIELD` being matched above.
 **The next two parameters are not used in the CLR, but only in the invertible neural network (INN)**. 
 `OBSERVABLES`: which of the `FIELDS` are directly observable quantities
 `UNOBSERVABLES`: which of the `FIELDS` must be inferred by the INN.

**data**
`VALID_RANGE`: a list of tuples. for each parameter in `UNOBSERVABLES`, add a minimum and maximum range **Actually I think this should be eliminated.**
`FILTERS`: **Why would this be different from FILTERS in extract above?**
`IMAGE_SIZE`: **Again, why is this different from above?**

### 3. Create training images
- The training images must be in `dataset_raw/DATASET_NAME/images/`. Parsing the image path currently assumes the following naming convention: `snap_{snapNum}-{haloID}_{projectionNum}.fits`. If you choose a different naming convention, you have to modify `TNGMockExtractor._split_filenames()` such that `snapnum, sub_ids` and `projections` are output for every image file in your training set.
- 
### 4. Create a dataframe compiling image paths and intrinsic values from the simulation 
The `FIELDS` specified above in `params.yaml` are read in from a simulation catalog, whose path is specified in `config.py`. `extractors.py` contains DataExtractor objects that creates Pandas dataframes with one row for each training image, with columns for path to the image and the properties of the galaxy/galaxy cluster in it. 
`TNG/Subhalos.py` defines Subhalo objects that read in TNG data, such as SubFind, FoF and BH particle properties. The base class `Subhalo` only relies on the existence of FoF and SubFind catalogues; subclasses like SubhaloStellar assume the existence of postprocessing catalogues for the observed photometry, morphology, and galaxy merger history. You can define your own subclass, pointing to specific quantities that you want to compute that are not in the default catalogues.
**Edit extractors.py and TNG/Subhalos.py to meet your needs.**
Then run `python -m scripts.preprocessing.extract.py`.

### 5. what does prepare.py do?
