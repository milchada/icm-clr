### Unsupervised Contrastive Learning of Simulated and Real Galaxy+ Images ### 
Contrastive learning is an unsupervised machine learning technique with powerful applications in computer vision and data discovery. In essence, a Convolutional Neural Network (CNN) is fed sets of images of the same object - in different viewing angles, colours, crop and zoom levels, for example - called positive sampleas, and images of other objects, called negative samples. It is then told to find a representation that minimises the distance between the positive sampes while maximising the distance to the negatives. In other words, it learns that the first set all represent the same object, in contrast to the rest. This mimics the way humans gain object permanence, and can tell that a chair from any angle or distance, viewed through any filtered lens, is still a chair; it also allows us to tell that two objects are different from each other, even if we do not know what one or either of them is. 

This project builds off of the `simclr` package built by the team at (Google) [https://github.com/google-research/simclr]. We are interested in learning from large catalogues of images of galaxies and conglomerations thereof, namely galaxy groups and clusters. These are all highly complex, non-linear systems with many moving and poorly understood parts, resulting in a rich diversity of observable properties. Contrastive learning can be used on images of galaxies and galaxy groups to:
- Cluster the images into categories in latent, or embedding, spaces of parameters that are not obvious summary statistics. Analysing these clusters aids data discovery, by highlighting patterns that may not be immediately obvious in a very large and heterogeneous dataset.
- Find analogues to observed systems. Once the model has learned to sort images in the representation space, it can also place a new input image in this representation space and find the nearest neighbours. This assumes, of course, that the image is inherently similar to the training sample - for example, that we are comparing images of stellar light in the same colours, or of the X-ray emitting gas in the same energy bands, with similar resolution and exposure times. 
- Assess the realism of simulations of galaxy evolution. Each simulated sample of galaxies/galaxy groups forms a certain shape, or distribution, in the embedding/representation space. Models that are effectively similar will produce populations that have similar distributions. If a model is wildly different from reality, on the other hand, its representative distribution will have a very different shape from the observed sample. 
- Identifying physical drivers of observed features. Once objects have been sorted in the representation space, we can explore the distribution of various physical properties throughout this space. While not necessary and sufficient proof of causality, this can inspire and/or validate theoretical models of the causal mechanisms at play.