# vision
vision part

input: 'cimg.jpg' ----binary image already,
cimg    ---np.array ---cleaning part is labeled 1 while whitespace 0

Three method to build minimum bounding boxes:

1. use a line of different(changeable) length to scan the image and find boxes that suit some parts (no code)

2. cluster and label pixels with constrains on:   --python/image_processing.py 
(1) the change of centre of mass rate by gradually increase pixels   ----return list seperating pixel index list
(2) optimization: geometry centre of a square box to be as close as the mass of centre (pixel intensity)
(3) set boxes value (number(default=3), size). ----visualization with MATLAB

3.Kmeans (Unsupervised clustering).    ----- machine/ ipython 
(1) regular Kmeans (Euclidean Distance) ----circle
(2) Guassian Mixture Model (probability model)  ----ellipse

