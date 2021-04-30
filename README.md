# one-shot-face-recognition
An implementation of a Siamese Convolutional network for  the task of One-shot facial recognition, as appears in the paper [Siamese Neural Networks for One-shot Image Recognition
](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf).  


## Dataset
The [LFW-a](https://talhassner.github.io/home/projects/lfwa/index.html) dataset contains pairs of photos either belonging to the same person (which we label 1) or to different people (which we label 0). The photos are in grayscale and of size 250 * 250.<br>

We use the  dataset with a predefined [train](http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt) \ [test](http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt) split.<br>
The train set contains 2200 pairs of images where 1100 are the same person and 1100 are not. In addition, 15% of the training set was used for validation in a stratified manner. The test set contains 1000 pairs of images where 500 are the same person and 500 are not. The test set does not contain photos of people that appeared in the train set.<br> 

