# CS-413 Project
## Predicting photographers retouching with Deep learning

In this project you will use a dataset of professional photographers' image manipulations from [2] and we ask you to build a machine learning system that learns experts' edits from the dataset. This is useful when automatically generating similar photographic manipulations based on the examples in a set of manipulated images. This is a well studied research area where you can find both traditional methods, e.g., [2], as well as recent deep learning based methods, e.g., [8].

[2]  V. Bychkovsky, S. Paris, E. Chan, and F. Durand.  Learning photographic global tonal adjustment with a database of input / output image pairs. In CVPR, 2011.
[8] Y. Hu, H. He, C. Xu, B. Wang, and S. Lin.  Exposure:  A white-box photo post-processing framework.  InSIGGRAPH, 2018.

### Unchronological timeline
#### 06.03.2020 Meeting
We have been looking for ideas to start the project.
Either we base our model simply on the raw input images to edited output, or we base the model on some kind of GAN.
We implemented a small GAN for the MNIST dataset to test the principles.
After looking for GAN possibilities we plan on pursuing this idea and found multiple concepts:
1. CycleGan
2. Pix2Pix
3. DiscoGan
We plan on going for an like Pix2Pix Gan because of its paired implementation for images, and maybe use ideas from the CycleGan also.

##### Questions 
-- 

#### 25.02.2020 Meeting
First meeting and basic explanation with the TAs.
TAs said we could have two approches : 
- CycleGAN-based : unsupervised learning, using cycleGAN as base, and maybe then improve the model and the performances. 
- LUT-based : understand the preferred 3D transformations of the artist. Harder to pu together, but training time would be reduced. 


For the first week, we should choose an approach, and based on it, try to select few images and try to train on them (overfit as well).
