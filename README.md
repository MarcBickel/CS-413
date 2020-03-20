# CS-413 Project
## Predicting photographers retouching with Deep learning

In this project you will use a dataset of professional photographers' image manipulations from [2] and we ask you to build a machine learning system that learns experts' edits from the dataset. This is useful when automatically generating similar photographic manipulations based on the examples in a set of manipulated images. This is a well studied research area where you can find both traditional methods, e.g., [2], as well as recent deep learning based methods, e.g., [8].

[2]  V. Bychkovsky, S. Paris, E. Chan, and F. Durand.  Learning photographic global tonal adjustment with a database of input / output image pairs. In CVPR, 2011.  
[8] Y. Hu, H. He, C. Xu, B. Wang, and S. Lin.  Exposure:  A white-box photo post-processing framework.  InSIGGRAPH, 2018.

### Unchronological timeline
#### 20.03.2020
This week we worked on the baseline methods where we implemented our first Jupyter notebook. Rescaling is easily done with methods from OpenCV library.  
Then after reading the RAW image and one of the expert modification, we computed their CIE Lab color instead of the standard RGB one and we could compare both luminance. We are also working a bit on an idea of a starGAN.

#### Questions

#### 13.03.2020
CoronaVirus isolated us so we didn't have a personnal meeting. Instead we shared our progress by messages and got feedback.
This week we have looked at multiples things linked to our project.
1. We found a library called RawPy that allows us to work on the raw images that are in the database we are working on.
2. As suggested last week, we compared raw images dimensions to the edited one as output. Sadly they do not seem to match.
3. We are trying to implement as baseline methods the one found in the paper [2]
4. We read about StarGAN as suggested and started a simple implementation to see its feasibility.

#### Questions
1. As images do not seem to be paired, could it be possible to work on them after some rescaling ?
2. What exactly are the baseline methods and the features we're supposed to work on ?

#### Feedback
1. Rescaling by interpolation first then later on work on patches of the image instead of the smaller one
2. Mapping luminance from input to transfered one

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

##### Feedback
1. Implement the baseline methods
2. Find out if there's a pixel correspondance between the RAW images and the edited outputs  
If not, either we can "modify" edited images to create the paired datasets or we cannot use any paired GAN network
3. First plan is pix2pix if databases are paired
4. Else we will work on cycleGAN

- Read about StartGAN
- Even if paired cycleGAN could yield good results, though it requires more training
- Maybe try two approaches and compare them at the end

#### 25.02.2020 Meeting
First meeting and basic explanation with the TAs.
TAs said we could have two approches : 
- CycleGAN-based : unsupervised learning, using cycleGAN as base, and maybe then improve the model and the performances. 
- LUT-based : understand the preferred 3D transformations of the artist. Harder to pu together, but training time would be reduced. 


For the first week, we should choose an approach, and based on it, try to select few images and try to train on them (overfit as well).
