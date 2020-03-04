# CS-413 Project
## Predicting photographers retouching with Deep learning

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
