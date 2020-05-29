# Multiple Generative Adversarial Networks Analysis for Predicting Photographers’ Retouching

To be able to run CycleGAN, Pix2Pix and StarGAN, use:
*git submodule update --init --recursive*

## Pre-processing

To convert images from raw to the required pre-processed format, use:
*python raw_to_adobe_png.py --origin <DNG images folder> --destination <Adobe PNG destination folder>*

To resize images, use:
**For StarGAN and CycleGAN**
*python resize.py --origin <images source folder> --destination <resized images destination folder> --extention <extention of the source images> --targetSize <size the longest edge should be>

**For Pix2Pix**
python ...

## Baseline method


## CycleGAN

## Pix2Pix

## StarGAN

## FID