# Multiple Generative Adversarial Networks Analysis for Predicting Photographers’ Retouching

To be able to run CycleGAN, Pix2Pix and StarGAN, use:
*git submodule update --init --recursive*

## Pre-processing

To convert images from raw to the required pre-processed format, use:
*
```
python raw_to_adobe_png.py --origin <DNG images folder> --destination <Adobe PNG destination folder>
```

To resize images, use:
* For StarGAN and CycleGAN
	- ```
	python resize.py --origin <images source folder> --destination <resized images destination folder> --extention <extention of the source images> --targetSize <size the longest edge should be>
	```

* For Pix2Pix
	- python ...

## Baseline method


## CycleGAN

## Pix2Pix
To use Pix2Pix, are needed a folder <images> with two subfolder A/ and B/, which include each train/ and test/ folder.

A and B folders are the domains for images going from A to B, or B to A

To create dataset to use with Pix2Pix model :
```
python <cycleGAN_folder>/datasets/combine_A_and_B.py --fold_A <images>/A/ --fold_B <images>/B/ --fold_AB <combined_image_folder>
```

to train the models :

**For pix2pixA**
```
python <cycleGAN_folder>/train.py --dataroot <combined image folder> --name pix2pixA --gpu 0,1 --model pix2pix
```
**For pix2pixB**
```
python <cycleGAN_folder>/train.py --dataroot <combined_image_folder> --name pix2pixB --gpu 0,1 --n_epochs 500 --n_epochs_decay 500 --preprocess resize_and_crop --save_epoch_freq 20 --save_latest_freq 20000 --model pix2pix --batch_size 64
```
**For pix2pixC**
```
python <cycleGAN_folder>/train.py --dataroot <combined_image_folder> --name pix2pixC --gpu 0,1 --n_epochs 200 --n_epochs_decay 300 --preprocess scale_width_and_crop --save_epoch_freq 20 --save_latest_freq 20000 --model pix2pix --batch_size 64
```
**For pix2pixD**
```
python <cycleGAN_folder>/train.py --dataroot <combined_image_folder> --name pix2pixD --gpu 0,1 --n_epochs 200 --n_epochs_decay 300 --preprocess scale_width_and_crop --save_epoch_freq 20 --save_latest_freq 20000 --model pix2pix --batch_size 64
```

To test the models :
```
python <cycleGAN_folder>/test.py --dataroot <combined_image_folder> --name pix2pix{A, B, C, D} --model pix2pix --num_test 1000 --gpu 0,1 --preprocess {scale_width, resize}_and_crop
```

## StarGAN

### Dependencies

Dependencies can be found [here](https://github.com/yunjey/StarGAN)

### Prepare the Dataset

To conform to the dataset structure required by StarGAN script, structure the data as described [here](https://github.com/yunjey/StarGAN/blob/master/jpg/RaFD.md ).

### Training and Testing

To run the training of the network:
```
python main.py --mode train --dataset RaFD --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
               --c_dim LABEL_DIM --rafd_image_dir TRAIN_IMG_DIR \
               --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
               --model_save_dir stargan_custom/models --result_dir stargan_custom/results
```

To test the model:
```
python main.py --mode test --dataset RaFD --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
               --c_dim LABEL_DIM --rafd_image_dir TEST_IMG_DIR \
               --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
               --model_save_dir stargan_custom/models --result_dir stargan_custom/results
```


## FID
