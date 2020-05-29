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
To use the baseline methods model, the file ```run_baseline.py``` needs to be opened and adapted. At the lines starting at 82 :
```
dir_raw = "raw_for_baseline/"
dir_a = "reducedDataSetPNG500/A/"
dir_b = "reducedDataSetPNG500/B/"
dir_c = "reducedDataSetPNG500/C/"
dir_d = "reducedDataSetPNG500/D/"
dir_e = "reducedDataSetPNG500/E/"
```
Change the ```"raw_for_baseline/"``` part for your folder with the raw images, and all other folders by your artists folders.
Finally, at the bottom, change the letter "D" by the artist for whom you want results.
```
#running + saving of test
for subdir, dirs, files in os.walk(dir_raw):
    for file in files:
        print(file)
        test_plot2(file, "D")
```

## CycleGAN
CycleGAN works in two phases: first a train phase, and then a test phase. The code and the different options available are best explained directly in the cycleGAN subfolder. 

To create a dataset that is usable by cycleGAN, a root folder has to contain 4 subfolders: trainA, trainB, testA, testB. trainA and trainB contain the images that are used to train the model, respectively in the A image domain and the B image domain. Same for testA and testB. 

The exact command ran to get the best results presented in the report is 

```python
python train.py --dataroot <dataset root folder> --name <experiment_nam> --model cycle_gan --gpu_ids <0, 1, 2, 3> --verbose --batch_size 4 --num_threads 8 --dataset_mode unaligned --save_epoch_freq 25 --norm instance --preprocess resize_and_crop --load_size 286 --crop_size 256 --gan_mode vanilla
```

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
To compute the Fréchet Inception Distance, run 
```
./fid_score.py <folder1> <folder2>
```
Where <folder1> and <folder2> contain the two sets of images that are going to be compared. The FID will be the distance between the two. Optional argument is *--gpu x*, with x being the gpu number that the computation is going to be ran on. If this argument is not provided, the computation will run on CPU. 

To achieve this from the results of a run of the ```test.py``` script from cycleGAN, we have created the *reorganise.py* script. Is it run as 
```
python reorganise.py <folder>
```
Where <folder> is the <experiment_name>/test_latest/images folder, that contains the output of the *test.py* script run beforehand. <experiment_name> is the name given in *test.py*

This will create subfolders with copies of the images, categorised as real_A, real_B, fake_A, fake_B. 
We then usually called 
```
run ./fid_score.py <experiment_name>/test_latest/images/fake_B <experiment_name>/test_latest/images/real_B
```
To get the distance between the generated B images and the ground truth B images, ie the retouched images. 
