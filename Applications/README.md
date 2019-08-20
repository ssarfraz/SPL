# Applications
Here, we provide the code for several image translations problems described in our paper. The used architectures are based on the common ResNet-9block encoder-decoder and SRCNN. For training, these models apply SPL for output and fitting secondary input based on the application. The project is build upon Tensorflow estimators.
## Dependencies
* Python3 
  * tensorflow==1.12.0
  * scipy==1.2.1
  * numpy==1.16.0
  * pillow==6.0.0
  * opencv-python==4.1.0.25
  * imageio==2.5.0

## Dataset Preparations
### General Image Translations
The trainer/predictor expects for the 'data' input argument a directory path containing the sub-directories:
* train
* val
* test

The images in these folders should be a horizontal concatenation of input and target images (each resized to 256x256), as can be seen below. Furthermore, due to SPL being based on pixel-level comparisons it is advantageous to provide aligned inputs.

![dataset input example](../imgs/input_general.png) 


### Super Resolution
The trainer/predictor expects for the 'data' input argument a directory path containing the sub-directories:
+ For Training
    * DIV2K_train_HR
    * DIV2K_train_LR_unknown/X4
+ For Validation
    * DIV2K_valid_HR
    * DIV2K_valid_LR_unknown/X4

The names of an image in lower resolution should correspond to the same image in higher resolution.    

### Makeup Transfer 
Similar to general image translations. Further explained [here](https://github.com/ssarfraz/SPL/FCC_Dataset).

## Training
```bash
python3 trainer.py --output <output directory> --dataset-name <type of dataset for corresponding application: [makeup,hilo,img_trans]> --data <data directory> --model <type of model: [hires,img_translation, makeup]> --batch-size <quite self explanatory>
```
## Testing
```bash
 python3 predictor.py --model-dir <path of model to load (output directory used in train)> --dataset-name <type of dataset to load> --data <path of data to load> --batch-size <batch size> --model <type of model to use> --result_dir <directory in which results are to be stored>
```

## Tensorboard
```bash
python3 -m tensorboard.main --port [specify] --logdir [model directory]
```
In the browser type localhost:port to see outputs.
