# The Spatial Profile Loss
Here we show the implementations of the SP-Loss in Tensorflow and Pytorch mentioned in the paper [**Content and Colour Distillation for Learning Image Translations with the Spatial Profile Loss**](https://arxiv.org/pdf/1908.00274.pdf). 

## Tensorflow

Exemplary usage of the tensorflow loss implementation.


```python
import tensorflow as tf
import tensorflow_spl_loss as spl

# The SPL can be initialized with information on whether to use scaling for the RGB-YUV conversion. 
# As we have operated in a value range [-1,1], we scaled our inputs to [0,1] . 
# However, assuming you are using only values in the range of [0,1] (e.g. sigmoid outputs), 
# you can disable any scaling by specifying in the initialization: 
# spl.SPL(use_conversion=False)



SPL = spl.SPL()
input = tf.random_uniform(shape=(1,256,256,3))
generated = tf.random_uniform(shape=(1,256,256,3))
target = tf.random_uniform(shape=(1,256,256,3))

sess = tf.Session()

# Generally
# To calculate the complete SPL between two inputs

spl_value = sess.run(SPL(target,generated))

# For Style Transfer (see our makeup Style transfer) 
# Seperate calls are also possible

gpl_value = sess.run(SPL.GP(input,generated))
cpl_value = sess.run(SPL.CP(target,generated))

spl_value = cpl_value + gpl_value


```

## Pytorch

Exemplary usage of the Pytorch loss implementation.

```python
import torch
import pytorch_spl_loss as spl

# Gradient Profile Loss
GPL =  spl.GPLoss()

# Color Profile Loss
# You can define the desired color spaces in the initialization
# default is True for all
CPL =  spl.CPLoss(rgb=True,yuv=True,yuvgrad=True)

target    = torch.randn(1,3,256,256)
generated = torch.randn(1,3,256,256)

gpl_value = GPL(generated,target)
cpl_value = CPL(generated,target)

spl_value = gpl_value + cpl_value

```


## Citation
If you use this work or dataset, please cite:
```latex
@inproceedings{spl,
    author    = {M. Saquib Sarfraz, Constantin Seibold , Haroon Khalid and Rainer Stiefelhagen}, 
    title     = {Content and Colour Distillation for Learning Image Translations with the Spatial Profile Loss}, 
    booktitle = {Proceedings of the 30th British Machine Vision Conference (BMVC)},
    year  = {2019}
}

```
