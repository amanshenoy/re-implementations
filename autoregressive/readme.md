# Autoregressive Models

The following sub-directory consists of implementations of a few popular autoregressive models, namely - an autoregressive RNN, MADE and Pixel-CNN. The implementation has been trained on Binary MNIST. 

Trained models have also been uploaded in `saved` directory, and sampling has been implemented as class methods. 

## Autoregressive RNN 

|<img src="https://github.com/amanshenoy/autoregressive/blob/master/demonstrations/ar_rnn_sample_1.png" alt="drawing" width="300"/>|<img src="https://github.com/amanshenoy/autoregressive/blob/master/demonstrations/ar_rnn_sample_2.png" alt="drawing" width="300"/> |<img src="https://github.com/amanshenoy/autoregressive/blob/master/demonstrations/ar_rnn_sample_3.png" alt="drawing" width="300"/> |
|:---:|:----:|:----:|
| Only pixel value as input | previous + inverted pixel appended | previous + location information appended | 

## Masked Auto-Encoder for Distribution Estimation (MADE)

|<img align='left' src="https://github.com/amanshenoy/autoregressive/blob/master/demonstrations/made_sample_1.png" alt="drawing"/>|<img align='center' src="https://github.com/amanshenoy/autoregressive/blob/master/demonstrations/made_sample_2.png" alt="drawing"/> |<img align='center' src="https://github.com/amanshenoy/autoregressive/blob/master/demonstrations/made_sample_3.png" alt="drawing" /> |<img align='right' src="https://github.com/amanshenoy/autoregressive/blob/master/demonstrations/made_sample_3.png" alt="drawing"/> |
|:---:|:----:|:----:|:----:|


## Class-Conditional PixelCNN

|<img src="https://github.com/amanshenoy/autoregressive/blob/master/demonstrations/pixel_cnn_6.png" alt="drawing" width="300"/>|<img src="https://github.com/amanshenoy/autoregressive/blob/master/demonstrations/pixel_cnn_7.png" alt="drawing" width="300"/> |<img src="https://github.com/amanshenoy/autoregressive/blob/master/demonstrations/pixel_cnn_8.png" alt="drawing" width="300"/> |
|:---:|:----:|:----:|
| Class Conditional for 6 | Class Conditional for 7 | Class Conditional for 8 | 

## Re-training and Sampling

The following is a training and sampling usage and `autoregressive_rnn.py` and `pixel_cnn.py` can be run in the exact same way. 

For re-training the models, while logging to tensorboard from directory `runs` and run number `run_num`

    >>> python made.py --train --epochs 10 --batch_size 128 --lr 5e-04 --save_dir \path\to\save --logdir runs --run_num 1
    
For sampling from the saved models 

    >>> python made.py --sample --num_samples 10 --logdir runs --run_num 1
