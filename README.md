# Air Quality Index Prediction

ADNNet: Attention-based deep neural network for Air Quality Index prediction


## Introduction

This repository provides the implementation of ADNNet for AQI prediction. The experiments have been performed datasets:  [**AQI Online Testing and Analysis Platform**](https://www.aqistudy.cn/).

If you use this repo, please cite our paper:

'''
@article{wu2024adnnet,
  title={ADNNet: Attention-based deep neural network for Air Quality Index prediction},
  author={Wu, Xiankui and Gu, Xinyu and See, KW},
  journal={Expert Systems with Applications},
  pages={125128},
  year={2024},
  publisher={Elsevier}
}
'''

## Getting Started

### (1) Install requirements

#### Python packages
Tested with TensorFlow2.8+Keras2.8.0.

    pip install tensorflow
    pip install pandas sklearn scipy
    pip install plotly
    pip install jupyter notebook ipykernel jupyterlab

### (2) Download the datasets

Download the [AQI Dataset](https://www.aqistudy.cn/) and put its content in the directory `AQI-Attention-based-DNN/data/`





  
## Test With Different Models
  
ADNNet.ipynb and ADNNet_multisteps.ipynb represent the implementation of one-step prediction and multi-step prediction respectively. 
--baseline: LSTM, N-BEATS, Informer, Autoformer 
  

```  
ADNNet.ipynb --feature=0 --model=0  
```
 
```
ADNNet_multisteps.ipynb --feature=0 --model=0  
```


