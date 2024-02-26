# Air Quality Index Prediction

ADNNet: Attention-based deep neural network for Air Quality Index prediction


## Introduction

This repository provides the implementation of ADNNet for AQI prediction. The experiments have been performed datasets:  [**AQI Online Testing and Analysis Platform**](https://www.aqistudy.cn/).


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
### Ablation Test  
  
ADNNet_nosmoothing/smoothing.py and ADNNet_noBayesian/Bayesian.py includes ablation test implementation.   

  
```  
python pytorch_ablation_test.py  
```

### Features Sensitivity Analysis

```  
python pytorch_features sensitivity.py  
```

