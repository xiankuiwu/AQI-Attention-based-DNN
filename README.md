# Air Quality Index prediction

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


### (3) Data preprocessing  
  
feature_selection.py provide the implementation of feature extraction and selection



  
## Test With Different Models
  
feature_based_model_paper.py includes several model implementation with different features combination.  
--baseline: LSTM, B-BEATS, Informer, Autoformer 
  

```  
python feature_based_model_paper.py --feature=0 --model=0  
```  
  
### Proposed hybrid model with snaphsotensemble  
  
pytorch_hybrid_model_snapshot_train.py includes hybrid model implementation.   
Before running, need to generate the dataset via data_process_for_hybrid_model.py.  
  
```  
python pytorch_hybrid_model_snapshot_train.py  
```
