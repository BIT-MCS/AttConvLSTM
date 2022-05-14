# AttConvLSTM
Additional materials for paper "Modeling Citywide Crowd Flows using Attentive Convolutional LSTM" accepted by ICDE 2021.
## :page_facing_up: Description
We present a novel deep model for this task, called "AttConvLSTM", which leverages a convolutional LSTM (ConvLSTM), Convolutional Neural Networks (CNNs) along with an attention mechanism, where ConvLSTM keeps spatial information as intact as possible during sequential analysis, and the attention mechanism can focus important crowd flow variations which cannot be identified by the recurrent module.
## :wrench: Dependencies
- Python == 3.5 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [Tensorflow == 1.4.1](http://jetware.io/versions/tensorflow:1.4.1)
- NVIDIA GPU (NVIDIA GTX TITAN XP) + [CUDA 10](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-MCS/AttConvLSTM.git
    cd AttConvLSTM
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```
## :zap: Quick Inference

Get the usage information of the project
```bash
cd code/AttConvLSTM/experiment/att_conv_lstm2/
python train.py -h
```
Then the usage information will be shown as following
```
usage: train.py [-h] PATH SEQ_LENGTH DATA_WIDTH
```
positional arguments:
  PATH     The directory of saving model
  SEQ_LENGTH  Total sequence length
  DATA_WIDTH         The resolution of data
 
optional arguments:
  -h, --help   show this help message and exit

## :computer: Training

We provide complete training codes for AttConvLSTM.<br>
You could adapt it to your own needs.

1. You can modify the config files 
[AttConvLSTM/code/AttConvLSTM/experiment/att_conv_lstm2/conf.py](https://github.com/BIT-MCS/AttConvLSTM/code/AttConvLSTM/experiment/att_conv_lstm2/conf.py) 
For example, you can set the batch size and training mode by modifying these lines
	```
	[4] 'BATCH' : 32,
[32]    'SAVE_MODEL' : True,
[33]    'LOAD_MODEL' : False,
[34]    'IS_TEST' : False,
	```
2. Training

	```
	python train.py 
	```

## :checkered_flag: Testing
1. Before testing, you should modify the file [AttConvLSTM/code/AttConvLSTM/experiment/att_conv_lstm2/conf.py](https://github.com/BIT-MCS/AttConvLSTM/code/AttConvLSTM/experiment/att_conv_lstm2/conf.py) as:
	```
    [32]    'SAVE_MODEL' : False,
    [33]    'LOAD_MODEL' : True,
    [34]    'IS_TEST' : True,
	```
2. Testing
	```
	python train.py
	```
## :scroll: Acknowledgement

Corresponding author: Chi Harold Liu.

## :e-mail: Contact

If you have any question, please email `363317018@qq.com`.
