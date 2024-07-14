# Time Series Classification for PTB-XL Dataset

## Overview
This project focuses on the classification of electrocardiogram (ECG) signals from the PTB-XL dataset using advanced deep learning models such as LSTM, Inception1d, ResNet1d (Wang), and XResNet1d101. The goal is to accurately diagnose various cardiac conditions based on the ECG data.

## Dataset
The PTB-XL dataset is a large dataset of 21837 clinical 12-lead ECGs from 18885 patients of different age groups and genders. It includes a wide range of diagnostic labels, covering diagnostic and form subclasses as well as stethoscope recordings. The dataset is available on PhysioNet.

### Accessing the Dataset
To access the PTB-XL dataset, you will need to:
1. Visit the [PhysioNet PTB-XL page](https://physionet.org/content/ptb-xl/1.0.1/).
2. Follow the instructions to download the dataset.

## Requirements
- Python 3.8
- Conda

## Installation
1. **Set up a Conda environment:**
   ```bash
   conda create -n myenv python=3.8
   conda activate myenv
   ```
2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
3. **Download the PTB-XL dataset:**

    Visit the [PhysioNet PTB-XL page][https://physionet.org/content/ptb-xl/1.0.1/].

    Follow the instructions to download the dataset.

    Extract the downloaded files and place them in the ptbxl directory within your project.


**Usage**

To train the models, run the following command:
```
python3 main.py
```
**Output**

Model checkpoints, losses will be saved in the ```output``` folder.