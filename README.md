# Brain Tumor Detection Using VGG19

This project focuses on detecting the presence of brain tumors in MRI images using deep learning techniques. A Convolutional Neural Network (CNN) based on **VGG19** architecture is used for binary classification — detecting whether a tumor is **present** or **not**.

## Project Overview

- **Type**: Binary Classification
- **Input**: Brain MRI Images
- **Output**: Tumor / No Tumor
- **Model**: Pretrained VGG19 (Transfer Learning)
- **Dataset**: [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/jjprotube/brain-mri-images-for-brain-tumor-detection)

## Tech Stack

- Python
- Jupyter Notebook
- TensorFlow / Keras
- OpenCV
- NumPy / Matplotlib

## Dataset

The dataset contains two categories of brain MRI scans:

- **Yes** – Images with brain tumors
- **No** – Images without tumors

Download the dataset from Kaggle:  
🔗 [https://www.kaggle.com/datasets/jjprotube/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/jjprotube/brain-mri-images-for-brain-tumor-detection)

## Model Architecture

- **Base Model**: VGG19 (pretrained on ImageNet)
- **Customization**: Top layers replaced with custom fully connected layers
- **Activation**: ReLU & Softmax
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

## Results

| Metric       | Value      |
|--------------|------------|
| Accuracy     | ~80%       |
| Loss         | Low and stable after training |
| Visuals      | Training/Validation curves plotted |

>  Note: Exact values may vary depending on train/validation split.

## How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/PiyushG1816/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
