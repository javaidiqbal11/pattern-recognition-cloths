# Cloths Pattern Recognition 

## Project Overview
This project focuses on cloth pattern recognition using deep learning approaches. The goal is to build a model that can classify different clothing patterns, such as stripes, polka dots, plaid, floral, and more. By leveraging deep learning techniques, particularly Convolutional Neural Networks (CNNs), we aim to achieve high accuracy in detecting and categorizing various cloth patterns. This application can be integrated into fashion recommendation systems, inventory management, or even fabric production processes.

### Setup
Python 3.10.10

**Install packages**
```shell
pip install -r requirements.txt
```

**CNN MOdel**
Train cnn model
```shell
python cnn.py
```
**Kaggle Notebooks**
There are two notebooks available to run the code and check output.
1. fabric-cnn.ipynb 
2. fabric-effiecientnet.ipynb


**ResNet**
Train cnn model
```shell
python resnet.py
```

**GCNN**
Check final outcomes 
```shell
python gcnn.py
```


## Future Improvements
Enhance dataset: Use a more diverse dataset with more patterns and variations in lighting, orientation, and fabric types.
Fine-tune the model: Apply transfer learning with a pre-trained model such as ResNet, VGG, or Inception for better performance.
Real-time detection: Implement real-time pattern recognition using a camera feed.
Mobile integration: Convert the model to run on mobile devices using TensorFlow Lite or ONNX.
