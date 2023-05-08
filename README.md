# Captcha Extraction using PyTorch

This project is designed to extract text from captcha images using a Deep C-RNN model implemented in PyTorch. The model utilizes the CTCLoss (Connectionist temporal classification Loss) algorithm to convert the predicted output into the actual text. 

## Project URL
The code for this project can be found on [GitHub](https://github.com/ritikjain51/CaptchaExtractor-Pytorch).

## Requirements
- PyTorch
- NumPy
- Matplotlib
- Torchvision

## Usage
1. Clone the repository: `git clone https://github.com/ritikjain51/CaptchaExtractor-Pytorch.git`
2. Install the required packages: `pip install -r requirements.txt`
3. Run the `train.py` file to train the model on the captcha dataset.

## Dataset
The dataset used in this project can be found in the `data/` directory.

## Model Architecture
The model consists of a Deep C-RNN network that processes the input captcha image and produces a predicted output. The CTCLoss algorithm is used to convert the predicted output into the actual text.

### CTC Loss
CTC Loss is a cost function used in sequence recognition tasks, such as speech recognition or handwriting recognition. It allows for the prediction of variable-length sequences by comparing the output sequence with the target sequence without aligning them. In the case of this project, the CTCLoss algorithm is used to compare the predicted output of the CRNN with the actual text from the captcha image. 

### CRNNs
CRNNs (Convolutional Recurrent Neural Networks) are a type of neural network that combines convolutional layers, which are effective at extracting local features from an image, with recurrent layers, which are effective at modeling sequential data. In the case of this project, the CRNN takes the captcha image as input and processes it using convolutional layers to extract features, followed by recurrent layers to model the sequential nature of the captcha text. The output of the CRNN is then passed through a linear layer to produce the final prediction, which is fed into the CTCLoss algorithm. 

## Acknowledgements
- This project was inspired by the work on captcha recognition by Zhu et al. in their paper "A New Approach to Recognize Captcha with Convolutional Neural Network" (2017).
- The captcha dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/fournierp/captcha-version-2-images).
