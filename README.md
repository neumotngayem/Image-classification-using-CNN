# Image classification using CNN

This is a part of a dataset in Competition 2 - Shopee Code League. The trainning include 10 image categories, around 1500 trainning images in each category. The 10 categories name as follows:

00: Maxi dress

01: Muslim dress

02: T-shirt

03: Hoodie shirt

04: Jean

05: Ring

06: Ear ring

07: Cap

08: Wallet

09: Bag

10: Phone case

All of the images are real images on Shopee platform - the leading e-commerce online shopping platform in Southeast Asia and Taiwan.

# Feature extraction

The feature of each image is extracted by using pre-trained model ResNet50, with weights are imagenet.

# CNN Model

After feature extracted by ResNet50, the model will go through a 1500 nodes hidden layer before go to the output layer. The nodes number can modify to achieve better accuracy.

# Steps to run

1. Run c2_train_feature_extraction.py

2. Run c2_test_feature_extraction.py

3. Run c2_training.py

4. Run c2_predict.py

# Accuracy

The trainning accuracy is reached at 0.9437 after 5 epochs.
![Training Accuracy](https://github.com/neumotngayem/Image-classification-using-CNN/blob/master/Trainning.png?raw=true)

The testing accuracy is reached at 0.81

![Testing Accuracy](https://github.com/neumotngayem/Image-classification-using-CNN/blob/master/Testing.png?raw=true)
