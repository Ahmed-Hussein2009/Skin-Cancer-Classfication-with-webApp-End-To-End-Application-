# Skin Cancer Classfication using Deep Learning(WebApp-End-To-End-Application)

## [:film_projector: Presentation Link](https://github.com/Ahmed-Hussein2009/Skin-Cancer-Classfication-with-webApp-End-To-End-Application-/blob/main/presentation%20%5BAutosaved%5D.pptx) | [:pencil2:How to use the tool](https://github.com/Ahmed-Hussein2009/Skin-Cancer-Classfication-with-webApp-End-To-End-Application-/blob/main/final-%20documentation%20(1).docx)



https://user-images.githubusercontent.com/33674887/151664832-599753fb-5119-487c-ab36-afee990a3eee.mp4

# Abstract
In cancer, there are over 200 different forms. Out of 200, melanoma is the deadliest form of skin cancer. The diagnostic procedure for melanoma starts with clinical screening, followed by dermoscopic analysis and histopathological examination. Melanoma skin cancer is highly curable if it gets identified at the early stages. The first step of Melanoma skin cancer diagnosis is to conduct a visual examination of the skin's affected area. Dermatologists take the dermatoscopic images of the skin lesions by the high-speed camera, which have an accuracy of 65-80% in the melanoma diagnosis without any additional technical support. With further visual examination by cancer treatment specialists and dermatoscopic images, the overall prediction rate of melanoma diagnosis raised to 75-84% accuracy. The project aims to build an automated classification system based on image processing techniques to classify skin cancer using skin lesions images.

# Introduction and Background
Among all the skin cancer type, melanoma is the least common skin cancer, but it is responsible for **75%** of death [SIIM-ISIC Melanoma Classification, 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification). Being a less common skin cancer type but is spread very quickly to other body parts if not diagnosed early. The **International Skin Imaging Collaboration (ISIC)** is facilitating skin images to reduce melanoma mortality. Melanoma can be cured if diagnosed and treated in the early stages. Digital skin lesion images can be used to make a teledermatology automated diagnosis system that can support clinical decision.

Currently, deep learning has revolutionised the future as it can solve complex problems. The motivation is to develop a solution that can help dermatologists better support their diagnostic accuracy by ensembling contextual images and patient-level information, reducing the variance of predictions from the model.

## The problem we tried to solve
*tool that can tell doctors and lab technologists the three highest probability diagnoses for a given skin lesion. It could help them quickly identify high risk patients and speed up their workflow. The app will produce a result in less than 3 seconds. To ensure privacy, user submitted images are pre-processed and analyzed locally and are never uploaded to an external server.

This app is powered by Artifical Intelligence. My goal for this project was to build an end to end solution - starting with model creation and ending with a live web app. Users are able to submit a picture of a skin lesion and get an instant prediction.

The app is able to classify 7 types of skin lesions as described in this paper:

The HAM10000 Dataset: A Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions
https://arxiv.org/abs/1803.10417.*

## Motivation
The overarching goal is to support the efforts to reduce the death caused by skin cancer. The primary motivation that drives the project is to use the advanced image classification technology for the well-being of the people. Computer vision has made good progress in machine learning and deep learning that are scalable across domains. With the help of this project, we want to reduce the gap between diagnosing and treatment. Successful completion of the project with higher precision on the dataset could better support the dermatological clinic work. The improved accuracy and efficiency of the model can aid to detect melanoma in the early stages and can help to reduce unnecessary biopsies.

## Application
We aim to make it accessible for everyone and leverage the existing model and improve the current system. To make it accessible to the public, we build an easy-to-use website. The user or dermatologist can upload the patient demographic information with the skin lesion image. With the image and patient demographic as input, the model will analyse the data and return the results within a split second. Keeping the broader demographic of people in the vision, we have also tried to develop the basic infographic page, which provides a generalised overview about melanoma and steps to use the online tool to get the results.

## [Data Augmentation](./Notebooks/Data%20Augumentation.ipynb)

In a small size dataset, image augmentation is required to avoid overfitting the training dataset. After data aggregation, we have around **46k images in the training set**. The dataset contains significant class imbalance, with most of the classes have an **"Unknown"** category (Table 2). We have defined our augmentation pipeline to deal with the class imbalance. The augmentation that helps to improve the prediction accuracy of the model is selected. The selected augmentation are as follows:
1. **Transpose**: A spatial level transformation that transposes image by swapping rows and columns.
2. **Flip**: A spatial level transformation that flip image either/both horizontally and/or vertically. Images are randomly flipped either horizontally or vertically to make the model more robust.
3. **Rotate**: A spatial level transformation that randomly turns images for uniform distribution. Random rotation allows the model to become invariant to the object orientation.
4. **RandomBrightness**: A pixel-level transformation that randomly changes the brightness of the image. As in real life, we do not have object under perfect lighting conditions and this augmentation help to mimic real-life scenarios.
5. **RandomContrast**: A pixel-level transformation that randomly changes the contrast of the input image. As in real life, we do not have object under perfect lighting conditions and this augmentation help to mimic real-life scenarios.
6. **MotionBlur**: A pixel-level transformation that applies motion blur using a random-sized kernel.
7. **MedianBlur**: A pixel-level transformation that blurs input image using a median filter.
8. **GaussianBlur**: A pixel-level transformation that blurs input image using a gaussian filter.
9. **GaussNoise**: A pixel-level transformation that applies Gaussian noise to the image. This augmentation will simulate the measurement noise while taking the images
10. **OpticalDistortion**: Optical distortion is also known as Lens error. It mimics the lens distortion effect.
11. **GridDistortion**: An image warping technique driven by mapping between equivalent families of curves or edges arranged in a grid structure.
12. **ElasticTransform**: A pixel-level transformation that divides the images into multiple grids and, based on edge displacement, the grid will be distorted. This transform helps the network to have a better understanding of edges while training.
13. **CLAHE**: A pixel-level transformation that applies Contrast Limited Adaptive Histogram Equalization to the input image. This augmentation improves the contrast of the images.
14. **HueSaturationValue**: A pixel-level transformation that randomly changes hue, saturation and value of the input image.
15. **ShiftScaleRotate**: A spatial level transformation that randomly applies affine transforms: translate, scale and rotate the input. The allow scale and rotate the image by certain angles
16. **Cutout**: A spatial level transformation that does a rectangular cut in the image. This transformation helps the network to focus on the different areas in the images.

Figure 5 illustrates the before and after augmented image. The augmentation is applied to only the training set while just normalising the validation and testing dataset.

![Data Augmentation](./readme_images/5.jpg)

*Figure 5 Training set augmentation.*

*After the data pre-processing and data augmentation, we have around 46,425 images in the training set, 11,606 images in the validation set and 10,875 images in the testing set. The training set is divided into an 80/20 ratio where 80% is used for training and 20% as a validation set.*

####  ***You can view more Augmented samples under [`./Data/Augmented Sample Images`](./Data/Augmented%20Sample%20Images/)***

# Overview of the Architecture
The project contains two flow diagrams. Figure 6 shows the model training pipeline, while Figure 7 shows the web UI flow. The first step after downloading the data is to clean and combine the data (Figure 6). The missing values in the patient demographic are imputed with the average values as the ratio of missing values is less than 5% in the overall dataset. The provided skin lesion images are of higher resolution, and it is not ideal for training the network on the high-resolution images (Figure 3 and 4). *In the data pre-processing steps, all images are cropped into 768x786 and 512x512 resolution to reduce random noise on the edges of the image.*

The data cleaning and pre-processing step are performed on all the dataset obtained from the 2020, 2019 and 2018 competition. Also, the image labels are reconciled and combined into a single training CSV file. The augmentation is performed on the fly during the model training process to reduce the storage space and improve efficiency. **During the model training part, **Nth** images are read from the training folder and augmentation is performed on the CPU while the EfficientNet is loaded in the GPU. Augmentation is performed on the CPU, and training on GPU help to reduce the training time (Figure 6).**

After each epoch, we check the validation accuracy of the model. If the validation accuracy does not increase after 15 epochs, the training process is stopped, and the best model weights are saved for the prediction (Figure 6). The prediction is performed on the test set, and results are stored in the CSV file. Along with the model weights, all diagnostic information for the model is stored locally.


# Conclusion

One of the deadliest cancer forms is melanoma, and the proportion of people getting affected by melanoma is increasing rapidly. To make the solution available to the public and dermatologists, we have successfully integrated the optimised model with our CAD system. 

The EfficientNet model is proved to be a better network for the skin cancer dataset. The network can generalise well on the dataset and have higher validation accuracy (Table 3 and Figure 22). Plus, the ensemble of the model helps to reduce model prediction error and biases. The model prediction error can be further reduced if the ensemble is more significant with varied configuration, as proposed in Table 4.

Along with optimising the training process, an equal amount of time is spent optimising the predictions. Based on the three core pillars of model serving, we have tick two of them: model size and latency. The last pillar (Prediction throughput) comes into account when the predictions are performed online over the internet. The prediction throughput measures how many predictions the system can perform in a given timeframe. The prediction throughput is beyond the project's scope but should be considered when deploying the model on the web. 

# References 

ISIC. (2018). Skin Lesion Analysis Towards Melanoma Detection. Retrieved March 20, 2021, from https://challenge2018.isic-archive.com/

ISIC. (2019). Skin Lesion Analysis Towards Melanoma Detection. Retrieved March 20, 2021, from https://challenge2019.isic-archive.com/

Mingxing, T., & Quoc, L. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. https://arxiv.org/abs/1905.11946

SIIM-ISIC Melanoma Classification. (2020). Identify melanoma in lesion images. Retrieved March 20, 2021, from https://www.kaggle.com/c/siim-isic-melanoma-classification

Tong, H., Zhi, Z., Hang, Z., Zhongyue, Z., Junyuan, X., Mu, L. (2018). Bag of Tricks for Image Classification with Convolutional Neural Networks. https://arxiv.org/abs/1812.01187

Qishen, H., Bo, L., Fuxu L. (2020). Identifying Melanoma Images using EfficientNet Ensemble: Winning Solution to the SIIM-ISIC Melanoma Classification Challenge. https://arxiv.org/abs/2010.05351


