
Project: Human Activity Recognition using Decision Trees

Introduction
Human Activity Recognition (HAR) involves identifying various activities performed by users based on data captured from sensors. In this project, we utilize the UCI-HAR dataset to implement a Decision Tree model for recognizing activities using smartphone accelerometer data.

Dataset
The UCI-HAR dataset consists of smartphone accelerometer and gyroscope data from 30 participants performing six activities while wearing a Samsung Galaxy S II smartphone on their waist. The activities include walking, walking downstairs, walking upstairs, sitting, standing, and laying.

Preprocessing
We perform the following steps for data preprocessing:

Organize accelerometer data using CombineScript.py.
Create a single dataset using MakeDataset.py.
Split the dataset into training, testing, and validation sets.
Focus on the first 10 seconds of activity.
Exploratory Data Analysis
Waveform Visualization
We plot the waveform for data from each activity class to observe differences or similarities between activities.

Static vs. Dynamic Activities
We analyze linear acceleration for each activity to determine if machine learning is necessary to differentiate between static (laying, sitting, standing) and dynamic activities (walking, walking downstairs, walking upstairs).

Decision Tree Implementation
We train a Decision Tree using the training set and report accuracy and a confusion matrix on the test set. We also experiment with varying tree depths to observe changes in accuracy.

Feature Engineering and Visualization
Principal Component Analysis (PCA)
We use PCA to compress acceleration time series into two features and visualize different activity classes. We repeat this process using features obtained from the TSFEL library.

Model Training with Features
We use features obtained from TSFEL to train a Decision Tree. We compare the accuracy and confusion matrix with the model trained on raw data. We also vary tree depths and compare accuracies.

Model Performance Analysis
We analyze if there are participants or activities where the model performance is poor and provide justifications.

Deployment and Testing
For deployment, we collect data using smartphone apps, ensuring the phone's consistent position and alignment. We collect at least 15 seconds of data per activity, trim edges, and report accuracy using both featurized and raw data. We train on the UCI dataset and test on the collected data, explaining the model's success or failure.
