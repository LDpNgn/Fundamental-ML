# Mini-Project for Fundamentals of Machine Learning Course
![background](./materials/ai_wp.jpg)
This repository contains the code and data for a mini-project on facial expression recognition using machine learning algorithms.

## 📑 Project Policy
- Team: group should consist of 3-4 students.

    |No.| Student Name    | Student ID |
    | --------| -------- | ------- |
    |1|Huỳnh Lê Minh Thư|21280110|
    |2|Phạm Ngọc Phương Uyên|21280119|
    |3|Nguyễn Thị Lan Diệp|21280123|

- The submission deadline is strict: **11:59 PM** on **June 22nd, 2024**. Commits pushed after this deadline will not be considered.

## 📦 Project Structure

The repository is organized into the following directories:

- **/data**: This directory contains the facial expression dataset. You'll need to download the dataset and place it here before running the notebooks. (Download link provided below)
- **/notebooks**: This directory contains the Jupyter notebook ```EDA.ipynb```. This notebook guides you through exploratory data analysis (EDA) and classification tasks.

## ⚙️ Usage

This project is designed to be completed in the following steps:

1. **Fork the Project**: Click on the ```Fork``` button on the top right corner of this repository, this will create a copy of the repository in your own GitHub account. Complete the table at the top by entering your team member names.

2. **Download the Dataset**: Download the facial expression dataset from the following [link](https://mega.nz/file/foM2wDaa#GPGyspdUB2WV-fATL-ZvYj3i4FqgbVKyct413gxg3rE) and place it in the **/data** directory:

3. **Complete the Tasks**: Open the ```notebooks/EDA.ipynb``` notebook in your Jupyter Notebook environment. The notebook is designed to guide you through various tasks, including:
    
    1. Prerequisite
    2. Principle Component Analysis
    3. Image Classification
    4. Evaluating Classification Performance 

    Make sure to run all the code cells in the ```EDA.ipynb``` notebook and ensure they produce output before committing and pushing your changes.

5. **Commit and Push Your Changes**: Once you've completed the tasks outlined in the notebook, commit your changes to your local repository and push them to your forked repository on GitHub.


## Getting Started
### 1. Prerequisite  

#### Load dataset  
The data consists of grayscale images of faces, each measuring 48x48 pixels. The faces have been automatically aligned to be roughly centered and occupy a similar area within each image. The task is to categorize each face based on the emotion expressed, assigning it to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The dataset contains a total of 35,887 examples.  
![img_df](./materials/img_df.png)  
The ```emotion``` column contains a numeric code ranging from 0 to 6, inclusive, for the emotion expressed by the image. The ```pixels``` column contains a string surrounded in quotes for each image.  

#### Checking missing and duplicated values  
The given data has no missing values ​​but has 1793 duplicate values. After removing duplicate values, the dataset is left with 34094 columns  

#### On the `emotion` column
![img_df_emotion](./materials/img_df_emotion.png)  
There is an imbalance between the labels in the given dataset:  
- The number of label 3 (Happy) is too much, almost twice as many as other labels  
- Meanwhile, the number of label 1 (Disgust) is too small, accounting for about 1.4% of the data set.  

This can lead to:  
- Classification algorithms that learn from imbalanced data tend to be biased towards the majority group (in this case label 3).  
- Conventional model performance metrics, such as accuracy, can be distorted by label imbalance...  

#### Convert dataset  
Convert the data on the `pixels` column to a numpy array with:  
- images (3-dimensional array): Stores images as 2D arrays with a shape of (len(df), 48, 48).   
- image_raws: (2-dimensional array): Stores raw pixel data as 1D arrays with a shape of (len(df), 2304).
![img_10faces](./materials/img_10faces.png)   

### 2. Principle Component Analysis
Unsupervised learning can be further categorized into two main tasks: data transformation and clustering. In this study, we will focus on data transformation using unsupervised learning techniques. These techniques aim to modify the data to make it easier for computers and humans to analyze and understand.  

One of the most common applications of unsupervised data transformation is dimensionality reduction. This process reduces the number of features (dimensions) in the data. When the data has a high number of features, it can be computationally expensive and difficult to analyze. Dimensionality reduction techniques help to overcome these challenges.  

Principal Component Analysis (PCA) is a popular technique for dimensionality reduction. It transforms the data into a new set of features called principal components (PCs). These PCs are ordered by their importance, capturing the most significant variations in the data. By selecting a subset of the most informative PCs, we can achieve a significant reduction in data size while preserving the essential information for analysis.  

#### Question 1: Can you visualize the data projected onto two principal components?  
#### Question 2: How to determine the optimal number of principal components using ```pca.explained_variance_```? Explain your selection process.  



### 3. Image Classification
The classification task will compare the performance using both:

- Original data: The data before applying PCA.
- Transformed data: The data projected onto the optimal number of principal components identified earlier. Utilize the **optimal number of principal components** identified in the previous question.

**Import these prerequisites before proceeding**  

**PCA with 158 principal components**  

**Split data into train/val/test subsets**

#### Model 1. RandomForestClassifier()
- On Original dataset  

### 4. Evaluating Classification Performance 
