# Handling-Imbalaced-Dataset-Upsampling-and-Downsampling
The project highlights data generation, reproducibility with random seeds, and techniques to prepare data for fair model training.

#Project Overview: 
This project demonstrates handling imbalanced datasets using downsampling of the majority class.
Imbalanced datasets are common in real-world scenarios (e.g., fraud detection, medical diagnoses) where one class dominates.
Downsampling helps create a balanced dataset, allowing machine learning models to learn fairly from both classes.

#Why It Matters: 
Imbalanced datasets can bias model predictions toward the majority class.
Downsampling reduces the size of the majority class to match the minority class, improving model fairness and accuracy.

#Key Features: 

Recreate an imbalanced dataset with two features and a binary target.
Separate majority and minority classes.
Downsample the majority class to balance the dataset.
Combine minority class with downsampled majority class.
Output class distribution after balancing.

#Technologies / Tools:

Python 3.x – programming language
VS Code – IDE used for development
Jupyter Notebook (.ipynb) – project file
Libraries: numpy, pandas, scikit-learn (resample)

#What I Did in the Code:

In this project, I created and handled an imbalanced dataset using Python. Here’s a step-by-step summary of what the code does:

Recreated an imbalanced dataset:

 -Generated 1000 samples with two features (feature1 and feature2).
 -90% of the samples belong to class 0 (majority) and 10% to class 1 (minority).
Generated features for each class:
 -Class 0 and class 1 have different distributions for both features to simulate real-world data.
Combined the classes into a single dataset:
 -Created a DataFrame DF containing all samples with a target column indicating the class.
Separated majority and minority classes:
 -Split the dataset into DF_majority (class 0) and DF_minority (class 1).
Balanced the dataset using two techniques:
 a) Downsampling:
   -Randomly selected samples from the majority class so that its size matches the minority class.
   -Used replace=False to avoid duplicates and random_state=42 for reproducibility.
   -Combined the downsampled majority class with the minority class to form DF_downsampled.
 b) Upsampling:
   -Randomly duplicated samples from the minority class so that its size matches the majority class.
   -Used replace=True to allow duplication and random_state=42 for reproducibility.
   -Combined the upsampled minority class with the majority class to form DF_upsample
Verified the result:
 -Checked class distributions after downsampling and upsampling to ensure the dataset is balanced.

#What I Achieved:

Successfully created a synthetic imbalanced dataset to simulate real-world scenarios.
Learned how to identify and separate majority and minority classes in a dataset.
Applied downsampling to balance the dataset, ensuring that both classes have equal representation.
Produced a balanced dataset that can be used for building fair and accurate machine learning models.
Gained hands-on experience in data preprocessing techniques for handling imbalanced datasets, which is a critical skill in data science and machine learning.



 
