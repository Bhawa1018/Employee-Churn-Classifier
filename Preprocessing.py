
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Load dataset
# (Ensure you've downloaded the dataset from https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset)
url = '/content/Employee.csv'  # Replace with actual path
df = pd.read_csv(url)

# Overview of the dataset
df.head()

import scipy.stats as stats

# Plot Q-Q plots and histograms
def plot_qq_histograms(data, features):
    for feature in features:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        stats.probplot(data[feature].dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {feature}')

        plt.subplot(1, 2, 2)
        sns.histplot(data[feature].dropna(), kde=True)
        plt.title(f'Histogram of {feature}')

        plt.show()

# Select numerical features for analysis
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
plot_qq_histograms(df, numerical_features)
