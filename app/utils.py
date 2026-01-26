# ================= Core Python =================
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ================= NLP / Text Processing =================
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# ================= Data Preprocessing =================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# ================= Sparse Matrix Support =================
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity

# ================= Regression Models =================
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # optional alternate model

# ================= Model Evaluation =================
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ================= Feature Ranking / Selection =================
from sklearn.inspection import permutation_importance
#import shap  # optional but recommended for explainability

# ================= Deployment (FastAPI) =================
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# ================= Server Runner =================
import uvicorn


df = pd.read_csv('D:\AI Journey\Resumee\data\AI_Resume_Screening.csv')
# print(df.head())



# print(df.describe())

# print(df.nunique()) Certifications


df['has_certification'] = (
    df['Certifications']
        .fillna('')
        .str.strip()
        .replace('None', '')
        .ne('')
        .astype(int)
)


df['Recruiter Decision'] = df['Recruiter Decision'].map({'Hire': 1, 'Reject': 0}).astype(int)


print(df['Certifications'].value_counts())

target = 'ai_score'

numeric_df = df.select_dtypes(include=['int64', 'float64'])
numeric_df = numeric_df.drop(columns=[target], errors='ignore')
numeric_df = numeric_df.drop(columns=['resume_id'], errors='ignore')

corr = numeric_df.corr()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Excluding ai_score)')
plt.show()

sns.regplot(x='Salary Expectation ($)', y='AI Score (0-100)', data=df)
plt.title('AI Score vs Salary Expectation')
plt.show()

sns.boxplot(x='has_certification', y='AI Score (0-100)', data=df)
plt.title('AI Score vs Certification')
plt.xlabel('Certification (1=Has, 0=None)')
plt.show()

sns.regplot(x='Experience (Years)', y='AI Score (0-100)', data=df)
plt.title('AI Score vs Experience')
plt.show()

sns.boxplot(x='Recruiter Decision', y='AI Score (0-100)', data=df)
plt.title('AI Score vs Recruiter Decision')
plt.xlabel('Recruiter Decision (1=Hire, 0=Reject)')
plt.show()

#df.groupby('recruiter_decision')[['ai_score','experience_years','salary_expectation','certification']].mean()








# print(df.isnull().sum())
print(df.info())
print(df.head())

