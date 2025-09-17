import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
#importing libraries for the ML part
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"  #pkl is the extension for the file containing the model which can be used later for prediction
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):   # a function that builds and returns the pipeline
    # Create a pipeline for numerical columns
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('oneHot', OneHotEncoder(handle_unknown='ignore'))   #will ignore the unknown categories during transform
    ]
    )

    #Contructing the full pipeline
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', cat_pipeline, cat_attribs),
    ])
    return full_pipeline


#if model pipeline doesn't exist, create it
if not os.path.exists(PIPELINE_FILE):
    housing = pd.read_csv("housing.csv")
    housing['income_cat'] = pd.cut(housing['median_income'],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing = housing.loc[train_index].drop('income_cat', axis=1)
    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis=1)
    num_attribs = housing.drop('ocean_proximity', axis=1).columns.tolist()
    cat_attribs = ['ocean_proximity']
      