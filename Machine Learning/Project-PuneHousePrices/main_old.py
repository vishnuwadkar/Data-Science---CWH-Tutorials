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
# 1.Load the dataset
housing = pd.read_csv("housing.csv")

#2.Create a stratified test set
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]   #set aside for testing

#now, we will work on the copy of training set
housing = strat_train_set.copy()

#3. Seperate labels and features
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)
# print(housing, housing_labels)

#4. Select numerical and categorical columns
num_attribs = housing.drop('ocean_proximity', axis=1).columns.tolist()
cat_attribs = ['ocean_proximity']

#5. Create a pipeline for numerical columns
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

#6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)

#7. Training the model

#Linear Regression Model
linReg = LinearRegression()
linReg.fit(housing_prepared, housing_labels)
lin_pred = linReg.predict(housing_prepared)
# lin_mse = rmse(housing_labels, lin_pred)
# print("Linear Regression Model: ", lin_mse)
lin_rmses = -cross_val_score(linReg, housing_prepared, housing_labels,
                            scoring='neg_root_mean_squared_error', cv=10)  #cv=10 means 10 fold cross validation
print("Linear Regression Model: ", lin_rmses.mean())
# OUTPUT : Linear Regression Model:  69022.03057347347



#Decison Tree Model
# DecTree = DecisionTreeRegressor()
# DecTree.fit(housing_prepared, housing_labels)
# dec_pred = DecTree.predict(housing_prepared)
# dec_mse = rmse(housing_labels, dec_pred)
# print("Decision Tree Model: ", dec_mse)
# #this gave a zero error, meaning it is overfitting

DecTree = DecisionTreeRegressor()
DecTree.fit(housing_prepared, housing_labels)
dec_pred = DecTree.predict(housing_prepared)
dec_rmses = -cross_val_score(DecTree, housing_prepared, housing_labels,
                            scoring='neg_root_mean_squared_error', cv=10)  #cv=10 means 10 fold cross validation
print("Decision Tree Model: ", dec_rmses.mean())
#OUTPUT : Decision Tree Model:  69107.96622915645


#Random Forest Model
RForest = RandomForestRegressor()
RForest.fit(housing_prepared, housing_labels)
rf_pred = RForest.predict(housing_prepared)
# rf_mse = rmse(housing_labels, rf_pred)
# print("Random Forest Model: ", rf_mse)
rf_rmses = -cross_val_score(RForest, housing_prepared, housing_labels,
                            scoring='neg_root_mean_squared_error', cv=10)  #cv=10 means 10 fold cross validation
print("Random Forest Model: ", rf_rmses.mean())
#OUTPUT : Random Forest Model:  49460.15629086552


#Thus, after considering the all the RMSsE values, we can say that Random Forest is the best model among the three, after using the cross validation technique

