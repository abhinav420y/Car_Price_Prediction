import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV


df = pd.read_csv('car_data.csv')
#car name, is of not much importance
df.drop('Car_Name',axis=1, inplace=True)

df['Current_Year'] = 2021
df['No_Year'] = df['Current_Year'] - df['Year']

df.drop(['Year','Current_Year'], axis=1, inplace=True)

#encoding categorical values
df = pd.get_dummies(df, drop_first=True)
df.head()

#getting dependent and independent features
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#applying our model

clf = RandomForestRegressor()

#hyperparameters(Randomized Search CV)
#no of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
#no of features to consider at every split
max_features = ['auto', 'sqrt']
#max no of levels in trees
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
#min number of samples reqquired to split a node
min_samples_split = [2, 5, 10, 15, 100]
#min number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

#create the random grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}
random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                           scoring='neg_mean_squared_error', n_iter=10,
                           cv=5, verbose=2, random_state=42, n_jobs=1 )

random.fit(X_train, y_train)
y_pred = random.predict(X_test)
print(y_pred)

#pickling our file
with open('rfr_model.pkl','wb') as f_out:
    pickle.dump(random, f_out)
    f_out.close()
