## -Preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inlineee
import seaborn as sns

df = pd.read_csv('cleaned_age_added.csv')

### Clean up

df['SURGICAL_SPECIALTY'].unique()

df['SURGICAL_SPECIALTY'] = df['SURGICAL_SPECIALTY'].map(lambda x:x.upper())

<hr>

### Duration conversion
### hh:mm -> min

def str_2_time(str):
    l = str.split(":")
    return int(l[0])*60+int(l[1])
df['Duration'] = df['Duration of Surgery (hh:mm).1'].map(str_2_time)

<hr>

### Missing values imputation and outliers removal

df.info()

#convert object to float
df['SN - BM - Pre-Op INR'] = df['SN - BM - Pre-Op INR'].apply(lambda x: float(x) if x != '.' else float('nan'))

#remove outliers
df = df[np.abs(df['SN - BM - Pre-Op INR']-df['SN - BM - Pre-Op INR'].mean())<=(5*df['SN - BM - Pre-Op INR'].std())]

#fill Nan with mean value
df['SN - BM - Pre-Op INR'] = df['SN - BM - Pre-Op INR'].fillna(df['SN - BM - Pre-Op INR'].mean())

#convert object to float
df['SN - BM - Pre-Op Platelet Count'] = df['SN - BM - Pre-Op Platelet Count'].apply(lambda x: float(x) if x != '.' else float('nan'))

#outliers removal
df = df[np.abs(df['SN - BM - Pre-Op Platelet Count']-df['SN - BM - Pre-Op Platelet Count'].mean())<=(5*df['SN - BM - Pre-Op Platelet Count'].std())]

#fill Nan with mean value
df['SN - BM - Pre-Op Platelet Count'] = df['SN - BM - Pre-Op Platelet Count'].fillna(df['SN - BM - Pre-Op Platelet Count'].mean())

df.info()

### Drop useless columns

df = df.drop(['Masked FIN'], 1)

df = df.drop(['Sequence No.'], 1)
df = df.drop(['Duration of Surgery (hh:mm).1'], 1)
df = df.drop(['EBL'], 1)

df = df.drop(['SN - BM - PRBC Ordered'], 1)
#since high dimensional data is not good for random forest, we drop SURG_PROCEDURE to get better results
df = df.drop(['SURG_PROCEDURE'], 1)

df.head()

df = df.dropna(subset=['age','SN - BM - Pre-Op INR','SN - BM - Pre-Op Platelet Count'])

df_complete = df.copy()
df_seg2 = df_complete[df_complete['ResultAfterSurgery'].isnull()]
df_seg2 = df_seg2.drop('ResultAfterSurgery',1)
df_seg1 = df_complete.dropna()

<hr>

## Modeling

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

### Dummify categorical variables

df.head(20)

df_toy = pd.get_dummies(df)
df_to_be_completed = df_toy[df_toy['ResultAfterSurgery'].isnull()]
df_toy = df_toy.dropna()

df_toy.info()

### Seperate target value

df_toy_target = df_toy['ResultAfterSurgery']
df_toy = df_toy.drop('ResultAfterSurgery',1)

#cross-valisdation for models
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, df_toy, df_toy_target, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

### Validation - Baseline KNN

n_neighbors = [1,2,5,10,20,50]
cv_knn = [rmse_cv(KNeighborsRegressor(n_neighbors=n_neighbor)).mean() 
            for n_neighbor in n_neighbors]

cv_knn = pd.Series(cv_knn, index = n_neighbors)
cv_knn.plot(title = "Validation")
plt.xlabel("# of neighbors")
plt.ylabel("rmse")

### Validation - Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

rf_trees = [10,20,50,100,300,500]
cv_rf = [rmse_cv(RandomForestRegressor(n_estimators = rf_tree,max_features='sqrt')).mean() 
            for rf_tree in rf_trees]

cv_rf = pd.Series(cv_rf, index = rf_trees)
cv_rf.plot(title = "Validation")
plt.xlabel("# of trees")
plt.ylabel("rmse")

### Neural Network

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler

X = df_toy
X = StandardScaler().fit_transform(X)
y = df_toy_target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = Sequential()
model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(1)))
model.compile(loss = "mse", optimizer = "sgd")
model.summary()

model.fit(X_train, y_train,validation_split = 0.25)

### Build the model

radm = RandomForestRegressor(n_estimators = 300)

radm.fit(df_toy,df_toy_target)

### Feature importance

indices = np.argsort(radm.feature_importances_)[::-1]
# Print the feature ranking
print('Feature ranking:')
for f in range(10):
    print('%d. feature %d %s (%f)' % (f+1 , indices[f], df_toy.columns[indices[f]],
                                      radm.feature_importances_[indices[f]]))

<hr>

### Predict post anemia status values

df_to_be_completed.info()

df_to_be_completed = df_to_be_completed.drop('ResultAfterSurgery',1)

df_seg2['ResultAfterSurgery'] = radm.predict(pd.get_dummies(df_to_be_completed))

#concatenate two segments
frames = [df_seg1,df_seg2]
df_missing_post_anemia_filled = pd.concat(frames)

df_missing_post_anemia_filled['SN - BM - Red Blood Cells'].max()

sns.violinplot(x=df_missing_post_anemia_filled['SN - BM - Red Blood Cells'])

df_missing_post_anemia_filled.info()

<hr>

### Predicting Red Blood Cell Amount for patients not getting anemia after surgery

#### Get records of patients not getting anemia after surgery

df_predict_RBC_raw = df_missing_post_anemia_filled[df_missing_post_anemia_filled['ResultAfterSurgery'] > 10]

df_predict_RBC_raw.info()

df_predict_RBC_raw

df_predict_RBC_target = df_predict_RBC_raw['SN - BM - Red Blood Cells']
df_predict_RBC_features = df_predict_RBC_raw.drop(['SN - BM - Red Blood Cells','SN - BM - Cryoprecipitate','SN - BM - Fresh Frozen Plasma','SN - BM - Platelets','Duration','Allogeneic Blood Transfusion'],1)

df_predict_RBC_features = df_predict_RBC_features.drop('ResultAfterSurgery',1)

df_predict_RBC_features = pd.get_dummies(df_predict_RBC_features)

#cross-valisdation for models
def rmse_cv_blood_predict(model):
    rmse= np.sqrt(-cross_val_score(model, df_predict_RBC_features, df_predict_RBC_target, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

#### KNN

n_neighbors = [1,2,5,10,20,50]
cv_knn =[rmse_cv_blood_predict(KNeighborsRegressor(n_neighbors=n_neighbor)).mean() 
            for n_neighbor in n_neighbors]

cv_knn = pd.Series(cv_knn, index = n_neighbors)
cv_knn.plot(title = "Validation")
plt.xlabel("# of neighbors")
plt.ylabel("rmse")

#### Random Forest

rf_trees = [10,20,50,100,300,500]
cv_rf = [rmse_cv_blood_predict(RandomForestRegressor(n_estimators = rf_tree,max_features='sqrt')).mean() 
            for rf_tree in rf_trees]

cv_rf = pd.Series(cv_rf, index = rf_trees)
cv_rf.plot(title = "Validation")
plt.xlabel("# of trees")
plt.ylabel("rmse")

#### Neural Network

model = Sequential()
model.add(Dense(32, input_dim = standardized_X.shape[1], W_regularizer=l1(1)))
model.add(Dense(16))
model.compile(loss = "mse", optimizer = "sgd")
standardized_X = StandardScaler().fit_transform(df_predict_RBC_features)
model.fit(standardized_X, df_predict_RBC_target,validation_split = 0.25)

### Finally we choose Random forest to predict RED BLOOD CELLS used during a surgery

model = Sequential()
model.add(Dense(32, input_dim = standardized_X.shape[1], W_regularizer=l1(1)))
model.add(Dense(16))
model.compile(loss = "mse", optimizer = "adagrad")
standardized_X = StandardScaler().fit_transform(df_predict_RBC_features)
model.fit(standardized_X, df_predict_RBC_target,validation_split = 0.25)

model = Sequential()
model.add(Dense(32, input_dim = standardized_X.shape[1], W_regularizer=l1(1)))
model.compile(loss = "mse", optimizer = "sgd")
standardized_X = StandardScaler().fit_transform(df_predict_RBC_features)
model.fit(standardized_X, df_predict_RBC_target,validation_split = 0.25)

model = Sequential()
model.add(Dense(32, input_dim = standardized_X.shape[1], W_regularizer=l1(1)))
model.add(Dense(16))
model.add(Dense(8))
model.compile(loss = "mse", optimizer = "sgd")
standardized_X = StandardScaler().fit_transform(df_predict_RBC_features)
model.fit(standardized_X, df_predict_RBC_target,validation_split = 0.25)

model = Sequential()
model.add(Dense(32, input_dim = standardized_X.shape[1], W_regularizer=l1(1)))
model.compile(loss = "mse", optimizer = "sgd")
standardized_X = StandardScaler().fit_transform(df_predict_RBC_features)
model.fit(standardized_X, df_predict_RBC_target,validation_split = 0.25)

model = Sequential()
model.add(Dense(8, input_dim = standardized_X.shape[1], W_regularizer=l1(1)))
model.compile(loss = "mse", optimizer = "sgd")
standardized_X = StandardScaler().fit_transform(df_predict_RBC_features)
model.fit(standardized_X, df_predict_RBC_target,validation_split = 0.25)

res = {
u'SURGICAL_SPECIALTY': u'12',
u'age': u'22',
u'Surgeon Hash Name': u'Dr. 22',
u'PATIENT_TYPE': u'22',
u'SN - BM - Pre-Op INR': u'22',
u'SN - BM - Pre-Op Platelet Count': u'22',
u'ResultsBeforeSurgery': u'22'
}

df_predict_RBC_features.info()

res.values()

tmp = pd.DataFrame([res.values()],columns=res.keys())
tmp['ResultsBeforeSurgery'] = tmp['ResultsBeforeSurgery'].map(lambda x:float(x))
tmp['SN - BM - Pre-Op INR'] = tmp['SN - BM - Pre-Op INR'].map(lambda x:float(x))
tmp['SN - BM - Pre-Op Platelet Count'] = tmp['SN - BM - Pre-Op Platelet Count'].map(lambda x:float(x))
tmp['age'] = tmp['age'].map(lambda x:float(x))

tmp.info()

dd = df_predict_RBC_features.append(tmp)

df_predict_RBC_features.shape

pd.get_dummies(dd).iloc[-1].values.shape

pd.DataFrame(res.items())

final_model = RandomForestRegressor(n_estimators = 300)
final_model.fit(df_predict_RBC_features,df_predict_RBC_target)