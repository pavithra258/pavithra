import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Dropout
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



data = pd.read_csv("C:/Users/ELCOT/Desktop/PROJECT for ES/thyroidDF.csv")
print(data.head())

print(data.isnull().sum())

# Drop redundant attributes and modify the original dataframe
data.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'referral_source', 'patient_id'], axis=1, inplace=True)

#re-mapping target values to diagnostic group
diagnoses = { 'A': 'hyperthyroid conditions',
              'B': 'hyperthyroid conditions',
              'C': 'hyperthyroid conditions',
              'D': 'hyperthyroid conditions',
              'E': 'hypothyroid conditions',
              'F': 'hypothyroid conditions',
              'G': 'hypothyroid conditions',
              'H': 'hypothyroid conditions',
              'I': 'binding protein',
              'J': 'binding protein',
              'K': 'general health',
              'L': 'replacement therapy',
              'M': 'replacement therapy',
              'N': 'replacement therapy',
              'O': 'antithyroid treatment',
              'P': 'antithyroid treatment',
              'Q': 'antithyroid treatment',
              'R': 'miscellaneous',
              'S': 'miscellaneous',
              'T': 'miscellaneous'}

data['target'] = data['target'].map(diagnoses)
data.dropna(subset=['target'], inplace=True)

print(data['target'].value_counts())

print(data[data.age>100])

#spliting the data
x=data.iloc[:,0:-1]
y= data.iloc[:,-1]


print(x)


print(x['sex'].unique())

x['sex'].replace(np.nan, 'F', inplace=True)


print(x['sex'].value_counts())

#converting the data
x['age']=x['age'].astype('float')
x['TSH']=x['TSH'].astype('float')
x['T3']=x['T3'].astype('float')
x['TT4']=x['TT4'].astype('float')
x['T4U']=x['T4U'].astype('float')
x['FTI']=x['FTI'].astype('float')
x['TBG']=x['TBG'].astype('float')


print(x.info())


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import numpy as np

# create an OrdinalEncoder object
ordinal_encoder = OrdinalEncoder(dtype='int64')

# apply ordinal encoding to the categorical features in x
x[x.columns[1:16]] = ordinal_encoder.fit_transform(x.iloc[:, 1:16])

# replace the nan values with 0
x.replace(np.nan, 0, inplace=True)

# print the transformed x dataframe
print(x)
# Replace missing values in x with 0
x.fillna(0, inplace=True)

# Use OrdinalEncoder to handle categorical values in x
ordinal_encoder = OrdinalEncoder(dtype='int64')
x[x.columns[1:16]] = ordinal_encoder.fit_transform(x.iloc[:, 1:16])

# Use LabelEncoder to encode y values
label_encoder = LabelEncoder()
y = pd.DataFrame(label_encoder.fit_transform(y), columns=['target'])

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Scale the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Handling Imbalanced data
os = SMOTE(random_state=0,k_neighbors=1)
x_bal, y_bal = os.fit_resample(x_train, y_train)
x_test_bal, y_test_bal = os.fit_resample(x_test, y_test)
print(y_train.value_counts())
print(x_bal)
# Convert arrays to dataframes
columns=['age','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_meds','sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid','lithium','goitre','tumor','hypopituitary','psych','TSH','T3','TT4','T4U','FTI','TBG']
x_train_bal = pd.DataFrame(x_bal, columns=columns)
y_train_bal = pd.DataFrame(y_bal, columns=['target'])
x_test_bal = pd.DataFrame(x_test_bal, columns=columns)
y_test_bal = pd.DataFrame(y_test_bal, columns=['target'])
x_bal = pd.DataFrame(x_bal, columns=columns)

#Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier().fit(x_bal,y_bal.values.ravel())
y_pred = rfr.predict(x_test_bal)

print(classification_report(y_test_bal,y_pred))


from sklearn.inspection import permutation_importance
results = permutation_importance(rfr, x_bal, y_bal, scoring='accuracy')
feature_importance = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 'sick', 'pregnant', 'thyroid_surgery','I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
importance = results.importances_mean
importance = np.sort(importance)
for i,v in enumerate(importance):
    i=feature_importance[i]
    print('feature: {:<20} Score: {}'.format(i,v))

plt.figure(figsize=(10,10))

plt.bar(feature_importance, importance.astype(float))

plt.xticks(rotation=30, ha='right')
plt.show()

print(x.head())

x_bal = x_bal.drop(['age','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_meds','sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid','lithium'], axis=1)
x_test_bal = x_test_bal.drop(['age','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_meds','sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid','lithium'], axis=1)

print(x_bal.head())

print(data.info())

import seaborn as sns
corrmat = x.corr()
f, ax = plt.subplots(figsize = (9,8))
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
plt.show()

# select features for model
cols = ['goitre','tumor','hypopituitary','psych','TSH','T3','TT4','T4U','FTI','TBG']

# create balanced training set
x_bal = x_train_bal[cols]
y_bal = y_train_bal

# create balanced test set
x_test_bal = x_test_bal[cols]
y_test_bal = y_test_bal

#Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
rfr1 = RandomForestClassifier().fit(x_bal, y_bal.values.ravel())
y_pred = rfr1.predict(x_test_bal)

print(classification_report(y_test_bal, y_pred))

from xgboost import XGBClassifier

xgb1 = XGBClassifier()
xgb1.fit(x_bal, y_bal)

y_pred = xgb1.predict(x_test_bal)
print(classification_report(y_test_bal, y_pred))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
sv = SVC()
sv.fit(x_bal,y_bal)
y_pred = sv.predict(x_test_bal)
print(classification_report(y_test_bal,y_pred))


#ANN Model

model = Sequential()
model.add(Dense(units = 128, activation='relu', input_shape=(10,)))
model.add(Dense(units = 128, activation='relu', kernel_initializer='random_uniform'))
model.add(Dropout(0.2))
model.add(Dense(units = 128, activation='relu', kernel_initializer='random_uniform'))
model.add(Dropout(0.2))
model.add(Dense(units = 128, activation='relu', kernel_initializer='random_uniform'))
model.add(Dense(units =1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_bal,y_bal, validation_data=[x_test_bal, y_test_bal], epochs=0)

rfr1.predict([[0,0,0,0,0.000000,0.0,0.0,1.00,0.0,40.0]])
model.predict([[0,0,0,0,0.000000,0.0,0.0,1.00,0.0,40.0]])
sv.predict([[0,0,0,0,0.000000,0.0,0.0,1.00,0.0,40.0]])


print(y_pred)

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import warnings

params = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

random_svc = RandomizedSearchCV(SVC(), params, scoring='accuracy', cv=5, n_jobs=-1)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    random_svc.fit(x_bal, y_bal)

print(random_svc.best_params_)
# Output: {'kernel': 'rbf', 'gamma': 0.1, 'C': 100}

sv1 = SVC(kernel='rbf', gamma=0.1, C=100)
sv1.fit(x_bal, y_bal)
y_pred = sv1.predict(x_test_bal)



#saving the model
import pickle
pickle.dump(xgb1, open('thyroid_1_model.pkl', 'wb'))
features = np.array([[0, 0, 0, 0, 0.000000, 0.0, 0.0, 1.00, 0.0, 40.0]])
print(label_encoder.inverse_transform(xgb1.predict(features)))
pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))
print(data['target'].unique())
print(y['target'].unique())


pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))
