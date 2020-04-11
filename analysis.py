# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# ------------------------------------------------------------
#
# Import dataset
#
# ------------------------------------------------------------
print("Loading dataset from github...")

url='https://github.com/jafj/Mod4_Chal4b/raw/master/breast-cancer.xls'
datatypes = {
   'age': str,
   'menopause': str,
   'tumor-size': str,
   'inv-nodes': str,
   'node-caps': str,
   'deg-malig': int,
   'breast': str,
   'breast-quad': str,
   'irradiat': str,
   'Class': str
}
dataset = pd.read_excel(url, dtype = datatypes, usecols = 'A:F,H:K')

# ------------------------------------------------------------
#
# Clean the data
#
# ------------------------------------------------------------
print("Cleaning data...")

# Remove entries with node-caps == '?'
data_tmp = dataset.iloc[:].values
data_tmp = np.delete(data_tmp, np.where(data_tmp == '?')[0], axis=0)

# Now extract independent and dependent variables
X = data_tmp[:, :8]
y = data_tmp[:, 8]

# Helper functions
def clean_tumor_size(dat):
   if dat == '2014-10-01 00:00:00':
      return '10-14'
   elif dat == '2019-09-05 00:00:00':
      return '5-9'
   return dat

def clean_inv_nodes(dat):
   if dat == '2019-08-06 00:00:00':
      return '6-8'
   elif dat == '2019-05-03 00:00:00':
      return '3-5'
   elif dat == '2019-11-09 00:00:00':
      return '9-11'
   elif dat == '2014-12-01 00:00:00':
      return '12-14'
   return dat

X[:,2] = [clean_tumor_size(x) for x in X[:,2]]
X[:,3] = [clean_inv_nodes(x) for x in X[:,3]]

# ------------------------------------------------------------
#
# Encode catagorical data
#
# ------------------------------------------------------------
print("Encoding catagorical data...")

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Manually encode age, tumor-size, and inv-nodes
X[:,0] = [int(re.match('([0-9]+)', x).group(0)) for x in X[:,0]]
X[:,2] = [int(re.match('([0-9]+)', x).group(0)) for x in X[:,2]]
X[:,3] = [int(re.match('([0-9]+)', x).group(0)) for x in X[:,3]]

# Automatically encode remaining fields
transformer = LabelEncoder()
X[:,4] = transformer.fit_transform(X[:,4])
transformer = LabelEncoder()
X[:,7] = transformer.fit_transform(X[:,7])
transformer = LabelEncoder()
y = transformer.fit_transform(y)

transformer = ColumnTransformer(
    transformers=[
        ("Meno",
         OneHotEncoder(),
         [1]
         )
    ], remainder='passthrough'
)
X = transformer.fit_transform(X)[:,1:] # Avoiding dummy variable trap

transformer = ColumnTransformer(
    transformers=[
        ("BreastQuad",
         OneHotEncoder(),
         [7]
         )
    ], remainder='passthrough'
)
X = transformer.fit_transform(X)[:,1:] # Avoiding dummy variable trap

# ------------------------------------------------------------
#
# Feature scaling and data splitting
#
# ------------------------------------------------------------
print("Feature scaling...")

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Split dataset into training and sest sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size = 0.25,
                                                    random_state = 0)

# ------------------------------------------------------------
#
# Data linearity detection
#
# ------------------------------------------------------------
print("Linearity detection...")

# Apply Grid Search (with k-value folding) to determine if the model is linear
# or non-linear
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
classifier = SVC()
parameters = [{'C': [0.01, 1, 10], 'kernel': ['linear']},
              {'C': [0.01, 1, 10], 'kernel': ['rbf'],
               'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

if best_parameters['kernel'] == 'rbf':
   print('** Data appears non-linear')
else:
   print('** Data appears linear')

# ------------------------------------------------------------
#
# Optimal model determination
#
# ------------------------------------------------------------
print("Optimal non-linear classification model determination...")

# The above shows this is a non-linear model. Let's try a selection of models
# with a variety of trial kPCA-driven feature decompositions. As the data are
# non-linear and we desire classification, look at K-NN, Decision Tress,
# Random Forest, Kernel SVN and XGBoost. Note using kPCA with rbf kernel as we
# believe the data is non-linear

# Helper method to run a gridsearch for a given estimator and parameter set
def run_grid_search(x_in, y_in, classifier,
                    gs_parameters, best_parameters, feature_dim):
   grid_search = GridSearchCV(estimator = classifier,
                              param_grid = gs_parameters,
                              scoring = 'accuracy',
                              cv = 10,
                              n_jobs = -1)
   grid_search = grid_search.fit(x_in, y_in)
   run_accuracy = grid_search.best_score_
   run_parameters = grid_search.best_params_
   if run_accuracy > best_parameters['accuracy']:
      print('    {}: Replacing {:.3f} with {:.3f}'.format(
                                             type(classifier).__name__,
                                             best_parameters['accuracy'],
                                             run_accuracy))
      return {'feature-dim': feature_dim,
              'accuracy': run_accuracy,
              'params': run_parameters}
   print('    {}: Keeping accuracy {:.3f} (this run: {:.3f})'.format(
                                             type(classifier).__name__,
                                             best_parameters['accuracy'],
                                             run_accuracy))
   return best_parameters

# All classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
classifiers = {'kSVN': SVC(),
               'kNN': KNeighborsClassifier(),
               'decision-tree': DecisionTreeClassifier(),
               'rand-forest': RandomForestClassifier(),
               'xgboost': XGBClassifier()}

# Record of search results
model_search = {'kSVN': {'accuracy': 0},
               'kNN': {'accuracy': 0},
               'decision-tree': {'accuracy': 0},
               'rand-forest': {'accuracy': 0},
               'xgboost': {'accuracy': 0}}

# Search 2 to 12 feature dimensions
from sklearn.decomposition import KernelPCA
best_xgboost_accuracy = 0
for i in range(1, 13):
   print('  kPCA features: {}'.format(i))
   # Run kPCA
   if i == 12:
      X_transformed = X_train
   else:
      transformer = KernelPCA(n_components=i, kernel='rbf')
      X_transformed = transformer.fit_transform(X_train)
   
   # kSVN
   parameters = [{'C': [0.5, 0.75, 1, 1.25, 1.5, 1.75], 'kernel': ['rbf'],
                  'gamma': [0.05, 0.075, 0.1, 0.15, 0.175]}]
   model_search['kSVN'] = run_grid_search(X_transformed, y_train,
                                          classifiers['kSVN'],
                                          parameters,
                                          model_search['kSVN'],
                                          i)
   
   # kNN
   parameters = [{'n_neighbors': [5, 6, 7, 8, 9, 10, 11, 12],
                  'weights': ['uniform', 'distance'],
                  'metric': ['minkowski'],
                  'p': [2]}]
   model_search['kNN'] = run_grid_search(X_transformed, y_train,
                                         classifiers['kNN'],
                                         parameters,
                                         model_search['kNN'],
                                         i)
   
   # Decision Tree
   parameters = [{'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random']}]
   model_search['decision-tree'] = run_grid_search(X_transformed, y_train,
                                                 classifiers['decision-tree'],
                                                 parameters,
                                                 model_search['decision-tree'],
                                                 i)
   
   # Random Forest
   parameters = [{'n_estimators': [5, 10, 25, 50, 75, 100],
                  'criterion': ['gini', 'entropy']}]
   model_search['rand-forest'] = run_grid_search(X_transformed, y_train,
                                                 classifiers['rand-forest'],
                                                 parameters,
                                                 model_search['rand-forest'],
                                                 i)

   # XGBoost
   parameters = [{'max_depth': [2, 3, 4],
                  'n_estimators': [5, 10, 20],
                  'learning_rate': [0.1, 0.01, 0.05]}]
   model_search['xgboost'] = run_grid_search(X_transformed, y_train,
                                             classifiers['xgboost'],
                                             parameters,
                                             model_search['xgboost'],
                                             i)

# Print details of the best model found
best_model = None
best_accuracy = 0
for k in model_search:
   if model_search[k]['accuracy'] > best_accuracy:
      best_accuracy = model_search[k]['accuracy']
      best_model = k
print('** The best model determined is {} with:'.format(best_model))
print('**   Accuracy: {:.3f}'.format(model_search[best_model]['accuracy']))
print('**   kPCA dimensions: {}'.format(model_search[best_model]['feature-dim']))
print('**   Parameters: {}'.format(str(model_search[best_model]['params'])))

# ------------------------------------------------------------
#
# Model confirmation
#
# ------------------------------------------------------------
print('Model confirmation...')

# Get our chosen classifier and configure it
params = model_search[best_model]['params']
classifier = classifiers[best_model].set_params(**params)

# Confirm k-fold validation on the training set
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train, y = y_train,
                             cv = 10)
msg = '** k-fold validation accuracy on training set: {:.3f}'
print(msg.format(accuracies.mean()))

# Predict on test set
print('** Predicting on test set')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)

print('** Confusion matrix:')
print(cm)
print('** Test set accuracy: {:.3f}'.format(accuracy))
