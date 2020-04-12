# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

"""
### **Overall strategy:**

1.   Load and clean data
2.   Encode categorical data, perform feature scaling, and split in to
     training and test sets
3.   Determine if data is linear or non-linear
4.   Run grid search (with k-fold cross validation) on appropriate linear or
     non-linear models to determine best model selection, testing all possible
     kPCA feature reduction dimensions
5.   Confirm model (through k-fold validation) from Step 5 on training data
     subset, and then test with testing data subset
"""

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
"""
# 1. Load and clean data

Data is loaded from a public Github repository. Cleaning is undertaken to
restore those data fields mis-identified as dates to their correctly-ordered
groupings, inferred by looking at the correctly interpreted categorical data
values.
"""
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
"""
Data assumptions and impution strategy:

*  *Age*, *Tumor Size* and *Inv Nodes* should be considered ordered categories
   in terms of model fitting (inferred from implicitly ordered groupings in
   data) - Replace entries with lower bound of data category as integer
*  *Node Caps* should be a binary category, but 2.8% of entries are unknown.
   Using OneHotEncoder, even removing one produced feature, will therefore
   result in highly-correlated new features. Therefore remove all 8 unknown
   entries.
*  *Menopause* and *Breast Quad* are unordered categories - transform with
   OneHotEncoder
*  *Irradiat* is a binary categoy - transform with LabelEncoder
*  Left or Right breast has no clinical significance - ignore *Breast* column
"""
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
"""
# 3. Data linearity detection

At this stage, the SCV model is tested with both a linear and rbf kernel to
determine if the data is linear or non-linear. Only the SVC kernal type is
considered at this point - model optimisation will occur in the next step.
"""
print("Linearity detection...")

# Apply Grid Search (with k-value folding) to determine if the model is linear
# or non-linear
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
classifier = SVC()
parameters = {'kernel': ['linear', 'rbf']}
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
"""
# 4. Optimal model determination

Strategy at this step is to take the key non-linear classification models, and
run them with an embedded grid search technique. In the outer loop, use kPCA
to reduce the input set dimensionality between 1 and 12, and for every
determined kPCA-optimised input set, run a grid search over a wide range of
model parameters.

The models under consideration are kSVN, kNN, Decision Tree, Random Forest,
and XGBoost, with search parameters as defined in the code below.

A record of the best model result it kept in the model_search dictionary. For
every kPCA dimension, the best grid-search result for each model is compared
with the current best observed model, and if the accuracy is better, the
dimensionality and model parameters are recorded.

Following the complete execution of the search strategy, the best of all the
models is selected.
"""
print("Optimal non-linear classification model determination...")

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
"""
# 5. Model confirmation

Now take the best selected model and parameter set, and run two verificaion
tests:

1.  Run k-fold validation on the training set to confirm the grid search
    accuracy
2.  Train the model on the data, and predict the results of the test set. Use
    a confusion matrix to determine the model performance.
"""
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
