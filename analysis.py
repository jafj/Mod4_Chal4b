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
X[:,0] = [int(re.match('([0-9])+', x).group(0)) for x in X[:,0]]
X[:,2] = [int(re.match('([0-9])+', x).group(0)) for x in X[:,2]]
X[:,3] = [int(re.match('([0-9])+', x).group(0)) for x in X[:,3]]

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
# Feature scaling
#
# ------------------------------------------------------------
print("Feature scaling...")

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

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
grid_search = grid_search.fit(X_scaled, y)
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
print("Optimal model determination...")

# The above shows this is a non-linear model. Let's try a selection of models
# with a variety of trial kPCA-driven feature decompositions. As the data are
# non-linear and we desire classification, look at K-NN, Random Forest and
# Kernel SVN. Note using kPCA with rbf kernel as we believe the data is
# non-linear

# Helper method to run a gridsearch for a given estimator and parameter set
def run_grid_search(x_in, y_in, classifier, parameters, feature_dim):
   highest_accuracy = 0
   best_parameters = None
   grid_search = GridSearchCV(estimator = classifier,
                              param_grid = parameters,
                              scoring = 'accuracy',
                              cv = 10,
                              n_jobs = -1)
   grid_search = grid_search.fit(x_in, y_in)
   run_accuracy = grid_search.best_score_
   run_parameters = grid_search.best_params_
   if run_accuracy > highest_accuracy:
      highest_accuracy = run_accuracy
      best_parameters = run_parameters
   return {'feature-dim': feature_dim,
           'accuracy': highest_accuracy,
           'params': best_parameters}

# All classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
classifiers = {'kSVN': SVC(),
               'kNN': KNeighborsClassifier(),
               'rand-forest': RandomForestClassifier()}

# Record of search results
model_search = {}

# Search 2 to 12 feature dimensions
from sklearn.decomposition import KernelPCA
kSVN_best_accuracy = 0
kNN_best_accuracy = 0
rand_forest_best_accuracy = 0
for i in range(2, 13):
   # Run kPCA
   if i == 12:
      X_transformed = X_scaled
   else:
      transformer = KernelPCA(n_components=i, kernel='rbf')
      X_transformed = transformer.fit_transform(X_scaled)
   
   # kSVN
   parameters = [{'C': [0.5, 0.75, 1, 1.25, 1.5, 1.75], 'kernel': ['rbf'],
                  'gamma': [0.05, 0.075, 0.1, 0.15, 0.175]}]
   model_search['kSVN'] = run_grid_search(X_transformed, y,
                                          classifiers['kSVN'],
                                          parameters, i)
   
   # kNN
   parameters = [{'n_neighbors': [5, 6, 7, 8, 9, 10, 11, 12],
                  'weights': ['uniform', 'distance'],
                  'metric': ['minkowski'],
                  'p': [2]}]
   model_search['kNN'] = run_grid_search(X_transformed, y,
                                         classifiers['kNN'],
                                         parameters, i)
   
   # kNN
   parameters = [{'n_estimators': [10, 25, 50, 75, 100],
                  'criterion': ['gini', 'entropy']}]
   model_search['rand_forest'] = run_grid_search(X_transformed, y,
                                                 classifiers['rand-forest'],
                                                 parameters, i)

best_model = None
best_accuracy = 0
for k in model_search:
   if model_search[k]['accuracy'] > best_accuracy:
      best_accuracy = model_search[k]['accuracy']
      best_model = k
print('** The best model determined is {} with:'.format(best_model))
print('**   kPCA dimensions: {}'.format(model_search[best_model]['feature-dim']))
print('**   Parameters: {}'.format(str(model_search[best_model]['params'])))

"""
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

print("Accuracy: {}".format(np.sum(np.diag(cm)) / np.sum(cm)))
"""