import pandas as pd 
import numpy as np 
#import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#import plotly as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


df=pd.read_csv('Cleaned-Data.csv')
df.head()

df=df.rename(columns={'Gender_Male':'Gender','Contact_No':'Contact','Dry-Cough':'Dry_Cough','Difficulty-in-Breathing':'Difficulty_in_Breathing',
                      'Sore-Throat':'Sore_Throat','None_Symptomps':'Symptomps','Nasal-Congestion':'Nasal_Congestion','Runny-Nose':'Runny_Nose',
                     'Age_0_9':'Age_0_9','Age_10-19':'Age_10_19','Age_20-24':'Age_20_24','Age_25-59':'Age_25_59','Age_60+':'Age_60'})
df=df.drop(columns=['Gender_Female','Contact_Yes','Contact_Dont-Know','Severity_Mild', 'Severity_Moderate', 'Severity_None',
       'Severity_Severe', 'Contact', 'Country'])

from imblearn.over_sampling import SMOTE
sm=SMOTE()

x=df.drop(columns=['None_Sympton'])
y=df['None_Sympton']

x_sm,y_sm=sm.fit_resample(x,y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x_sm,y_sm,test_size=0.2,random_state=0)

pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('pca1',PCA(n_components=2)),
                     ('lr_classifier',LogisticRegression(random_state=0))])



pipeline_dt=Pipeline([('scalar2',StandardScaler()),
                     ('pca2',PCA(n_components=2)),
                     ('dt_classifier',DecisionTreeClassifier())])

pipeline_randomforest=Pipeline([('scalar3',StandardScaler()),
                     ('pca3',PCA(n_components=2)),
                     ('rf_classifier',RandomForestClassifier())])

pipelines = [pipeline_lr, pipeline_dt, pipeline_randomforest]

best_accuracy=0.0
best_classifier=0
best_pipeline=""

pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest'}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(xtrain, ytrain)
    


for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(xtest,ytest)))


for i,model in enumerate(pipelines):
    if model.score(xtest,ytest)>best_accuracy:
        best_accuracy=model.score(xtest,ytest)
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))




# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]



# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)

model = RandomForestClassifier()

rf_RandomGrid = RandomizedSearchCV(estimator = model, param_distributions = param_grid, cv = 10, verbose=2, n_jobs = 4)

rf_RandomGrid.fit(xtrain, ytrain)

rf_RandomGrid.best_params_

ypred=rf_RandomGrid.predict(xtest)


print("Accuracy is :",accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))
cm=confusion_matrix(ytest,ypred)
sns.heatmap(cm,annot=True)

import pickle

pickle.dump(rf_RandomGrid,open('pickel_model.pkl','wb'))

model=pickle.load(open('pickel_model.pkl','rb'))


    