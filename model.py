import pandas as pd
import numpy as np

churn_df = pd.read_csv('/home/gourav/Desktop/DataScience/portfolio_projects/churn_prediction/clean_df.csv')

for i in churn_df[['partner', 'dependents','phoneService', 'multipleLines','onlineSecurity', 'onlineBackup', 'deviceProtection', 'techSupport',
           'streamingTV', 'streamingMovies','paperlessBilling','churn']]:
         churn_df[i] =  churn_df[i].map({'No':0,'Yes':1})
        

#relevent columns
df_model = churn_df[['gender', 'seniorCitizen', 'partner', 'dependents',
       'tenure', 'phoneService', 'multipleLines', 'internetService',
       'onlineSecurity', 'onlineBackup', 'deviceProtection', 'techSupport',
       'streamingTV', 'streamingMovies', 'contract', 'paperlessBilling',
       'monthlyCharges', 'totalCharges','paymentMethod','churn']]

#get dummy Data
df_dum = pd.get_dummies(df_model)

#train test split
X = df_dum.drop('churn',axis=1)
y = df_dum.churn.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#multiple linear regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,accuracy_score
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

y_pred  = y_pred>0.5
confusion_matrix(y_test,y_pred) 
accuracy_score(y_test,y_pred)


#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=500,random_state=0)
regr.fit(X_train, y_train)    
y_pred = regr.predict(X_test)

y_pred  = y_pred>0.5
confusion_matrix(y_test,y_pred) 
accuracy_score(y_test,y_pred)

#Fitting Xgboost to Training set
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot

# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()

# fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



# Fit model using each importance as a threshold
thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
