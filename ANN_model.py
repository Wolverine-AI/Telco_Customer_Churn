#part 1
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

# Part 2 - Now let's make the ANN!

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Initialising the ANN
classifier = Sequential()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0) 
#classifier.summary()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#classifier.summary()
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#classifier.compile(optimizer = SGD(), loss = 'binary_crossentropy', metrics = ['accuracy'])

#classifier.summary()
# Fitting the ANN to the Training set
#classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
history = classifier.fit(X_train, y_train, batch_size =10, epochs = 20)

classifier.summary()    
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
    

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

score = classifier.evaluate(X_test,y_test)
print(score)
print('loss = ', score[0])
print('acc = ', score[1])

#Adding more layers
classifier = Sequential()
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the third hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#classifier.summary()
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 30)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

score = classifier.evaluate(X_test,y_test)
print(score)
print('loss = ', score[0])
print('acc = ', score[1])


# Neurons
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

score = classifier.evaluate(X_test,y_test)
print(score)
print('loss = ', score[0])
print('acc = ', score[1])

