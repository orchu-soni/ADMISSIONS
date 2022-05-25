from google.colab import files
uploaded=files.upload()
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense ,Dropout,BatchNormalization
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
df = pd.read_csv('Admission_Predict.csv')
df.head()
df=df.drop("Serial No.",axis=1)
Y=np.array(df[df.columns[-1]])
X=np.array(df.drop(df.columns[-1],axis=1))
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler =  MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
def baseline_model():
    model = Sequential()
    model.add(Dense(16, input_dim=7, activation='relu'))
    model.add(Dense(16, input_dim=7, activation='relu'))
    model.add(Dense(16, input_dim=7, activation='relu'))
    model.add(Dense(16, input_dim=7, activation='relu'))
    model.add(Dense(1))    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
estimator = KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=3, verbose=1)
estimator.fit(X_train,y_train)
prediction = estimator.predict(X_test)
print("ORIGINAL DATA")
print(y_test)
print()
print("PREDICTED DATA")
print(prediction)
from sklearn.metrics import accuracy_score
 
train_error =  np.abs(y_test - prediction)
mean_error = np.mean(train_error)
 
print("Mean Error: ",mean_error)