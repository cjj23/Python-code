# Artificial Neural Network

# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import re

#Return the average of a number '0-4'. Also, sort out dates:
def average2Nums(numString):
    if isinstance(numString, int):
        return numString
    elif isinstance(numString, datetime.date):
        #If day = 1, then use average of month & year (without 2000)
        if numString.day == 1:
                return (numString.month + numString.year-2000)/2
        else:
            return (numString.day + numString.month)/2
    
    nums = re.findall(r'\d+', numString)
    if len(nums) == 2:
        return (int(nums[0]) + int(nums[1]))/2   
    elif len(nums) == 0:
        return numString
    elif numString.isdigit():# == int(numString):
        return int(numString)
    else:
        return numString

#Vectorise function, so it can be applied to arrays:
f = lambda ns: average2Nums(ns)
vf = np.vectorize(f)

# Encoding categorical data:
def encodeCategoricalData(data):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer 

    #ONE HOT encodings. Do these first - will change order of columns, but can then tidy the others up later.
    #ONE HOT done on columns 1 (Menopause), 4 (Node-caps) and 7 (Breast quad)
    #1: Menopause
    ct = ColumnTransformer([("Menopause", OneHotEncoder(),[1])], remainder="passthrough") # The last arg ([0]) is the list of columns you want to transform in this step
    data = ct.fit_transform(data)
    #Remove line 1 - 'lt40' 
    data = np.delete(data, 1, 1)
    #0 = GE40, 1 =  premeno

    #5 (was 4): Node-caps
    ct = ColumnTransformer([("Node-caps", OneHotEncoder(),[5])], remainder="passthrough") # The last arg ([0]) is the list of columns you want to transform in this step
    data = ct.fit_transform(data)
    #Remove line 0 - '?' 
    data = np.delete(data, 0, 1)
    #0 = 'no', 1 =  'yes'

    #9 (was 7): Breast-quad
    ct = ColumnTransformer([("Breast-quad", OneHotEncoder(),[9])], remainder="passthrough") # The last arg ([0]) is the list of columns you want to transform in this step
    data = ct.fit_transform(data)
    #Remove line 0 - '?'
    data = np.delete(data, 0, 1)
    #0 = 'central', #1 'left-low' =, #2 = 'left-up', #3 = 'right-low', #4 = 'right-up'

    #9 (was 0): Age - convert ie 40-49 to 45 (0.5 added because range is ie 45 to <50)
    data[:, 9] = vf(data[:,9]) + 0.5
    #1: Menopause. One Hot (done above)
    #10 (was 2): Tumor size - Convert nos (0.5 added because range is ie 15 to <20)
    data[:, 10] = vf(data[:,10]) + 0.5
    #11 (was 3): inv-nodes - Convert nos
    data[:, 11] = vf(data[:,11])
    #4: Node-caps. One Hot (Yes/No, but also '?)'
    #12 (was 5): Deg-malig. Can stay as a number
    #13 (was 6): Breast - L/R, so LabelEncoder
    labelencoder_X_13 = LabelEncoder()
    data[:, 13] = labelencoder_X_13.fit_transform(data[:, 13])
    #7: Breast-quad. (LU, RU, LL, RL, Cent, ?) ONE HOT
    #14 (was 8): Irradiat - Yes/No, so LabelEncoder
    labelencoder_X_14 = LabelEncoder()
    data[:, 14] = labelencoder_X_14.fit_transform(data[:, 14])

    #COMPLETE LIST OF COLUMNS:
    #0-4: Breast-quad: 0=central, 1=left-low, 2=left-up, 3=right-low, 4=right-up (none=?)
    #5-6: Node-caps:   5=no, 6=yes (neither=?)
    #7-8: Menopause:   7=GE40, 8=premeno (neither=lt40)
    #9: Age (average of range)
    #10: Tumor size (average of range)
    #11: Inv nodes (average of range)
    #12: Deg malig (as a number)
    #13: Breast L or R (0=L, 1=R)
    #14: Irradiat? (0=no, 1=yes)
    
    return data

# Importing the dataset

#A: COLAB - Run following files:
#from google.colab import files
#import io
#import pandas as pd   
#uploaded = files.upload()
#data = io.BytesIO(uploaded['breast-cancer.xls']) 
#dataset = pd.read_excel(data , sheet_name = 'breast-cancer', header = 0, skiprows = 2)

#B: Not in COLAB - use the below line to load the file directly:
dataset = pd.read_excel('breast-cancer.xls', 'breast-cancer')

X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 9].values
#Encode using the function above:
X = encodeCategoricalData(X)

#Apply Labelencoder to y (Class - recurrance or not):
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#0=No recurrence, 1=Recurrence

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#-----------------------------------------------------------------
# Part 2 - Make the ANN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
#Sequence of layers
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
classifier.add(Dropout(rate = 0.3))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.3))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'RMSprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 4, epochs = 100)


#-----------------------------------------------------------------
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy_test = (cm[0][0] + cm[1][1])/sum(sum(cm))
print ("\n------------")
print ("End accuracy = ", accuracy_test)
print ("---------------")

#Predict if patient will have recurrence:
 #COMPLETE LIST OF COLUMNS:
    #0-4: Breast-quad: 0=central, 1=left-low, 2=left-up, 3=right-low, 4=right-up (none=?)
    #5-6: Node-caps:   5=no, 6=yes (neither=?)
    #7-8: Menopause:   7=GE40, 8=premeno (neither=lt40)
    #9: Age (average of range)
    #10: Tumor size (average of range)
    #11: Inv nodes (average of range)
    #12: Deg malig (as a number)
    #13: Breast L or R (0=L, 1=R)
    #14: Irradiat? (0=no, 1=yes)

#Try patient with any other conditions:
spec_patient = [0, 0, 1, 0, 0,
                0, 1,
                0, 1,
                45,
                17.5,
                1,
                3,
                1,
                0]

spec_patient = np.array([spec_patient])           
spec_patient_transfomed = sc.transform(spec_patient)
patient_pred = classifier.predict(spec_patient_transfomed)
print ("Prediction for patient", spec_patient, " = ")
print ("\nPatient prediction = ", patient_pred)
#0=No recurrence, 1=Recurrence

#print (X[0])
#Not required, but useful for testing later:
#X_transform = sc.transform(X)
#y_all = classifier.predict(X_transform)
#print (y_all[0])
