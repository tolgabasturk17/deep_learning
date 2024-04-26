# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.

# load data set
x_l = np.load('../data/Sign-language-digits-dataset/X.npy')
Y_l = np.load('../data/Sign-language-digits-dataset/Y.npy')
#img_size = 64
#plt.subplot(1, 2, 1)
#plt.imshow(x_l[260].reshape(img_size, img_size))
#plt.axis('off')
#plt.subplot(1, 2, 2)
#plt.imshow(x_l[900].reshape(img_size, img_size))
#plt.axis('off')

# Join a sequence of arrays along an row axis.
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

# Then lets create x_train, y_train, x_test, y_test arrays
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

#Now we have 3 dimensional input array (X) so we need to make it flatten (2D) in order to use
# as input for our first deep learning model.
# Our label array (Y) is already flatten(2D) so we leave it like that.
# Lets flatten X array(images array).

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)

#As you can see, we have 348 images and each image has 4096 pixels in image train array.
# Also, we have 62 images and each image has 4096 pixels in image test array.
# Then lets take transpose. You can say that WHYY, actually there is no technical answer.
# I just write the code(code that you will see oncoming parts) according to it :)

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.

# load data set
x_l = np.load('../data/Sign-language-digits-dataset/X.npy')
Y_l = np.load('../data/Sign-language-digits-dataset/Y.npy')
#img_size = 64
#plt.subplot(1, 2, 1)
#plt.imshow(x_l[260].reshape(img_size, img_size))
#plt.axis('off')
#plt.subplot(1, 2, 2)
#plt.imshow(x_l[900].reshape(img_size, img_size))
#plt.axis('off')

# Join a sequence of arrays along an row axis.
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

# Then lets create x_train, y_train, x_test, y_test arrays
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

#Now we have 3 dimensional input array (X) so we need to make it flatten (2D) in order to use
# as input for our first deep learning model.
# Our label array (Y) is already flatten(2D) so we leave it like that.
# Lets flatten X array(images array).

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)

#As you can see, we have 348 images and each image has 4096 pixels in image train array.
# Also, we have 62 images and each image has 4096 pixels in image test array.
# Then lets take transpose. You can say that WHYY, actually there is no technical answer.
# I just write the code(code that you will see oncoming parts) according to it :)

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential # initialize neural network library
from tensorflow.keras.layers import Dense # build our layers library
from scikeras.wrappers import KerasClassifier
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer="uniform", activation="relu", input_dim= x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer="uniform", activation="relu", ))
    classifier.add(Dense(units = 1, kernel_initializer="uniform", activation="sigmoid"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, epochs=100)
accuracies = cross_val_score(estimator=classifier, X= x_train, y = y_train, cv=3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean" + str(mean))
print("Accuracy variance" + str(variance))

