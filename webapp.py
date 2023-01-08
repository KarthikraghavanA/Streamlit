# Creating Web App using Streamlit

# Loading Libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mlt
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set Title
st.title('Web App for Machine Learning')

image = Image.open('AIMLDL.jpg')  # Loading Image
st.image(image=image, use_column_width=True)  #To display using streamlit

st.subheader('Simple Web App using Streamlit')  # Set a sub header
st.write('Let us explore different classifiers and datasets')

dataset_name = st.sidebar.selectbox('Select the dataset:', ('Breast Cancer', 'Iris', 'Wine'))
classifier_name = st.sidebar.selectbox('Select the classifier:', ('SVM', 'KNN'))

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()

    x = data.data
    y = data.target

    return x,y

x, y = get_dataset(dataset_name)
st.dataframe(x)
st.write(f'Shape of the dataset:{x.shape}')
st.write(f'Unique target variables:{len(np.unique(y))}')

plt.figure()
sns.boxplot(data=x, orient='h')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

plt.hist(x)
st.pyplot()

# Building the classifer

def add_parameter(clf_name):
    params = dict()
    if clf_name == 'SVM':
        c = st.sidebar.slider('C', 0.01, 10.0)
        gamma = st.sidebar.slider('gamma', 0.01, 10.0)
        params['C'] = c
        params['gamma'] = gamma
    else:
        k = st.sidebar.slider('Number of Neigbbors', 1, 50, 1)
        params['K'] = k

    return params

params = add_parameter(classifier_name)

# Accessing our classifier

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C = params['C'], gamma=params['gamma'])
    else:
        clf = KNeighborsClassifier(n_neighbors=params['K'])

    return clf

classifier = get_classifier(classifier_name ,params)

random_state = st.sidebar.slider('Select Random State', 0, 150, 0)
test_size = st.sidebar.slider('Select Test set size', +0.01, +1.0)
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=test_size,random_state=random_state)

classifier.fit(xtrain, ytrain)

ypred = classifier.predict(xtest)

st.write(f'Accuracy with train data {classifier.score(xtrain, ytrain)}')
st.write(f'Accuracy with test data {classifier.score(xtest, ytest)}')

st.write('Predictions : ', ypred)