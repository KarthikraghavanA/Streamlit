# Loading Libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mlt
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# Set Title
st.title('Web App for Machine Learning')
img = Image.open('AIMLDL.jpg')
st.image(img, use_column_width=True)

def main():
    activities = ['EDA', 'Visualization', 'Preprocessing','Modeling', 'About us']
    option = st.sidebar.selectbox('Select the activity : ', activities)

    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')
        data = st.file_uploader('Upload dataset', type=['csv','tsv','xlsx','txt','json'])

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))
            st.success('Data Successfully loaded')

            if st.checkbox('Display shape'):
                st.write(f'Shape of the datatset:{df.shape}')
            if st.checkbox('Display columns of the dataset'):
                st.write(f'Columns of the datatset:{df.columns}')
            if st.checkbox('Select Multiple Columns'):
                selected_columns = st.multiselect(f'Select preferred columns:',df.columns)
                st.dataframe(df[selected_columns])
            st.warning('SELECT THE COLUMNS REQUIRED TO PROCESS SUMMARY')
            if st.checkbox('Display summary'):
                st.write(df[selected_columns].describe())
            if st.checkbox('Display Null values count'):
                st.write(df.isnull().sum())
            st.info('Correlation will only process for numerical values, not for categorical values')
            if st.checkbox('Display Correlation'):
                st.write(df.corr())

    elif option == 'Visualization':
        st.subheader('Data Visualization')
        data = st.file_uploader('Upload dataset', type=['csv', 'tsv', 'xlsx', 'txt', 'json'])

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))
            st.success('Data Successfully loaded')

            st.set_option('deprecation.showPyplotGlobalUse', False)

            if st.checkbox('Select Multiple columns to plot'):
                selected_columns = st.multiselect('Select the required columns', df.columns)
                st.dataframe(df[selected_columns])

            if st.checkbox('Display Heatmap for selected columns'):
                st.write(sns.heatmap(df[selected_columns].corr(), annot=True))
                st.pyplot()

            if st.checkbox('Display Pairplot for selected columns'):
                st.write(sns.pairplot(df[selected_columns], diag_kind='kde'))
                st.pyplot()

            st.info('Pie Chart will only process for categorical values, not for numerical values')
            if st.checkbox('Display Pie Chart'):
                all_columns = df.columns.to_list()
                pie_columns = st.selectbox('Select column to display', all_columns)
                plt.pie(df[pie_columns].value_counts().values, labels = df[pie_columns].value_counts().index, autopct='%.2f%%')
                st.pyplot()

    elif option == 'Preprocessing':
        st.subheader('Data Preprocessing')

        data = st.file_uploader('Upload dataset', type=['csv', 'tsv', 'xlsx', 'txt', 'json'])

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))
            st.success('Data Successfully loaded')

            cols = st.multiselect('Select the columns to drop', df.columns)
            df.drop(columns=cols,axis = 1, inplace = True)
            st.dataframe(df)



    elif option == 'Modeling':
        st.subheader('Modeling')

        data = st.file_uploader('Upload dataset', type=['csv', 'tsv', 'xlsx', 'txt', 'json'])

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))
            st.success('Data Successfully loaded')

            if st.checkbox('Select the target'):
                target = st.selectbox('Select the required columns', df.columns)

                y = df[target]
                X = df.drop(target, axis = 1)

                st.write('Target distribution', y.value_counts())

                seed = st.sidebar.slider('Seed', 1, 200)
                classifier = st.sidebar.selectbox('Select the classifier:',('KNN','SVM', 'Logistic Regression', 'Naive Bayes', 'Decision Tree'))

                def add_parameters(name_of_clf):
                    param = {}
                    if name_of_clf == 'SVM':
                        C = st.sidebar.slider('C',0.01,15.0)
                        gamma = st.sidebar.slider('Gamma', 0.01, 15.0)
                        param['C'] = C
                        param['Gamma'] = gamma
                    if name_of_clf == 'KNN':
                        k = st.sidebar.slider('K',1,50)
                        param['K'] = k
                    if name_of_clf == 'Decision Tree':
                        d = st.sidebar.slider('Max Depth',1,10)
                        param['max_depth'] = d
                    return param

                params = add_parameters(classifier)

                def get_classifier(name_of_clf, param):
                    clf = None
                    if name_of_clf == 'SVM':
                        clf = SVC(C=param['C'], gamma=param['Gamma'])
                    if name_of_clf == 'KNN':
                        clf = KNeighborsClassifier(n_neighbors=param['K'])
                    if name_of_clf == 'Decision Tree':
                        clf = DecisionTreeClassifier(max_depth=param['max_depth'])
                    return clf

                classifier = get_classifier(classifier, params)

                random_state = st.sidebar.slider('Select Random State', 0, 150, 0)
                test_size = st.sidebar.slider('Select Test set size', 0.01, 1.0)
                xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=random_state)

                classifier.fit(xtrain, ytrain)

                ypred = classifier.predict(xtest)

                st.write(f'Accuracy with train data {classifier.score(xtrain, ytrain)}')
                st.write(f'Accuracy with test data {classifier.score(xtest, ytest)}')

                st.write('Predictions : ', ypred)


    else:
        st.subheader('About')
        st.write('I am Karthikraghavan, I am a Full Stack Data Scientist')
        st.write('The interactive web app is for showcasing my skills on Machine Learning projects')
        st.balloons()

if __name__ == '__main__':
    main()