import streamlit as st # For using streamlit

from PIL import Image # For displaying images

import scipy # import scipy

st.title('Title') # Set Title

st.subheader('This is a subheading') # set a sub header

image = Image.open('streamlit.jpg') # Loading Image

st.image(image=image, use_column_width=True) # To display using streamlit

st.write('Writing text') # Writing text

st.markdown('Writing Markdown') # Writing Markdown

st.success('Success Message Display') # Success Message Display

st.info('Display Info') # Display Info

st.warning('Display Warning') # Display Warning

st.error('Show Error') # Show Error

st.help(range) # Help for range function, change function name

import numpy as np # import Numpy

import pandas as pd # import pandas

df = np.random.rand(10,20) # Creating random number with numpy

st.dataframe(df) # Dataframe with Streamlit

st.text('---------'*100) # Text Line for seperation

df2 = pd.DataFrame(np.random.randn(10,20), columns= [i for i in range(21,41)]) # Creating random number dataframe with pandas

st.dataframe(df2.style.highlight_max(axis=1)) # Dataframe with Streamlit, also higghlights max value in column

st.text('---------'*100) # Text Line for seperation

chart_data = pd.DataFrame(np.random.randn(20,3), columns=['a','b','c']) # Creating random number dataframe with pandas

st.line_chart(chart_data) # Display Line Chart

st.text('---------'*100) # Text Line for seperation

st.area_chart(chart_data) # Display area chart

st.text('---------'*100) # Text Line for seperation

st.bar_chart(chart_data) # Display bar chart

import matplotlib.pyplot as plt # import matplotlib

# st.set_option('deprecation.showPyplotGlobalUse', False) # To allow us to use st.pyplot without any issues

arr = np.random.normal(1,1,size=100) # Creating a random array

plt.hist(arr,bins=20) # Loading using matplotlib.pyplot

st.pyplot() # Display the matplotlib plot

st.text('---------'*100) # Text Line for seperation

import plotly # importing plotly

import plotly.figure_factory as ff # import igure factory

x1 = np.random.randn(200)-2;x2 = np.random.randn(200); x3 = np.random.randn(200)-2 # adding values to x1,x2,x3

data = [x1,x2,x3] # Adding figure values

group_labels = ['Group1','Group2','Group3'] # giving labels

fig = ff.create_distplot(data, group_labels, bin_size=[.25,.25,5]) # Creating displot from plotly

st.plotly_chart(fig, use_container_width=True) # Plot distribution plot using plotly

st.text('---------'*100) # Text Line for seperation

df = pd.DataFrame(np.random.randn(100,2)/[50,50]+[37.76,-122.4], columns=['lat','lon']) # Creating random number dataframe

st.map(df) # Create map

st.text('---------'*100) # Text Line for seperation

if st.button('Button In'): # Creating Button

    st.write('Button In Selected') # Display after selecting Button

else : # Creating Button

    st.button('Button Out') # Creating Button

    st.write('Button Out Selected')  # Display after selecting Button

st.text('---------'*100) # Text Line for seperation

genre = st.radio('Select the radio option', ('Option1','Option2')) # Creates a radio button

if genre == 'Option1': # Selects the option in radio button

    st.write('Comedy is selected') # text displays

st.text('---------'*100) # Text Line for seperation

option = st.selectbox('This is a select box', ('Option1','Option2','Option3')) # Select Box

st.write('The box is selected',option) # text displays with option

st.text('---------'*100) # Text Line for seperation

option = st.multiselect('This is a select box', ('Option1','Option2','Option3')) # Multiple options can be select

st.write('The box is selected',option) # text displays with option

st.text('---------'*100) # Text Line for seperation

age = st.slider('Slide to select option',0,150,10) # Slide bar with range from 0 - 150 with default as 10

st.write('The option is selected',age) # text displays with option

st.text('---------'*100) # Text Line for seperation

values = st.slider('Select of range of values',0,200,(15,80)) # Slide between a range 0 - 200 with default at 15 as low and 80 as high

st.write(f'you selected a range between {values}') # text displays with option

st.text('---------'*100) # Text Line for seperation

values = st.number_input('Input from user') # Input from user

st.write(f'Input number is {values}') # text displays with option

st.text('---------'*100) # Text Line for seperation
st.text('---------'*100) # Text Line for seperation

upload_file = st.file_uploader('Select options user can upload', type = 'csv') # User can upload files/ images / videos from their end

if upload_file is not None: # Condition

    data = pd.read_csv(upload_file) # Upload data

    st.write(data) # Show data

    st.success('csv File uploaded') # Display success message

else:

    st.error('File is empty') # Display error

st.text('---------'*100) # Text Line for seperation

second_upload_file = st.file_uploader('Select options user can upload', type = 'tsv') # User can upload files/ images / videos from their end

if second_upload_file is not None: # Condition

    data = pd.read_csv(second_upload_file) # Upload data

    st.write(data) # Show data

    st.success('csv File uploaded') # Display success message

else:

    st.error('File is empty, Kindly upload a file') # Display error

st.text('---------' * 100)  # Text Line for seperation

col = st.color_picker('preferred color','#00FFAA') # add color picker

st.write(f'The color is {col}') # Text message display

st.text('---------' * 100)  # Text Line for seperation

opt = st.sidebar.selectbox('Sidebar',('1','2','3')) # set a sidebar

st.write(f'The option is {opt}') # Text message display

st.text('---------' * 100)  # Text Line for seperation

import time # import time

bar = st.progress(0) # progress bar

for percent_complete in range(100):
    time.sleep(0.1)
    bar.progress(percent_complete + 1)

st.text('---------' * 100)  # Text Line for seperation

with st.spinner('Spinning'): # spin progress bar
    time.sleep(5)
st.success('Loading / Spinning Completed')

st.text('---------' * 100)  # Text Line for seperation

st.balloons() # Baloons

