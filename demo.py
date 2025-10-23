import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

st.title('Lotus-Gold Consulting')
st.markdown('---')
st.image('11.jpg')
st.markdown('---')
st.subheader('Decision Tree Model')

df = pd.read_csv('music.csv')

with st.expander('This is the music dataset'):
    st.dataframe(df)
    
x = df.drop(columns=['genre'])
y = df['genre']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = DecisionTreeClassifier()
model.fit(x,y)

predictions = model.predict(x_test)

score = accuracy_score(y_test,predictions)

score

st.markdown("""
            <html>
                This Data Driven application 
                allows you to select your age 
                and your gender then it tells 
                you which genre of music is 
                suitable for you
            </html>
            """, unsafe_allow_html=True)

st.subheader('Select Your Age & Determin Your Genre Of Music')
age = st.slider('Select Your Age Range',10,70,18)
gender = st.selectbox('Pick Your Gender',options=['Male','Female'])
gender_code = 1 if gender == 'Male' else 0

if st.button('Click'):
    prediction = model.predict([[age,gender_code]])
    st.success(f'Correct: **{prediction[0]}**')

st.caption('Copyright: 2025 Lotus-Gold Consulting')

    
with st.sidebar:
    st.header('Lotus-Gold')
    st.date_input('Select You Date Of Visit')
    st.time_input('Select Time')
   