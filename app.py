import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from PIL import Image
model = XGBRegressor()


st.title('Zameen Price Prediction')
st.sidebar.header('Zameen Data')
image = Image.open('zameen_logo.png')
st.image(image, '')

# FUNCTION
def user_report():
  property_id = st.sidebar.slider('property_id', 50,1000, 1 )
  location_id = st.sidebar.slider('location_id', 0,100, 1 )
  property_type = st.sidebar.slider('property_type', 0,30, 1 )
  location = st.sidebar.slider('location', 0,10, 1 )
  city = st.sidebar.slider('city', 0,3, 1 )
  latitude = st.sidebar.slider('latitude', 2000,2020, 2000)
  longitude = st.sidebar.slider('longitude', 1,10, 1)
  baths = st.sidebar.slider('baths', 1,30, 1)
  purpose = st.sidebar.slider('purpose', 1,30, 1)
  bedrooms = st.sidebar.slider('bedrooms', 1,30, 1)
  date_added = st.sidebar.slider('date_added', 1,30, 1)
  area_type = st.sidebar.slider('area_type', 1,30, 1)
  area_size = st.sidebar.slider('area_size', 1,30, 1)
  area_category = st.sidebar.slider('area_category', 1,30, 1)
  year_added = st.sidebar.slider('year_added', 1,30, 1)
#   price = st.sidebar.slider('price', 1,30, 1)
  user_report_data = {
      'property_id':property_id,
      'location_id':location_id,
      'property_type':property_type,
      'location':location,
      'city':city,
      'latitude':latitude,
      'longitude':longitude,
      'baths':baths,
      'purpose':purpose,
    #   'bedrooms':bedrooms,
      'date_added':date_added,
    #   'Area Type': area_type,
      'Area Size': area_size,
      'Area Category': area_category,
      'year_added': year_added
    #   'price': price
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data
df = pd.read_csv('data.csv')
y = df.price
X = df.drop('price', axis=1)
user_data = user_report()
st.header('Zameen Properties')
st.write(user_data)
model = model.fit(X, y)
salary = model.predict(user_data)
st.subheader('Property Price')
st.subheader(salary)