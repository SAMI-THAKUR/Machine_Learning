import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import altair as alt #pip install altair

data = pd.DataFrame(
    np.random.randn(100,3),
    columns=['a','b','c']
)

chart = alt.Chart(data).mark_circle().encode(
    x = 'a',y='b',tooltip =['a','b']
)
city = pd.DataFrame({
    'awesome cities' : ['Chicago', 'Minneapolis', 'Louisville', 'Topeka'],
    'lat' : [41.868171, 44.979840,  38.257972, 39.030575],
    'lon' : [-87.667458, -93.272474, -85.765187,  -95.702548]
})

st.map(city)