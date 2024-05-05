import streamlit as st
import pandas as pd
import numpy as np
import time


a = [1,2,3,4,5,6,7,8]
n = np.array(a)
nd = n.reshape((2,4))
dic = {
    "name":["harsh","Gupta"],
    "age":[21,32],
    "city":["noida","delhi"]
}
ad = pd.read_csv("Advertising.csv")

st.dataframe(ad , width=700 , height=100)
st.table(n)
st.header("JSON")
st.json(dic)
st.write(dic)

@st.cache
def ret_time(a):
    time.sleep(5)
    return time.time()

if st.checkbox("1"):
    st.write(ret_time(1))

if st.checkbox("2"):
    st.write(ret_time(2))