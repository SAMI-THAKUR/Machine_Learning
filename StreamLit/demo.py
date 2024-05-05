import streamlit as st
import pandas as pd

st.title('Sami Thakur')
st.header('Samir Thakur')
st.subheader('Sami Thakur')
st.text('Sami Thakur')

st.markdown(""" # h1 tag
## h2 tag
### h3 tag
:moon:<br>
:sunglasses:
** bold **
_ italics _
""",True)

# --> # h1 tag
# --> ## h2 tag
# --> ### h3 tag
# --> **  bold tag **
# --> _ italics tag _

st.latex(r''' a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)''')
d ={
    "name":"Harsh",
    "language":"Python",
    "topic":"Streamlit"
} 
st.write(d)