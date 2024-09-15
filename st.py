import streamlit as st 
import pandas as pd
from PIL import Image
import numpy as np
import io
import plotly.express as px


w1 = np.array(pd.read_csv('data/weights_and_biases/w1.csv'))
b1 = np.array(pd.read_csv('data/weights_and_biases/b1.csv'))
w2 = np.array(pd.read_csv('data/weights_and_biases/w2.csv'))
b2 = np.array(pd.read_csv('data/weights_and_biases/b2.csv'))
w3 = np.array(pd.read_csv('data/weights_and_biases/w3.csv'))
b3 = np.array(pd.read_csv('data/weights_and_biases/b3.csv'))
w1 = np.delete(w1,0,axis=1)
b1 = np.delete(b1,0,axis=1)
w2 = np.delete(w2,0,axis=1)
b2 = np.delete(b2,0,axis=1)
w3 = np.delete(w3,0,axis=1)
b3 = np.delete(b3,0,axis=1)


def softmax(z):
    return np.exp(z) / sum(np.exp(z))

def prediction(l):
    return np.argmax(l, 0)

def forward(w1,b1,w2,b2,w3,b3,x):
    z1 = w1.dot(x.T) + b1
    a1 = np.maximum(0,z1)
    z2 = w2.dot(a1) + b2
    a2 = np.maximum(0, z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return prediction(a3)

st.header(body= "nu:green[MNIST]", divider= 'green')
with st.container(border=True):
    st.subheader("upload a picture of a handwritten below")
    uploaded_file = st.file_uploader(label='')
    if uploaded_file is not None:
        image = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image))
        image = image.resize((28, 28), 1)
        img = np.array(image)
        img = np.dot(img[...,:3],[0.2989, 0.5870, 0.1140])
        fig = px.imshow(img, aspect='equal')
        st.plotly_chart(fig, use_container_width=True)


        img = img.reshape(1, 28*28) / 255.

        st.subheader(f'the number in the picture is guessed to be -{forward(w1,b1,w2,b2,w3,b3,img)[0]}-')
