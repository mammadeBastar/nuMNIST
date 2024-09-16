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
    uploaded_file = st.file_uploader(label='upload', label_visibility='hidden')
    dark_bg = st.toggle("my picture has a dark background")
    if uploaded_file is not None:
        denoise = st.slider("noise filter:", 0, 200, 20)
        zoom = st.slider("zoom", 0, 100)
        image = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image))
        w, h = image.size
        image = image.crop((zoom * w / 200, zoom*h / 200, w - (zoom * w/200), w - (zoom * h / 200)))
        image = image.resize((28, 28), 1)
        img = np.array(image)
        if not dark_bg:
            img = 255 - img
        img = np.dot(img[...,:3],[0.2989, 0.5870, 0.1140])
        for i in range(28):
            for j in range(28):
                if img[i, j] < denoise:
                    img[i, j] = 0
                    continue
                r = 0
                for x in [-1,0 ,1]:
                    for y in [-1, 0, 1]:
                        try:
                            if img[i+x, j+y] < 1:
                                r += 1
                        except: pass
                if r >= 7:
                    img[i, j] = 0
                if img[i, j] < 100 + denoise:
                    img[i, j] = 100 + denoise
        fig = px.imshow(img, aspect='equal')
        st.plotly_chart(fig, use_container_width=True)


        img = img.reshape(1, 28*28) / 255.

        st.subheader(f'the number in the picture is guessed to be -{forward(w1,b1,w2,b2,w3,b3,img)[0]}-')
