
import datetime
import numpy as np #Imports numpy package of Python3
import matplotlib.pyplot as plt #Imports matplotlib library of Python  to plot images..
from skimage import io, img_as_float #io module of skimage is to enable reading and writing of images
                                    #img_as_float convert  an image to floating point format , with values in [0,1]
import os    #os module in Python give functions to interact with the operating system
from musica_1_streamlit_version import *
from os import system
import cv2
import streamlit as st

st.markdown("<h1 style='text-align: center; color: red;'>Open Ended Lab Project : Image Enhancement</h1>",
            unsafe_allow_html=True)
st.sidebar.markdown("By  ")
st.sidebar.markdown("**_Asif M.S._**")
st.sidebar.markdown("121901007")

st.sidebar.markdown("Under the Supervision of :")
st.sidebar.markdown("**_Dr. Mahesh R. Panicker_**")

file = st.file_uploader("Please upload an image ")
img_o = img_as_float(io.imread(file, as_gray=True))

img_o_fn = np.log( 1 + img_o)

L = 7 # Number of Levels for Laplacian Pyramid

a = np.full(L, 1)  # Creates a 1D array of length L and filled with the value 1
p = np.full(L, 1)  # Creates a 1D array of length L and filled with the value 1
p = np.zeros((L,1))
M = 1.0
xc = 0.01 * M  # Lower Intensity Limit for Gamma Correction # st.sidebar.slider('x_c', 0.0000, 0.0100, 0.0010,step=0.0001)#0.01 * M  # Lower Intensity Limit for Gamma Correction
params = {'M': M,'a': a,'p': p,'xc': xc }

musica_img = entire_musica(img_o_fn,L,params)

musica_ln_inverse = np.exp(musica_img) - 1

musica_second = entire_musica(musica_ln_inverse,L,params)

musica_final = np.log(1 + musica_second)

img_e = musica_final

img_e = (img_e*255).astype(np.uint8)
img_denoised = img_e

#Reference SNR Calculation
p = np.zeros((L,1))
params = {'M': M,'a': a,'p': p,'xc': xc}
ref_img = entire_musica(img_o,L,params)
snr_ref = signaltonoise(ref_img,axis=None)

h = st.sidebar.slider('Filter Strength for Denoising', 1, 100, 20)

#Denosing increases the SNR
cv2.fastNlMeansDenoising(img_denoised ,img_denoised,h,7,21) #First parameter is the source image , second parameter is the destination image

img_denoised = (img_denoised - np.min(img_denoised))/np.ptp(img_denoised)

r =  st.sidebar.slider('r', 0.0, 1.0, 0.0)

img_final = img_denoised + r*img_o # The final image is a result of weighted addition of the denoised image and the original image
img_final = (img_final - np.min(img_final))/np.ptp(img_final)

imageOrgLocation = st.empty()
imageRefLocation = st.empty()
imageEnhDenoiseLocation = st.empty()
imageFinal = st.empty()

imageOrgLocation.image(img_o, caption='Original Image (SNR = '+str(signaltonoise(img_o,axis = None))+" ) ", use_column_width=True)
imageRefLocation.image(ref_img, caption='Normal MUSICA Enhanced Image (SNR = '+str(signaltonoise(ref_img,axis = None))+" ) ", use_column_width=True)
imageEnhDenoiseLocation .image(img_denoised ,caption = "Denoised Enhanced Image (SNR = "+str(signaltonoise(img_denoised,axis = None))+" ) ", use_column_width=True)
imageFinal.image(img_final, caption = "Final Image (SNR = "+str(signaltonoise(img_final,axis = None))+" ) ", use_column_width=True)




