# Script for running MUSICA algorithm on a grayscale image:
# Written by Lafith Mattara on 2021-06-05 , Modified by Mahesh on 2021-08-10 and small modifications by Asif
import cv2
import datetime
import numpy as np  # Imports numpy package of Python3
import matplotlib.pyplot as plt  # Imports matplotlib library of Python  to plot images..
from skimage import io, img_as_float  # io module of skimage is to enable reading and writing of images
# img_as_float convert  an image to floating point format , with values in [0,1]
import os  # os module in Python give functions to interact with the operating system
# import time #time module in Python gives various time related functionalities
from musica_1_streamlit_version import *  # This statement imports the musica.py module created previously which contains imporlogt functions for image proceslogg
import streamlit as st

# Streamlit

# %% User Inputs
st.markdown("<h1 style='text-align: center; color: black;'>Interactive Image Contrast Enhancement</h1>",
            unsafe_allow_html=True)
st.sidebar.markdown("Contact: Dr. Mahesh R Panicker (mahesh@iitpkd.ac.in) ")
st.sidebar.markdown("(c) Center for Computational Imaging, IIT Palakkad")

L = st.sidebar.slider('Number of levels', 1, 7, 4)
a = np.full(L, 1)
gammaCorrFlag = st.sidebar.checkbox('Enable Gamma Correction', value=False)
if gammaCorrFlag:
    xc1 = st.sidebar.slider('Lower Intensity Limit for Gamma Correction', 0.0, 3.0, 1.0)
    params1 = {
        'M': 1,
        'a': 1,
        'p': 1,
        'xc': xc1
    }
xc = float(st.sidebar.text_input('Lower Intensity Limit for Laplacian Pyramid Correction', 0.01))

p = np.zeros((L, 1))
for ii in range(L):
    if ii == 0:
        p[ii] = st.sidebar.slider('p-value for level-' + str(ii), 0.0, 1.0, 0.5)
    else:
        p[ii] = st.sidebar.slider('p-value for level-' + str(ii), 0.0, 1.0, 1.0)

params = {
    'M': 1,
    'a': a,
    'p': p,
    'xc': xc
}

file = st.file_uploader("Please put the images here")

#############################################################################################
begin_time = datetime.datetime.now()  # To store the current programing starting time



img_o = img_as_float(io.imread(file, as_gray=True))



img_o_log = np.log(1 + img_o)

img_enhanced = entire_musica(img_o_log , gammaCorrFlag, L, params)  # Gives the normal MUSICA image

img_inv = np.exp(img_enhanced) - 1

img_twice_musica = entire_musica(img_inv, gammaCorrFlag, L, params)

img_log_2 = np.log (1 + img_twice_musica )

img_e = img_log_2



"""
#For Image 2
img_e = (img_e*255).astype(np.uint8)
img_gaussian = img_e
img_gaussian = cv2.fastNlMeansDenoising(img_e ,img_gaussian,30,7,21)
"""
#For Image 1
img_e = (img_e*255).astype(np.uint8)

img_e = (img_e*255).astype(np.uint8)

img_gaussian = img_e
img_gaussian = cv2.fastNlMeansDenoising(img_e ,img_gaussian,30,7,21)

#img_e = (img_e*255).astype(np.uint8)
#img_gaussian = img_e
img_gaussian = cv2.fastNlMeansDenoising(img_e ,img_gaussian,30,7,21)

#img_enhanced_inv = np.exp(img_enhanced) - 1

#img_enhanced_inv = (img_enhanced_inv*255).astype(np.uint8)

#img_musica_2 =  entire_musica(img_enhanced_inv, gammaCorrFlag, L, params)


#
#img_enhanced_denoised = img_enhanced#np.zeros((img_enhanced.shape[0],img_enhanced.shape[1]))

#img_enhanced_denoised  = np.log(1 +img_enhanced)#cv2.fastNlMeansDenoising(img_enhanced,img_enhanced_denoised,30,7,21)#h= 30.0,templateWindowSize= 7,searchWindowSize=21



imageOrgLocation = st.empty()
imageEnhLocation = st.empty()
imageEnhTwiceLocation = st.empty()
imageEnhDenoiseLocation = st.empty()


imageOrgLocation.image(img_o, caption='Original Image (SNR = '+str(signaltonoise(img_o,axis = None))+" ) ", use_column_width=True)
imageEnhLocation.image(img_enhanced, caption='Enhanced Image (SNR = '+str(signaltonoise(img_enhanced,axis = None))+" ) ", use_column_width=True)
imageEnhTwiceLocation.image(img_twice_musica,caption='MUSICA Twice Image (SNR = '+ str(signaltonoise(img_twice_musica,axis = None))+" ) ", use_column_width=True)
imageEnhDenoiseLocation .image(img_gaussian ,caption = "Denoised Enhanced Image SNR = "+str(signaltonoise(img_gaussian,axis = None))+" ) ", use_column_width=True)



print(datetime.datetime.now() - begin_time)

# Histogram Plotting...

#plt.figure()
#plt.hist(img_o.ravel(), 256, [0, 1],label = "Original Image");
#plt.title("Original Image")
#plt.figure()
#plt.hist(img_enhanced.ravel(), 256, [0, 1],label = "Normal MUSICA");
#plt.title("Normal MUSICA")
#plt.figure()
#plt.hist(img_enhanced_log_before.ravel(), 256, [0, 1],label = "Log Before MUSICA ");
#plt.title("Log Before MUSICA ")
#plt.show()
