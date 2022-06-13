import datetime
import numpy as np #Imports numpy package of Python3
import matplotlib.pyplot as plt #Imports matplotlib library of Python  to plot images..
from skimage import io, img_as_float,img_as_ubyte #io module of skimage is to enable reading and writing of images
                                    #img_as_float convert  an image to floating point format , with values in [0,1]
import os    #os module in Python give functions to interact with the operating system
from musica_1_streamlit_version import *

from os import system
import math
import cv2
import streamlit as st

def cnr(img , L , N ):
    """

    :param img: Input Image
    :param L: The number of Layers of Decomposition
    :param N: The size of the neighbourhood we are considering for local standard deviation image calculation
    :return: CNR Image
    """
    detail_layer_3 = Layer_3(img, L)
    local_std_img = local_standard_deviation(detail_layer_3, N)
    local_std_img_8 = img_as_ubyte(local_std_img)
    hist = cv2.calcHist([local_std_img_8], [0], None, [256], [0, 256])
    pos = np.where(hist == max(hist))
    cnr_image = local_std_img_8 / pos[0]
    cnr_image = (cnr_image - np.min(cnr_image)) / np.ptp(cnr_image)

    roi = cv2.selectROI("Please Select the Region of Interest ", cnr_image)
    roi_cropped = cnr_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] #Code taken from GeeksforGeeks
    cnr_roi = roi_cropped.mean()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cnr_image, cnr_roi


st.markdown("<h1 style='text-align: center; color: red;'>Open Ended Lab Project : Image Enhancement</h1>",
            unsafe_allow_html=True)
st.sidebar.markdown("By  ")
st.sidebar.markdown("**_Asif M.S._**")
st.sidebar.markdown("121901007")

st.sidebar.markdown("Under the Supervision of :")
st.sidebar.markdown("**_Dr. Mahesh R. Panicker_**")

file = st.file_uploader("Please upload an image ")
img_o = img_as_float(io.imread(file, as_gray=True))
img_o_8 = img_as_ubyte(img_o)

log_or_sigmoid = st.radio('Please select the variant of the algorithm ', ('ln(1+X)','Sigmoid'))

hist_show = st.checkbox("Please tick the checkbox to display the corresponding histograms of the images ")

if log_or_sigmoid == 'ln(1+X)':
  img_o_fn = np.log( 1 + img_o)

  L = 7 # Number of Levels for Laplacian Pyramid

  a = np.full(L, 1)  # Creates a 1D array of length L and filled with the value 1
  p = np.full(L, 1)  # Creates a 1D array of length L and filled with the value 1
  #p = np.zeros((L,1))

  p_init = 0.5
  p[0] = st.sidebar.slider('Level 0 ', 0.00, 1.00, p_init)
  p[1] = st.sidebar.slider('Level 1 ', 0.00, 1.00, p_init)
  p[2] = st.sidebar.slider('Level 2 ', 0.00, 1.00, p_init)
  p[3] = st.sidebar.slider('Level 3 ', 0.00, 1.00, p_init)
  p[4] = st.sidebar.slider('Level 4 ', 0.00, 1.00,p_init)
  p[5] = st.sidebar.slider('Level 5 ', 0.00, 1.00, p_init)
  p[6] = st.sidebar.slider('Level 6 ', 0.00, 1.00, p_init)

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
  #p = np.zeros((L,1))
  params = {'M': M,'a': a,'p': p,'xc': xc}
  ref_img = entire_musica(img_o,L,params)

  cnr_image_ref, cnr_roi_ref = cnr(ref_img, L, 9)

  snr_ref = signaltonoise(ref_img,axis=None)
  ref_img_8 = img_as_ubyte(ref_img)

  h = st.sidebar.slider('Filter Strength for Denoising', 1, 100, 20)

  #Denosing increases the SNR
  cv2.fastNlMeansDenoising(img_denoised ,img_denoised,h,7,21) #First parameter is the source image , second parameter is the destination image

  img_denoised = (img_denoised - np.min(img_denoised))/np.ptp(img_denoised)

  r =  st.sidebar.slider('r', 0.0, 1.0, 0.0)

  img_final = img_denoised + r*img_o # The final image is a result of weighted addition of the denoised image and the original image
  img_final = (img_final - np.min(img_final))/np.ptp(img_final)

  img_final_8 = img_as_ubyte(img_final)

  N = 9  # Neighbourhood Window Size for Local Standard Deviation Calculation
  cnr_image, cnr_roi = cnr(img_final, L, N)

  imageOrgLocation = st.empty()

  if hist_show:
    fig1 = plt.figure()
    plt.hist(img_o_8.ravel(), 256, [0, 256])
    plt.title("Histogram : Original Image ")
    plt.xlabel("Intensity ( pixel ) Values ")
    plt.ylabel("Count of Occurrence")
    st.pyplot(fig1)

  imageRefLocation = st.empty()
  cnr_imageRefLocation = st.empty()

  if hist_show:
    fig2 = plt.figure()
    plt.hist(ref_img_8.ravel(), 256, [0, 256])
    plt.title("Histogram : Normal MUSICA Image ")
    plt.xlabel("Intensity ( pixel ) Values ")
    plt.ylabel("Count of Occurrence")
    st.pyplot(fig2)

  imageEnhDenoiseLocation = st.empty()
  imageFinal = st.empty()

  cnr_image_Location = st.empty()

  if hist_show:
    fig3 = plt.figure()
    plt.hist(img_final_8.ravel(), 256, [0, 256])
    plt.title("Histogram : Final Image ")
    plt.xlabel("Intensity ( pixel ) Values ")
    plt.ylabel("Count of Occurrence")
    st.pyplot(fig3)

  imageOrgLocation.image(img_o, caption='Original Image (SNR = '+str(signaltonoise(img_o,axis = None))+" ) ", use_column_width=True)
  imageRefLocation.image(ref_img, caption='Normal MUSICA Enhanced Image (SNR = '+str(signaltonoise(ref_img,axis = None))+" ) ", use_column_width=True)
  cnr_imageRefLocation.image(cnr_image_ref, caption='CNR Image : Normal MUSICA  (CNR of the chosen ROI = ' + str(cnr_roi_ref) + " ) ",use_column_width=True)
  imageEnhDenoiseLocation .image(img_denoised ,caption = "Denoised Enhanced Image (SNR = "+str(signaltonoise(img_denoised,axis = None))+" ) ", use_column_width=True)
  imageFinal.image(img_final, caption = "Final Image (SNR = "+str(signaltonoise(img_final,axis = None))+" ) ", use_column_width=True)
  cnr_image_Location.image(cnr_image, caption='CNR Image ( CNR of the chosen ROI = ' + str(cnr_roi) + " ) ", use_column_width=True)
else:
  img_o_fn = 1/(1+np.exp(-img_o))    #np.log( 1 + img_o)

  L = 7 #st.sidebar.slider('Number of Levels Of Decomposition', 4, 10, 7) #4 # Number of Levels for Laplacian Pyramid

  a = np.full(L, 1)  # Creates a 1D array of length L and filled with the value 1
  p = np.full(L, 1)  # Creates a 1D array of length L and filled with the value 1

  p_init = 0.5
  p[0] = st.sidebar.slider('Level 0 ', 0.00, 1.00, p_init)
  p[1] = st.sidebar.slider('Level 1 ', 0.00, 1.00, p_init)
  p[2] = st.sidebar.slider('Level 2 ', 0.00, 1.00, p_init)
  p[3] = st.sidebar.slider('Level 3 ', 0.00, 1.00, p_init)
  p[4] = st.sidebar.slider('Level 4 ', 0.00, 1.00,p_init)
  p[5] = st.sidebar.slider('Level 5 ', 0.00, 1.00, p_init)
  p[6] = st.sidebar.slider('Level 6 ', 0.00, 1.00, p_init)

  M = 1.0
  xc = 0.01 * M  # Lower Intensity Limit for Gamma Correction # st.sidebar.slider('x_c', 0.0000, 0.0100, 0.0010,step=0.0001)#0.01 * M  # Lower Intensity Limit for Gamma Correction
  params = {'M': M,'a': a,'p': p,'xc': xc }

  musica_img = entire_musica(img_o_fn,L,params)

  inverse_sigmoid = np.zeros(musica_img.shape)

  for i in range(musica_img.shape[0]):
    for j in range(musica_img.shape[1]):
        if musica_img[i][j] != 0 and musica_img[i][j] != 1:
            inverse_sigmoid[i][j] = math.log(musica_img[i][j]/(1 - musica_img[i][j]))



  #pos = np.where(musica_img!=0)

  #inverse_sigmoid = np.log(np.divide(musica_img[pos],1-musica_img[pos],where=musica_img[pos]!=1))

  inverse_sigmoid = (inverse_sigmoid - np.min(inverse_sigmoid))/np.ptp(inverse_sigmoid)

  musica_second = entire_musica(inverse_sigmoid,L,params)

  sigmoid_second = 1/(1+np.exp(-musica_second))

  img_e = sigmoid_second

  img_e = (img_e*255).astype(np.uint8)
  img_denoised = img_e

  #Reference SNR Calculation
  params = {'M': M,'a': a,'p': p,'xc': xc}
  ref_img = entire_musica(img_o,L,params)

  cnr_image_ref, cnr_roi_ref = cnr(ref_img, L, 9)

  snr_ref = signaltonoise(ref_img,axis=None)
  ref_img_8 = img_as_ubyte(ref_img)

  h = st.sidebar.slider('Filter Strength for Denoising', 1, 100, 5)

  #Denosing increases the SNR
  cv2.fastNlMeansDenoising(img_denoised ,img_denoised,h,7,21) #First parameter is the source image , second parameter is the destination image

  img_denoised = (img_denoised - np.min(img_denoised))/np.ptp(img_denoised)

  r =  st.sidebar.slider('r', 0.0, 1.0, 0.0)

  img_final = img_denoised + r*img_o
  img_final = (img_final - np.min(img_final))/np.ptp(img_final)

  img_final_8 = img_as_ubyte(img_final)

  N = 9  # Neighbourhood Window Size for Local Standard Deviation Calculation
  cnr_image, cnr_roi = cnr(img_final, L, N)

  imageOrgLocation = st.empty()

  if hist_show:
    fig1 = plt.figure()
    plt.hist(img_o_8.ravel(), 256, [0, 256])
    plt.title("Histogram : Original Image ")
    plt.xlabel("Intensity ( pixel ) Values ")
    plt.ylabel("Count of Occurrence")
    st.pyplot(fig1)

  sigmoid_orgLocation = st.empty()
  imageRefLocation = st.empty()
  cnr_imageRefLocation = st.empty()

  if hist_show:
    fig2 = plt.figure()
    plt.hist(ref_img_8.ravel(), 256, [0, 256])
    plt.title("Histogram : Normal MUSICA Image ")
    plt.xlabel("Intensity ( pixel ) Values ")
    plt.ylabel("Count of Occurrence")
    st.pyplot(fig2)


  musica_imgLocation = st.empty()
  inv_sigmoidLocation = st.empty()
  musica_secondLocation = st.empty()
  sigmoid_secondLocation = st.empty()
  image_denoisedLocation = st.empty()
  image_FinalLocation = st.empty()

  cnr_image_Location = st.empty()

  if hist_show:
    fig3 = plt.figure()
    plt.hist(img_final_8.ravel(), 256, [0, 256])
    plt.title("Histogram : Final Image ")
    plt.xlabel("Intensity ( pixel ) Values ")
    plt.ylabel("Count of Occurrence")
    st.pyplot(fig3)


  imageOrgLocation.image(img_o, caption='Original Image (SNR = '+str(signaltonoise(img_o,axis = None))+" ) ", use_column_width=True)
  #sigmoid_orgLocation.image(img_o_fn, caption='Sigmoid(Original Image)  (SNR = '+str(signaltonoise(img_o_fn,axis = None))+" ) ", use_column_width=True)
  imageRefLocation.image(ref_img, caption='Normal MUSICA Enhanced Image (SNR = '+str(signaltonoise(ref_img,axis = None))+" ) ", use_column_width=True)
  cnr_imageRefLocation.image(cnr_image_ref, caption='CNR Image : Normal MUSICA  (CNR of the chosen ROI = ' + str(cnr_roi_ref) + " ) ",use_column_width=True)
  #musica_imgLocation.image(musica_img, caption ='Org --> Sigmoid --> MUSICA (SNR = '+str(signaltonoise(musica_img,axis = None))+" ) ", use_column_width=True)
  #inv_sigmoidLocation.image(inverse_sigmoid, caption ='Org --> Sigmoid --> MUSICA-->Inverse Sigmoid (SNR = '+str(signaltonoise(inverse_sigmoid,axis = None))+" ) ", use_column_width=True)
  #musica_secondLocation.image(musica_second, caption ='Org --> Sigmoid --> MUSICA --> Inverse Sigmoid --> MUSICA (SNR = '+str(signaltonoise(musica_second,axis = None))+" ) ", use_column_width=True)
  #sigmoid_secondLocation.image(sigmoid_second, caption ='Org --> Sigmoid --> MUSICA --> Inverse Sigmoid --> MUSICA --> Sigmoid (SNR = '+str(signaltonoise(sigmoid_second,axis = None))+" ) ", use_column_width=True)
  image_denoisedLocation.image(img_denoised, caption ='Org --> Sigmoid --> MUSICA --> Inverse Sigmoid --> MUSICA --> Sigmoid --> Denoised (SNR = '+str(signaltonoise(img_denoised,axis = None))+" ) ", use_column_width=True)
  image_FinalLocation.image(img_final, caption ='Org --> Sigmoid --> MUSICA --> Inverse Sigmoid --> MUSICA --> Sigmoid --> Denoised --> Add Original  (SNR = '+str(signaltonoise(img_final,axis = None))+" ) ", use_column_width=True)
  cnr_image_Location.image(cnr_image, caption='CNR Image (CNR of the chosen ROI = ' + str(cnr_roi) + " ) ", use_column_width=True)

st.markdown("<h1 style='text-align: center; color: blue;'>Observations: </h1>",
            unsafe_allow_html=True)
st.write("**Normal MUSICA :** CNR = " + str(cnr_roi_ref))
st.write("**Proposed :** CNR = " + str(cnr_roi))

if cnr_roi>cnr_roi_ref:
  st.markdown("<h1 style='text-align: center; color: red;'>The proposed method is better than Normal MUSICA</h1>",unsafe_allow_html=True)
else:
  st.markdown("<h1 style='text-align: center; color: red;'>Normal MUSICA is better than the proposed method </h1>",unsafe_allow_html=True)

