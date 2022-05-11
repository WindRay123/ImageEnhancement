# Script for running MUSICA algorithm on a grayscale image:
# Written by Lafith Mattara on 2021-06-05 and Modified by Mahesh on 2021-08-10 and small modifications by Asif

import numpy as np
import copy
from skimage.transform import pyramid_reduce, pyramid_expand
from skimage.transform import resize
import matplotlib.pyplot as plt


def non_linear_gamma_correction(img, params):
    """Non linear gamma correction

    Parameters
    ----------
    img : Image
    params : dict
        Store values of a, M and p.

    Returns
    -------
    en_img: Enhanced Image

    """
    # Non linear operation goes here:
    M = params['M']
    p = params['p']
    a = params['a']

    en_img = a * M * np.multiply(
        np.divide(
            img, np.abs(img), out=np.zeros_like(img), where=img != 0),
        np.power(
            np.divide(
                np.abs(img), M), p))
    return en_img


def display_pyramid(pyramid):
    """Function for plotting all levels of an image pyramid

    Parameters
    ----------
    pyramid : list
        list containing all levels of the pyramid
    """
    rows, cols = pyramid[0].shape
    composite_image = np.zeros((rows, cols + (cols // 2)), dtype=np.double)
    composite_image[:rows, :cols] = pyramid[0]
    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
    fig, ax = plt.subplots()
    ax.imshow(composite_image, cmap='gray')
    plt.show()


def isPowerofTwo(x):
    # check if number x is a power of two
    return x and (not (x & (x - 1)))


def findNextPowerOf2(n):
    # taken from https://www.techiedelight.com/round-next-highest-power-2/
    # Function will find next power of 2

    # decrement `n` (to handle cases when `n` itself
    # is a power of 2)
    n = n - 1
    # do till only one bit is left
    while n & n - 1:
        n = n & n - 1  # unset rightmost bit
    # `n` is now a power of two (less than `n`)
    # return next power of 2
    return n << 1


def resize_image(img):
    """MUSICA works for dimension like 2^N*2^M.
    Hence padding is required for arbitrary shapes

    Parameters
    ----------
    img : numpy.ndarray
        Original image

    Returns
    -------
    numpy.ndarray
        Resized image after padding
    """
    row, col = img.shape
    # check if dimensions are power of two
    # if not pad the image accordingly
    if isPowerofTwo(row):
        rowdiff = 0
    else:
        nextpower = findNextPowerOf2(row)
        rowdiff = nextpower - row

    if isPowerofTwo(col):
        coldiff = 0
    else:
        nextpower = findNextPowerOf2(col)
        coldiff = nextpower - col

    img_ = np.pad(
        img,
        ((0, rowdiff), (0, coldiff)),
        'reflect')
    return img_


def gaussian_pyramid(img, L):
    """Function for creating a Gaussian Pyramid

    Parameters
    ----------
    img : numpy.ndarray
        Input image or g0.
    L : Int
        Maximum level of decomposition.

    Returns
    -------
    list
        list containing images from g0 to gL in order
    """
    # Gaussian Pyramid
    tmp = copy.deepcopy(img)
    gp = [tmp]
    for layer in range(L):
        tmp = pyramid_reduce(tmp, preserve_range=True)
        gp.append(tmp)
    return gp


def laplacian_pyramid(img, L):
    """Function for creating Laplacian Pyramid

    Parameters
    ----------
    img : numpy.ndarray
        Input image or g0.
    L : Int
        Max layer of decomposition

    Returns
    -------
    list
        list containing laplacian layers from L_0 to L_L in order
    list
        list containing layers of gauss pyramid
    """
    gauss = gaussian_pyramid(img, L)
    # Laplacian Pyramid:
    lp = []
    for layer in range(L):
        # logger.debug('Creating layer %d' % (layer))
        tmp = pyramid_expand(gauss[layer + 1], preserve_range=True)
        tmp = gauss[layer] - tmp
        lp.append(tmp)
    lp.append(gauss[L])
    return lp, gauss


def enhance_coefficients(laplacian, L, params):
    """Non linear operation of pyramid coefficients

    Parameters
    ----------
    laplacian : list
        Laplacian pyramid of the image.
    L : Int
        Max layer of decomposition
    params : dict
        Store values of a, M and p.

    Returns
    -------
    list
        List of enhanced pyramid coeffiencts.
    """
    # Non linear operation goes here:
    M = params['M']
    p = params['p']
    a = params['a']
    xc = params['xc']
    for layer in range(L):
        x = laplacian[layer]
        G = a[layer] * M
        x_x = np.divide(
            x, np.abs(x),
            out=np.zeros_like(x),
            where=x != 0)
        x_x = np.divide(
            x, xc,
            out=x_x,
            where=np.abs(x) < xc)
        x_mp = np.power(
            np.divide(
                np.abs(x), M), p[layer])
        x_mp = np.power(
            np.divide(
                xc, M,
                out=x_mp,
                where=np.abs(x) < xc),
            p[layer])

        laplacian[layer] = G * np.multiply(x_x, x_mp)
    return laplacian


def reconstruct_image(laplacian, L):
    """Function for reconstructing original image
    from a laplacian pyramid

    Parameters
    ----------
    laplacian : list
        Laplacian pyramid with enhanced coefficients
    L : int
        Max level of decomposition

    Returns
    -------
    numpy.ndarray
        Resultant image matrix after reconstruction.
    """
    # Reconstructing original image from laplacian pyramid
    rs = laplacian[L]
    for i in range(L - 1, -1, -1):
        rs = pyramid_expand(rs, preserve_range=True)
        rs = np.add(rs, laplacian[i])
    return rs


def musica(img, L, params, debug=True):
    """Function for running MUSICA algorithm

    Parameters
    ----------
    img : numpy.ndarray
        Input image
    L : int
        Max level of decomposition
    params : dict
        Contains parameter values required
        for non linear enhancement
    plot : bool, optional
        To plot the result, by default False

    Returns
    -------
    numpy.ndarray
        Final enhanced image with original dimensions
    """

    nr, nc = img.shape
    img = resize(img, (2000, 2000), anti_aliasing=True)
    img_resized = resize_image(img)
    lp, _ = laplacian_pyramid(img_resized, L)
    lp = enhance_coefficients(lp, L, params)
    rs = reconstruct_image(lp, L)
    rs = rs[:img.shape[0], :img.shape[1]]
    rs = resize(rs, (nr, nc), anti_aliasing=True)
    rs = (rs - np.min(rs.flatten())) / (np.max(rs.flatten()) - np.min(rs.flatten()))
    rs = np.uint16(rs * 255.0)
    return rs


def entire_musica(img_o_, L, params):
    # %% Full Image Gamma Correction
    gammaCorrFlag = False  # Set to True if an over-all Gamma correction is needed (mostly for too dark images)
    if gammaCorrFlag:
        xc1 = 1  # Lower Intensity Limit for Gamma Correction
        params1 = {'M': 1, 'a': 1, 'p': 1, 'xc': xc1}

    img_o_ = (img_o_ - np.mean(img_o_.flatten())) / (
                np.max(img_o_.flatten()) - np.min(img_o_.flatten()))  # Normalize the image

    if gammaCorrFlag:
        img_e = non_linear_gamma_correction(img_o_, params1)
    else:
        img_e = img_o_

    img_e = (img_e - np.min(img_e.flatten())) / (
                np.max(img_e.flatten()) - np.min(img_e.flatten()))  # Normalize the image

    img_enhanced = musica(img_e, L, params)

    img_enhanced = img_enhanced - np.mean(img_enhanced.flatten())

    img_o_ = (img_o_ - np.min(img_o_.flatten())) / (
                np.max(img_o_.flatten()) - np.min(img_o_.flatten()))
    img_enhanced = (img_enhanced - np.min(img_enhanced.flatten())) / (
                np.max(img_enhanced.flatten()) - np.min(img_enhanced.flatten()))
    return img_enhanced


def rotate_circular(lis, n):
    """
    This function rotates the input list circularly by n element
    :param lis: The list which is to be rotated circularly by n element

    :return: The circular rotated list
    """
    l = lis

    for i in range(n):
        l = lis
        tmp = l[0]

        for j in range(len(l) - 1):
            l[j] = l[j + 1]
        l[len(l) - 1] = tmp

    return l


def p_gen(L):  # Function to generate the value of all combinations of p for different number of layers L
    list_tp_l2 = []  #############################################For 2 layers ; L = 2  :
    L1_ = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(L1_)):
        L1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        rotate_circular(L1, i)
        list_tp = list(zip(L1_, L1))
        list_tp_l2.append(list_tp)  # list_tp_ stores all the possible combinations of p values for the two layer case

    m2 = []  ############################################### For 3 Layers ; L = 3

    for i in range(11):
        for j in range(11):
            m2.append(list_tp_l2[j][i])

    if L == 2:
        return m2

    a, b = list(zip(*m2))

    a = list(a)
    b = list(b)

    list_tp_l3 = []
    for k in range(len(L1_)):
        for i in range(len(L1_)):
            L1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            rotate_circular(L1, i)
            list_tp = list(zip(a[k * len(L1_):(k + 1) * len(L1_)], b[k * len(L1_):(k + 1) * len(L1_)], L1))
            list_tp_l3.append(
                list_tp)  # list_tp_ stores all the possible combinations of p values for the three layer case

        #######################################################################################################################################
    # For 4 layers ;L = 4
    m3 = []
    for i in range(121):  # Changing the layer 3 ,  p combinations into a 1-D list  so that each tuple can be unzipped..
        for j in range(11):
            m3.append(list_tp_l3[i][j])

    if L == 3:
        return m3

    a4, b4, c4 = list(
        zip(*m3))  # Unzipping the layer 3 , p combinations into respective variables so that they can be mixed and matched to form 4 layer combinations of p
    a4 = list(a4)
    b4 = list(b4)
    c4 = list(c4)

    list_tp_l4 = []
    for k in range(len(L1_) ** 2):
        for i in range(len(L1_)):
            L1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            rotate_circular(L1, i)
            list_tp = list(zip(a4[k * len(L1_):(k + 1) * len(L1_)], b4[k * len(L1_):(k + 1) * len(L1_)],
                               c4[k * len(L1_):(k + 1) * len(L1_)], L1))
            list_tp_l4.append(
                list_tp)  # list_tp_ stores all the possible combinations of p values for the two layer case

    l4_p_combo = []  # 1 D list containing all permutations and combinations of 4 layer p combinations
    for i in range(1331):
        for j in range(11):
            l4_p_combo.append(list_tp_l4[i][j])

    test = set(l4_p_combo)  # test stores the unique elements in l4_p_combo

    if (len(test) == len(l4_p_combo)):
        print(
            "Success!!")  # success means that out of all the combinations of p generated ..none of them is repeated implying that all of the 4 level p combinations have been created..

    if L == 4:
        return l4_p_combo


def p_index_find(p_all):
    """
    The aim of this function is to output the index of the image (varying from 0 to 14640) given the input p values
    ( 4 layer case )
    """

    print("Enter the 4 p values ( 4 Layer case ) whose index you want to know (in 4 separate lines): ")
    p0 = float(input())
    p1 = float(input())
    p2 = float(input())
    p3 = float(input())

    for i in range(len(p_all)):
        p = list(p_all[i])
        if p[0] == p0 and p[1] == p1 and p[1] == p1 and p[2] == p2 and p[3] == p3:
            print("The index for ", p[0], p[1], p[2], p[3], "is :", i)
            return
    print("The entered value is not present ")


def signaltonoise(a, axis=0, ddof=0): #Took the code from https://github.com/scipy/scipy/issues/9097
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
