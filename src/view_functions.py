import math
import os

import cv2
import numpy as np
import scipy.ndimage as ndi
import scipy.ndimage as ndimage
import skimage.exposure
from PIL import Image, ImageDraw
# from miseval import evaluate
from skimage import exposure, morphology
from skimage.filters import gaussian
from skimage.filters import threshold_multiotsu
from skimage.measure import regionprops
from skimage.morphology import dilation
from skimage.morphology import square
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import active_contour
from dcmrtstruct2nii import dcmrtstruct2nii

import nibabel


def ConvertRTS2NII(RTSFilePath, DicomDatabase, OutFilename, ret=False):
    """
  This is DICOM series to the RT Struct. conversion.
  :param DicomDatabase: The Path to dcm images.
         RTSFilePath : path of input  RTS.dcm file
         OutFilename : The 2d mask if selected slice
         return : saved by Format  nii
    """
    try:
        dcmrtstruct2nii(RTSFilePath, DicomDatabase, OutFilename)
        lis = os.listdir(OutFilename)
        im = None
        for i in lis:
            if "mask" in i:
                im1 = np.array(nibabel.load(OutFilename + "/" + i).get_fdata())
                if im is None:
                    im = im1
                else:
                    for j in range(im.shape[2]):
                        im[:, :, j] = add_label(im[:, :, j], im1[:, :, j])
        if ret:
            return im.astype(np.uint8)
        img = nibabel.Nifti1Image(im, affine=np.eye(4))
        path = OutFilename + "/mask_total.nii.gz"
        nibabel.save(img, path)
        return path
    except Exception as e:
        print(e)
    return OutFilename


def ImageInfo(image):
    """
    This is the image information extraction.
    :param image: The image.
    :return: The information image.
    """

    out = image.header
    return out


def layer_inspector(image, which_layer):
    """
    This is the image layers inspector.
    :param image: The image.
    :param which_layer : selected layer
    :return: The selected Layer of image.
    """

    _, _, d_max = image.shape
    if which_layer > d_max:
        print('Error : Layer excide boundaries')
    else:
        out = image[:, :, which_layer]

        return out


def contrast(image, which_layer, contrast_value):
    """
    This is the Changing the Contrast of images.
    :param image: The image.
    :param which_layer : selected layer
    :param contrast_value : value of gamma correction in Contrast
    :return The selected Layer of image.
    """

    _, _, d_max = image.shape
    if which_layer > d_max:
        print('Error : Layer excide boundaries')
    else:
        frame = image[:, :, which_layer]
        # Gamma:
        out = exposure.adjust_gamma(frame, contrast_value)
        # or Logarithmic:
        # out = exposure.adjust_log(Frame, ContrastValue)
        return out


def crop(image, x0, y0, z0, w, h, d):
    """
    This is the Crop algorithm.
    :param image: The image.
    :param x0 : starting point
    :param y0 : starting point
    :param z0 : starting point
    :param w : width of crop
    :param h : height of crop
    :param d : depth of crop
    :return: The cropped image.
    """

    w_max, h_max, d_max = image.shape
    if (x0 + w > w_max) | (y0 + h > h_max) | (z0 + d > d_max):
        print('Error : Size exceed boundaries')
    else:
        out = image[x0:x0 + w, y0:y0 + h, z0:z0 + d]

        return out


def label(image, which_layer, which_label, new_label):
    """
    This is the Modifying label of Contour/Segmentation(Templating) algorithm.
    :param image: The image.
    :param which_layer : selected layer
    :param which_label : selected label
    :param new_label : the value of new label to be replaced
    :return: The new label image.
    """

    _, _, d_max = image.shape
    if which_layer > d_max:
        print('Error : Layer exceed boundaries')
    else:
        frame = image[:, :, which_layer]
        out = frame
        out[(out == which_label)] = new_label

        return out


def order(frame):
    """
    This is the Changing the Order of Contour/Segmentation algorithm.
    :param frame: The image.
    :return: The sorted label image.
    """

    out = frame
    print(np.max(frame))
    for i in range(0, np.max(frame)):
        out[(frame == i)] = np.abs(- i)

    return out


def transparency(image, whichLayer, whichLabel, NewTransparency):
    """
    This is the Changing the Transparency of Contour/Segmentation algorithm.
    :param image: The image.
     whichLayer : selected layer
     whichLabel : selected label
     NewLabel : the value of new lebel to be replaced
       :return: The new Transparency image.
    """

    _, _, d_max = image.shape
    if whichLayer > d_max:
        print('Error : Layer excide boundaries')
    else:
        frame = image[:, :, whichLayer]
        out = frame
        out[(out == whichLabel)] = NewTransparency

        return out


def visibility(image, whichLayer, whichLabel):
    """
    This is the Choosing which Contour/Segmentation to be Shown algorithm.
    :param image: The image.
     whichLayer : selected layer
     whichLabel : selected label
       :return: The selected label .
    """

    _, _, d_max = image.shape
    if whichLayer > d_max:
        print('Error : Layer excide boundaries')
    else:
        frame = image[:, :, whichLayer]
        out = frame
        out = (out == whichLabel)

        return out


def manual(image, whichLayer, polygon):
    """
    This is the Manual segmentation(Brush/Polygon) algorithm.
    :param image: The image.
     whichLayer : selected layer
     polygon : selected polygon
       :return: The Segmented regions.
    """

    _, _, d_max = image.shape
    if whichLayer > d_max:
        print('Error : Layer excide boundaries')
    else:
        frame = image[:, :, whichLayer]
        img_array = frame
        mask_img = Image.new('1', (img_array.shape[1], img_array.shape[0]), 0)
        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
        mask = np.array(mask_img)
        out = mask

        return out


def PET(image, which_layer, polygon):
    """
    This is the PET tumor segmentation algorithm.
    :param image: The image.
    :param which_layer : selected layer
     polygon : selected polygon
       :return: The Segmented regions.
    """

    frame = image[:, :, which_layer]
    ### Step 1 :
    patch_kw = dict(patch_size=5,
                    patch_distance=6)
    sigma_est = np.mean(estimate_sigma(frame))
    NLM_denoised = denoise_nl_means(frame, h=0.6 * sigma_est, sigma=sigma_est,
                                    fast_mode=True, **patch_kw)
    ### Step 2 :
    bright_pixel = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]], dtype=np.uint8)
    Morph_filtering = dilation(NLM_denoised, square(3))
    ### Step 3 :
    Diffr = np.abs(frame - Morph_filtering);

    ### Step 4 :
    img_array = Diffr
    mask_img = Image.new('1', (img_array.shape[1], img_array.shape[0]), 0)
    ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask_img)
    new_img_array = np.empty(img_array.shape, dtype='uint8')
    new_img_array = img_array
    new_img_array = new_img_array * mask
    threshold = np.mean(np.nonzero(new_img_array))
    out = (new_img_array >= threshold / 255)

    return out


def RG(I, sigma, seed_x, seed_y, thresh):
    # seedGrowingSeg segments a 2D image using region growing algorithm

    # FUNCTION DESCRIPTION:
    #   1) Receives 2D slice and performs pre-processing Gaussian blurring
    #   2) Iteratively finds all non-segmented pixels connected to the current
    #      segmented region
    #   3) Compares similarity of connected pixels to mean of segmented region
    #   4) If smallest pixel intensity difference is below threshold value
    #      then this pixel is added to segmentation
    #   5) If smallest pixel instensity difference is greater than
    #      threshold value, or if maximum number of iterations has beed
    #      exceeded, then algorithm has converged

    # INPUTS:
    #   I - 2D image normalised between 0 and 1 - float
    #   sigma - width of Gaussian kernel for blurring - float
    #   seed_x - x coordinate of initial seed point - int
    #   seed_y - y coordinate of initial seed point - int
    #   thresh - threshold for convergence of segmentation - float

    # OUTPUTS:
    #   Im_blur - output of 2D Gaussian blurring on I - float
    #   seg - logical segmentation output - bool
    #   region_size - number of pixels in segmented region - int

    # FUNCTION DEPENDENCIES:
    #   numpy
    #   scipy.ndimage

    n_rows, n_cols = I.shape

    # %% pre processing - 2D gaussian blurring to smooth out noise
    Im_blur = ndi.gaussian_filter(I, sigma, order=0, mode='reflect')

    # %% preallocate memory for segmentation output, also specify maximum
    # number of iterations as 10% of image area
    seg = np.zeros([n_rows, n_cols], bool)
    max_num_iterations = round(n_rows * n_cols / 10)
    iterations = np.arange(1, max_num_iterations)

    # preallocate memory for coords of segmentation logical labels
    all_x_coords = np.zeros(max_num_iterations, int)
    all_y_coords = np.zeros(max_num_iterations, int)

    # initialise first iteration
    seg[seed_y, seed_x] = 1
    region_mean = Im_blur[seed_y, seed_x]
    all_x_coords[0] = seed_x
    all_y_coords[0] = seed_y

    # %% define 2D (i.e. in-plane) 4 degree connectivity of a given pixel centred at (0,0)
    # pixels are connected if their edges touch
    kernel = np.zeros([4, 2], int)
    kernel[0, 0] = 1
    kernel[1, 1] = -1
    kernel[2, 0] = -1
    kernel[3, 1] = 1
    connectivity = np.arange(4)

    # %% start iterative region growing loop

    for num_iter in iterations:
        # preallocate temporary matrix of all unassigned connected pixels
        connected_pix = np.zeros([n_rows, n_cols])
        # acquire coordinates of all connected pixels
        for i in connectivity:
            conn_y_coord = all_y_coords[0:num_iter] + kernel[i, 0]
            conn_x_coord = all_x_coords[0:num_iter] + kernel[i, 1]
            # loop through each of these connected pixels and add to temporary matrix
            n_coords = np.arange(len(conn_x_coord))
            for j in n_coords:
                # if pixel address is contained in image
                if 0 <= conn_y_coord[j] < n_rows and 0 <= conn_x_coord[j] < n_cols:
                    # if not already part of current segmentation
                    if seg[conn_y_coord[j], conn_x_coord[j]] != 1:
                        connected_pix[conn_y_coord[j], conn_x_coord[j]] = 1
        # multiply blurred image by this logical mask
        connected_pix_intensities = Im_blur * connected_pix
        # find the pixel which has the smallest absolute intensity difference to the
        # current region mean
        sim_metric_all = np.reshape(abs(region_mean - connected_pix_intensities), n_cols * n_rows)
        # calculate smallest current similarity metric and location of respective pixel
        # in flattened array
        sim_metric = min(sim_metric_all)
        ind = np.argmin(sim_metric_all)
        # if this absolute intensity difference is smaller than threshold then add
        # this pixel to segmentation region and update region mean
        if sim_metric < thresh:
            # convert this 1D idx to 2D coords
            [new_y_idx, new_x_idx] = np.unravel_index(ind, (n_rows, n_cols))
            # and add to all_coords
            all_x_coords[num_iter] = new_x_idx
            all_y_coords[num_iter] = new_y_idx
            # and add to segmentation
            seg[new_y_idx, new_x_idx] = 1
            # update region mean
            region = Im_blur * seg
            region_mask = region[region != 0]
            region_mean = np.mean(region_mask)
        else:
            # break out of outer for loop
            break

    # calculate total number of pixels in segmentation
    region_size = sum(sum(region != 0))

    return np.array(seg, dtype=np.uint8)


def snake(frame, seedcenter_x, seedcenter_y, Radius):
    """
    This is the Snake Segmentation algorithm.
    :param frame: The input image frame.
     seedcenter_x ,seedcenter_y : the seed center for segmentation
     Radius : the radius of circle enclosing the seed to initial segmentation
       :return: The output segment .
    """

    original = frame
    s = np.linspace(0, 2 * np.pi, 400)
    r = seedcenter_x + Radius * np.sin(s)
    c = seedcenter_y + Radius * np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(gaussian(frame, 2, preserve_range=False),
                           init, alpha=0.015, beta=10, gamma=0.001)

    r_mask = np.zeros_like(frame, dtype='bool')
    r_mask[np.round(snake[:, 0]).astype('int'), np.round(snake[:, 1]).astype('int')] = 1
    r_mask = ndimage.binary_fill_holes(r_mask)
    out = r_mask

    return out


def threshold(image, whichLayer):
    """
    This is the Threshold Segmentation algorithm.
    :param image: The image.
     whichLayer : selected layer
       :return: The Segmented regions.
    """

    _, _, d_max = image.shape
    if whichLayer > d_max:
        print('Error : Layer excide boundaries')
    else:
        frame = image[:, :, whichLayer]
        thresholds = threshold_multiotsu(frame)
        out = np.digitize(frame, bins=thresholds)

        return out


def multi_threshold(image, whichLayer, thresholds):
    """
    This is the multilevel Threshold Segmentation algorithm.
    :param image: The image.
     whichLayer : selected layer
     thresholds : manual list of thresholds (range 0-255)
       :return: The Segmented regions.
    """

    _, _, d_max = image.shape
    if whichLayer > d_max:
        print('Error : Layer excide boundaries')
    else:
        frame = image[:, :, whichLayer]
        original = frame
        thresholds = sorted(thresholds)
        frame = ((frame - frame.min()) * (1 / (frame.max() - frame.min()) * 255)).astype('uint8')
        out = np.digitize(frame, bins=thresholds)

        return out


def Polygons(Label_Image, thresh):
    """
    This is the Manual segmentation(Brush/Polygon) algorithm.
    :param image: The image.
        thresh: minimum distance threshold
       :return: The created polygon
    """

    gray = Label_Image
    ############### End optional block ##################

    # Apply a bilateral filter.
    # This filter smooths the image, reduces noise, while preserving the edges
    bi = gray

    # Apply Harris Corner detection.
    # The four parameters are:
    #   The input image
    #   The size of the neighborhood considered for corner detection
    #   Aperture parameter of the Sobel derivative used.
    #   Harris detector free parameter
    #   --You can tweak this parameter to get better results
    #   --0.02 for tshirt, 0.04 for washcloth, 0.02 for jeans, 0.05 for contour_thresh_jeans
    #   Source: https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
    dst = cv2.cornerHarris(bi, 2, 3, 0.02)

    # Dilate the result to mark the corners
    dst = cv2.dilate(dst, None)

    # Create a mask to identify corners
    mask = np.zeros_like(gray)

    # All pixels above a certain threshold are converted to white
    mask[dst > 0.01 * dst.max()] = 255

    # Convert corners from white to red.
    # img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Create an array that lists all the pixels that are corners
    coordinates = np.argwhere(mask)

    # Convert array of arrays to lists of lists
    coordinates_list = [l.tolist() for l in list(coordinates)]

    # Convert list to tuples
    coordinates_tuples = [tuple(l) for l in coordinates_list]

    # Compute the distance from each corner to every other corner.
    def distance(pt1, pt2):
        (x1, y1), (x2, y2) = pt1, pt2
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    # Keep corners that satisfy the distance threshold
    coordinates_tuples_copy = coordinates_tuples
    i = 1
    for pt1 in coordinates_tuples:
        for pt2 in coordinates_tuples[i::1]:
            if (distance(pt1, pt2) < thresh):
                coordinates_tuples_copy.remove(pt2)
        i += 1

    out = coordinates_tuples_copy

    return out


from rt_utils import RTStructBuilder
import numpy as np


def DICOM2RTS(DicomDatabase, MASK_FROM_ML_MODEL, PathOutput):
    """
  This is DICOM series to the RT Struct. conversion.
  :param DicomDatabase: The Path to dcm images.
         MASK_FROM_ML_MODEL : 3D mask of slices (segmentation), [size_each_slice*number of slices]
         PathOutput : path of output to save RTS.dcm file
      :return: The images visualization.
    """

    # Create new RT Struct. Requires the DICOM series path for the RT Struct.
    rtstruct = RTStructBuilder.create_new(dicom_series_path=DicomDatabase)

    # Add the 3D mask as an ROI.
    # The colour, description, and name will be auto generated
    rtstruct.add_roi(mask=MASK_FROM_ML_MODEL)

    # Add another ROI, this time setting the color, description, and name
    rtstruct.add_roi(
        mask=MASK_FROM_ML_MODEL,
        color=[255, 0, 255],
        name="RT-Utils ROI!"
    )

    rtstruct.save(PathOutput)
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=DicomDatabase,
        rt_struct_path=(PathOutput + ".dcm")
    )

    # View all of the ROI names from within the image
    print(rtstruct.get_roi_names())

    # Loading the 3D Mask from within the RT Struct
    mask_3d = rtstruct.get_roi_mask_by_name("RT-Utils ROI!")
    return mask_3d


def merge_labels(frame, Label1, Label2):
    """
    This is the Merge Contour/Segmentation algorithm.
    :param frame: The image.
    :param Label1 : first selected label
    :param Label2 : second selected label
    :return: The selected label .
    """

    out = frame
    out[(out == Label2)] = Label1

    return out


def subtract(frame, Label1, Label2):
    """
    This is the Subtract  Contour/Segmentation algorithm.
    :param frame: The image.
     Label1 : first selected label
     Label2 : second selected label
       :return: The output labels .
    """
    out = frame
    p1 = (out == Label1)
    p2 = (out == Label1)
    temp = (p1 ^ p2) > 0
    out[(out == Label1)] = 0
    out[(out == Label2)] = 0
    out = out + temp * Label1

    return out


def annotation(x0, y0, x1, y1):
    """
    This is for Distance meter.
    :param image: The image.
         whichLayer : selected layer
         start point : [x0, y0]
         End point : [x1, y1]
         return: The Euclidean Distance of two points
    """

    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def calculate_surface(image):
    """
    This is for Calculate Surface of a 3d mask image.
    :param image: The image.
         return: The Surface of image
    """

    suf = 0
    _, _, d_max = image.shape
    for i in range(0, d_max):
        frame = image[:, :, i]

        thresholds = threshold_multiotsu(frame)
        bw = np.digitize(frame, bins=thresholds)
        bw = bw > 0
        regions = regionprops(bw.astype(int))

        suf = suf + regions[0].perimeter

    out = suf

    return out


def calculate_volume(image):
    """
    This is for Calculate Volume of a 3d mask image.
    :param image: The image.
         return: The Volume of image
    """

    vol = 0
    _, _, d_max = image.shape
    for i in range(0, d_max):
        frame = image[:, :, i]
        vol = vol + np.sum(frame)

    out = vol

    return out


def label_comparision(Label1, Label2, metric):
    """
    This is the Label Comparision using DSC, Volume, JC, HD, etc, algorithm.
     Label1 : first selected label
     Label2 : second selected label
     metric : the criteria from list :
     Index in miseval
                      "DSC"		"Dice"
                      "IoU"		"Jaccard"
                      "SENS"		"Sensitivity"
                      "SPEC"		"Specificity"
                      "PREC"		"Precision"
                      "ACC"		"Accuracy"
                      "BACC"		"BalancedAccuracy"
                      "ARI"		"AdjustedRandlndex"
                      "AUC"		"AUC_trapezoid"
                      "KAP"		"Kappa"
                      "HD"		"HausdorffDistance"
                      "AHD"		"AverageHausdorffDistance"
                      "VS"		"VolumetricSimilarity"
                      "MCC"		"MatthewsCorrelationCoefficient"
                      "nMCC"		"MCC_normalized"
                      "aMCC"		"MCC_absolute"
                      "BD"		"Distance"
                      "Hinge"		"HingeLoss"
                      "CE"		"CrossEntropy"
                      "TP"		“TruePositive”
                      "FP"		"FalsePositive"
                      "TN"		"TrueNegative"
                      "FN"		"FalseNegative"

       :return: The evaluated criteria.
    """

    # out = evaluate(Label1, Label2, metric)
    out = 0
    return out


def morphology_(frame):
    """
    This is the Morphology filters on masks (label Smoothing) algorithms.
    :param frame: The image.
       :return: The label smoothed image.
    """

    # load image
    img = frame
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
    out = skimage.exposure.rescale_intensity(blur, in_range=(127.5, 255), out_range=(0, 255))

    return out


def preprocess(frame, What, param):
    """
    This is some most general Pre Processing algorithms.
    :param frame: The image.
     What : choose what kind of processing including :
          [Resampling, Changing Orientation, Contrast Enhancement, CLAHE, Clip]
    params : for
                resampling---> new size, exp: (400,400)
                changing_orientation---> degree of rotation, exp: (45)
                Contrast_Enhancement---> contrast level , exp: (0.5)
                CLAHE---> clipLimit level , exp : (600)
                Clip---> (left, top, right, bottom) of image to crop , exp: (1,1,10,10)

       :return: The processed image.
    """
    if What == 'resampling':
        out = cv2.resize(frame, param)
    elif What == 'changing_orientation':
        im = Image.fromarray(frame)
        out = im.rotate(param)
    elif What == 'Contrast_Enhancement':
        out = cv2.convertScaleAbs(frame, alpha=param, beta=50)
    elif What == 'CLAHE':
        im = np.array(frame, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=param, tileGridSize=(1, 1))
        out = clahe.apply(im)
    elif What == 'Clip':
        x = param[0]
        y = param[1]
        w = param[2]
        h = param[3]
        out = frame[y:y + h, x:x + w]
    else:
        return None

    return out


def post_process(frame, What, param):
    """
    This is some most general Post Processing algorithms.
    :param frame: The image.
     What : choose what kind of processing including :
          [remove_small_objects/dilation/erosion/opening/closing/skeletonize]
    params : only used for remove_small_objects to remove object less than params
       :return: The processed image.
    """
    if What == 'remove_small_objects':
        min_size = param
        out = morphology.remove_small_objects(frame, min_size)
    elif What == 'binary_dilation':
        out = morphology.binary_dilation(frame)
    elif What == 'binary_erosion':
        out = morphology.binary_erosion(frame)
    elif What == 'binary_opening':
        out = morphology.binary_opening(frame)
    elif What == 'binary_closing':
        out = morphology.binary_closing(frame)
    elif What == 'skeletonize':
        out = morphology.skeletonize(frame)
    else:
        return None

    return out


def add_label(PastLabels, NewLabel):
    """
    This is the adding label to the existing labels algorithm. 
    :param frame: The image Frame.
     whichLayer : selected layer
     pastLabels : past labels
     NewLabel : the value of new lebel to be replaced
       :return: The updated labels image.
    """

    MaxL = np.max(PastLabels)
    NewLabelBW = (MaxL + 1) * np.array(NewLabel > 0)
    out = PastLabels + NewLabelBW
    return out
