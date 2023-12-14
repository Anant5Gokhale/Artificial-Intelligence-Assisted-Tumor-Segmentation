'''Code was organised for easier repetition of experiments. Patient 00005 data is used. Pickle is used to
store mask data, so if error occurs mask segmentation data is recovered and 1hr does not have to be put again.
Mask selection method is imporved and is showing good results'''

'''Result: out of 26 images 22 show good segmentation.'''

''' Mean + Standard Deviation _/
    Gaussian Blur X
    Improcessing+otsu X
    All Images Contain Tumors _/
    Full Image Used _/ 
    Mask Selection Method in development
    Ellipse Drawn _/
    Weighted Addition 0.5 0.5
    Push pixels below mean
    Push Intensity 100
    Image Modality and Patient T1CE BRATS 2021 and 00005
    Number of Images Used 26
    Results availiable: https://drive.google.com/uc?export=download&id=1tpydLyFZczJmdsIsC2uyYgb4qoFdPRig'''

# IMPORT LIBRARIES: for Image Processing
# Trying with 00005 images
# Line 59 what is ellipse info axes?
FILENAME = 'no_crash_0_5py'
RCL = (117,123)
RCR = (190,183)
RECT_AREA = abs(RCL[0]-RCR[0])* abs(RCL[1]-RCR[1])
ELLIPSE_AREA = 3392
ELLIPSE_CENTRE = (153,153)
ELLIPSE_AXES = (36, 30)
LARGEST_IMG_PATH = '/Users/dr.rajeshsgokhale/Downloads/extensionForResearchcondaEnv/img00000-to-00005_t1ce/00005_100.png'
LIST_OF_IMS = list(range(84,135,2))
FOLDER_DIR = '/Users/dr.rajeshsgokhale/Downloads/extensionForResearchcondaEnv/img00000-to-00005_t1ce'
PATIENT_NUM = '00005'
LEN_IMS = len(LIST_OF_IMS)
ResultTuple =  ((RCL, RCR), ELLIPSE_AREA, ELLIPSE_CENTRE, ELLIPSE_AXES)
d_new = []
tup_temp = (in1, in2, in3, in4, area, centre, axes)=  RCL[0], RCR[0], RCL[1], RCR[1], ELLIPSE_AREA, ELLIPSE_CENTRE, ELLIPSE_AXES
d_new.append(tup_temp) 
import cv2 
import copy
import numpy as np
import matplotlib.pyplot as plt
#READ THE LARGEST IMAGE IN THE DATASET
image = cv2.imread(LARGEST_IMG_PATH)
def some_val2(image_, centre, axes):
    # Define the center, axes lengths, and angle of the ellipse
    center = ELLIPSE_CENTRE  # Adjust the coordinates as needed
    axes_length = ELLIPSE_AXES  # Adjust the major and minor axes lengths as needed
    angle = 0  # Adjust the rotation angle as needed

    # Create an empty mask image with the same dimensions as the original image
    mask = np.zeros_like(image)

    # Draw the ellipse on the mask (255 is used to set the ellipse pixels to white)
    cv2.ellipse(mask, center, axes_length, angle, 0, 360, (255, 255, 255), -1)

    # Create a boolean mask to identify non-zero intensity pixels
    non_zero_mask = mask > 0

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    # Calculate the mean intensity of non-zero pixels within the elliptical region
    mean_plusstd = np.mean(masked_image[non_zero_mask]) + np.std(masked_image[non_zero_mask])
    print(f'Mean Intensity of Non-Zero Pixels: {mean_plusstd}')
    # Display the masked image and mean intensity
    return mean_plusstd
def some_val(im, pixel_value):
    #MAKE COPY OF ROI
    mean_ij = copy.deepcopy(im)
    #TUPLE UNPACKING TO GET SHAPE
    m,n,o = mean_ij.shape
    #ITERATE OVER ROWS AND COLUMNS OF MEAN_IJ
    for i in range(m):
        for j in range(n):
            '''Increase the pixel intenstiy
            below 30,90,120,150,180... to
            a larger value'''
            if np.mean(mean_ij[i][j])<(pixel_value):
                # print(np.mean(mean_ij[i][j]))
                mean_ij[i][j] = 0

    ''' Weighted sum of orignal image (image_roi) with mean_ij allows to account for details in orignal
    image. The weight of image_roi is high (upto 0.95) to account for details'''

    #WEIGHTED SUM 
    image_roi =cv2.addWeighted(im, 0.5, mean_ij, 0.5, 0)

    # Checking if contour area fits the value assigned
    gray_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    
    return image_roi


# FIND THE GLOBAL INTENSITY FOR PROCESSING
mean_intensity_global = some_val2(image[in3:in4+1,in1:in2+1,:],centre, axes)
plt.imshow(some_val(image, mean_intensity_global))
# Get images of image index
import os
import glob

# Define the folder directory
folder_dir = FOLDER_DIR
image_files = [0]*LEN_IMS
for i,num in enumerate(LIST_OF_IMS):
    image_files[i] = folder_dir+'/'+ PATIENT_NUM +'_'+str(num)+'.png'
# # Use glob to find all image files in the folder that start with '00000'
# image_files = glob.glob(os.path.join(folder_dir, '00002*'))
# for strng in image_files:
#     print(strng[-20:])

# Show first few images 
n = len(image_files)
rows = int(np.ceil(n/4))
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(10, 20))
for i in range(n):
    row = i // cols
    col = i % cols
    im = cv2.imread(image_files[i])
    in1, in2, in3, in4, area, cen, axe = d_new[0]
    im = im[in3:in4+1,in1:in2+1,:]
    axes[row, col].imshow(im, cmap = 'bone')
    axes[row, col].axis('off')  # Turn off axis labels
    axes[row, col].set_title(f'{image_files[i][-8:]}')  # Set a title if needed
# In case there are remaining empty subplots, hide them
for i in range(n, rows * cols):
    axes.flatten()[i].axis('off')

plt.tight_layout()
plt.show()
# Get image region of interest. APPLY PROCESSING
imageHolder = []
for i, each in enumerate(image_files):
    # READ IMAGE
    image = cv2.imread(each)
    # SELECT BOUNDINX-BOX REGION
    in1, in2, in3, in4, area, cen, axe = d_new[0]
    meanintensity_ = some_val2(image[in3:in4+1,in1:in2+1,:],cen,axe )
    image_roi = some_val(image,meanintensity_)
    # image_roi = cv2.GaussianBlur(image_roi,(11,11),0)
    # mean_intensity_ = some_val2(image_roi, cen, axe)
    # image_roi = some_val(image_roi, mean_intensity_)
    imageHolder.append(image_roi)
# show images
n = len(imageHolder)
rows = int(np.ceil(n/4))
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(10, 20))
for i in range(n):
    row = i // cols
    col = i % cols
    im = imageHolder[i]
    axes[row, col].imshow(im, cmap = 'bone')
    axes[row, col].axis('off')  # Turn off axis labels
    axes[row, col].set_title(f'{image_files[i][-8:]}')  # Set a title if needed
# In case there are remaining empty subplots, hide them
for i in range(n, rows * cols):
    axes.flatten()[i].axis('off')

plt.tight_layout()
plt.show()

# IMPORT LIBRARIES : TORCH, TORCHVISION, SYS, SEGMENT_ANYTHING 
import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

#INITIALISE SAM MODEL
sam_checkpoint = "/Users/dr.rajeshsgokhale/Downloads/extensionForResearchcondaEnv/sam_vit_h_4b8939.pth"
model_type = "vit_h"

#SELECT DEVICE. MPS IS AVAILABLE IN PYTORCH ONLY FOR MACOS VENTURA USERS.
# OTHER DEVICE -> 'cuda'
device = 'mps'
if device =='mps':
    print('Using Metal Performance Shaders--MPS: True')

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

#MASK GENERATOR OBJECT INITIALISATION. 
mask_generator_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.96,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
# Send Image Holder into SAM
MasksSAM = []
import time
for i,image in enumerate(imageHolder):
    image = imageHolder[i]
    start_time = time.time()
    masks = mask_generator_.generate(image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time// 60)
    elapsed_seconds = int(elapsed_time % 60)
    print(f'Mask generated\t{i}\nTime taken\t{elapsed_minutes}min {elapsed_seconds}secs\nNumber of Annotations\t{len(masks)}')

    MasksSAM.append(masks)
import pickle5 as pickle
with open('/Users/dr.rajeshsgokhale/Downloads/'+FILENAME+'.pkl','wb') as f:
    pickle.dump(MasksSAM,f)

import numpy as np
import cv2
image = cv2.imread(LARGEST_IMG_PATH,cv2.IMREAD_GRAYSCALE)
boundary_img  = np.zeros_like(image,dtype=np.uint8)
cv2.rectangle(boundary_img, RCL, RCR, (255, 255, 255), thickness=cv2.FILLED)
print(boundary_img.shape)
def region_out(boundary_img, mask):
    for row in range(len(boundary_img)):
        for col in range(len(boundary_img[row])):
            if np.logical_and(boundary_img[row, col] == 0, mask[row, col] == True):
                return 0
            else:
                continue
    return 1        
def calc_mask_area(iiiiiii):
    count_true = np.count_nonzero(iiiiiii['segmentation'])
    return count_true    
xval, yval = MasksSAM[0][0]['segmentation'].shape
AR_LIMIT = (xval*yval) / 65
centre = ELLIPSE_CENTRE
import math
import supervision as sv
for i, ith_mask in enumerate(MasksSAM):
    print(i)
    im = imageHolder[i]
    mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=ith_mask)
    annotated_image = mask_annotator.annotate(scene=im.copy(), detections=detections)
    sv.plot_images_grid(
    images = [imageHolder[i], annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image'])
import copy
copiedMasksSAM = copy.deepcopy(MasksSAM)
for masksOfEachImage in copiedMasksSAM:
    for mask in masksOfEachImage:
        # mid_x, mid_y = mask['segmentation'].shape[1] // 2, mask['segmentation'].shape[0] // 2
        x,y,w,h = mask['bbox']
        X_center, Y_center = x + (w / 2), y + (h / 2)
        eu_dist = math.sqrt(abs(centre[0]-(X_center))**2 + abs(centre[1]- Y_center)**2)
        mask['eu_dist1'] = eu_dist
        eu_dist2 = math.sqrt(abs(x-centre[0])**2 + abs(y-centre[1])**2)
        mask['eu_dist2'] = eu_dist2
sorted_masks = []
# Also check that x,y,w,h of masks should not lie in limit
for masks in copiedMasksSAM:
    sorted_masks.append(sorted(masks, key=(lambda x: x['eu_dist1']), reverse=False)) 
    
imBuffer, are,flag = [], AR_LIMIT,0
''' loop for selecting the largest mask in 20 percent of the total masks
The masks are sorted according to ones closest to the centre. Line245,246
247 changed. Also permlimit 1 should be larger out of the ellipse axes'''
for j, masksOfEachImage in enumerate(sorted_masks):
    sh1,sh2 = mask['segmentation'].shape
    _imbuffer,flag,are = np.zeros((sh1,sh2),dtype=bool),0,AR_LIMIT
    print('-------------------')
    print(f'MASK INDEX: {j}\nTOTAL NUMBER OF MASKS: {len(masksOfEachImage)}')
    for i,mask in enumerate(masksOfEachImage):
        rcleft = (66,117)
        rcright = (120,164)
        perm_limit1 = max(ELLIPSE_AXES[0],ELLIPSE_AXES[1])
        perm_limit2 = 1.0*math.sqrt(((120-66)**2 + (164-117)**2))
        areA = calc_mask_area(mask)
        print(f'Mask Area: {areA}')
        if i > (int(len(masksOfEachImage)*0.60)):
            print(f'----Moving to next Image. i exceeds{(int(len(masksOfEachImage)*0.60))}. Mask checking over---')
            break
        # Replace 2420 by image area
        const_area = RECT_AREA
        if const_area>areA:
            print(f'Mask fits in rectangle')
            if mask['eu_dist1'] < perm_limit1:
                x,y,_,_ = mask['bbox']
                #want centre of drawn box
                centre = ELLIPSE_CENTRE
                # axis of x needs to be changed
                if mask['eu_dist2'] < perm_limit2:
                    print(f'Mask within permissible limits')
                    # if areA < AR_LIMIT*65/40:
                    #     print(f'lower limit maintained. Looking for more than one tumor region ')
                    
                    if region_out(boundary_img, mask['segmentation']):
                        print('imprinting')
                        _imbuffer = np.logical_or(_imbuffer, mask['segmentation'] )
                        are,flag = AR_LIMIT,1   
    print('-------------------')            
    imBuffer.append(_imbuffer)
# show images
import matplotlib.pyplot as plt

n = len(imBuffer)
rows = int(np.ceil(n/4))
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(10, 20))
for i in range(n):
    row = i // cols
    col = i % cols
    im = imBuffer[i]
    axes[row, col].imshow(im, cmap = 'bone')
    axes[row, col].axis('off')  # Turn off axis labels
    axes[row, col].set_title(f'{i}')  # Set a title if needed
# In case there are remaining empty subplots, hide them
for i in range(n, rows * cols):
    axes.flatten()[i].axis('off')

plt.tight_layout()
plt.show()
