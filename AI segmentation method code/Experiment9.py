'''Now trying the same method over a different patients images.'''

'''Result good segmentation as in earlier Patient. Mask selection method bugs related to not choosing any region at all
was solved '''

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
    Image Modality and Patient T1CE BRATS 2021 and 00003
    Number of Images Used 24
    Results availiable: https://drive.google.com/uc?export=download&id=1m7y2KPOKrs18wyxYL_yeah5U5mo12Ork'''

# IMPORT LIBRARIES: for Image Processing
# Trying with 00003 images
# Line 59 what is ellipse info axes?
FILENAME = 'no_crash_0++py'
RCL = (84,103)
RCR = (139,147)
RECT_AREA = abs(RCL[0]-RCR[0])* abs(RCL[1]-RCR[1])
ELLIPSE_AREA = 1866
ELLIPSE_CENTRE = (111,125)
ELLIPSE_AXES = (27, 22)
ResultTuple =  ((RCL, RCR), ELLIPSE_AREA, ELLIPSE_CENTRE, ELLIPSE_AXES)
d_new = []
tup_temp = (in1, in2, in3, in4, area, centre, axes)=  RCL[0], RCR[0], RCL[1], RCR[1], ELLIPSE_AREA, ELLIPSE_CENTRE, ELLIPSE_AXES
d_new.append(tup_temp) 
import cv2 
import copy
import numpy as np
import matplotlib.pyplot as plt
#READ THE LARGEST IMAGE IN THE DATASET
image = cv2.imread('/Users/dr.rajeshsgokhale/Downloads/extensionForResearchcondaEnv/img00000-to-00005_t1ce/00003_110.png')
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
folder_dir = '/Users/dr.rajeshsgokhale/Downloads/extensionForResearchcondaEnv/img00000-to-00005_t1ce'
image_files = [0]*24
for i,num in enumerate(list(range(78,125,2))):
    image_files[i] = folder_dir+'/00003_'+str(num)+'.png'
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
for i, each in enumerate(image_files[:20]):
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
with open('/Users/dr.rajeshsgokhale/Desktop/'+FILENAME+'.pkl','wb') as f:
    pickle.dump(MasksSAM,f)
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
for each in copiedMasksSAM:
    print(len(each))
# Calculate the mid-location of the mask['segmentation']
xval, yval = MasksSAM[0][0]['segmentation'].shape
AR_LIMIT = (xval*yval) / 65
print(AR_LIMIT)
def calc_mask_area(iiiiiii):
    count_true = np.count_nonzero(iiiiiii['segmentation'])
    return count_true

centre = ELLIPSE_CENTRE
import math
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
        perm_limit2 = 0.75*math.sqrt(((120-66)**2 + (164-117)**2))
        areA = calc_mask_area(mask)
        print(f'Mask Area: {areA}')
        if i > (int(len(masksOfEachImage)*0.60)):
            print(f'----Moving to next Image. i exceeds{(int(len(masksOfEachImage)*0.10))}. Mask checking over---')
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
                    are,flag = AR_LIMIT,1
                    _imbuffer = np.logical_or(_imbuffer, mask['segmentation'] )
                    # else:
                    #     # Area is bigger than AR_LIMIT/40 (Only one mask for tumor)
                    #     # OR
                    #     # Get the mask on _imbuffer
                    #     are, flag = areA, 1
                    #     _imbuffer = mask['segmentation']    
    print('-------------------')            
    imBuffer.append(_imbuffer)
# show images
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

#mistakes:[4][20],[5][0], [6]
# NUM = 5
# NUM2 = 19
# perm_limit1 = 0.25*(max((mask['segmentation'].shape[1]), (mask['segmentation'].shape[0])))
# perm_limit2 = 0.4*math.sqrt(((mask['segmentation'].shape[1])**2 + (mask['segmentation'].shape[0])**2))
# print(f'permissible1:{perm_limit1}\npermissible2:{perm_limit2}')
# print(f'eu_dist:\t')
# print((sorted_masks[NUM2][NUM]['eu_dist']))
# print(sorted_masks[NUM2][NUM]['eu_dist'] < perm_limit1)
# x,y,w,h = sorted_masks[NUM2][NUM]['bbox']
# mid_x, mid_y = sorted_masks[NUM2][NUM]['segmentation'].shape[1] // 2, sorted_masks[NUM2][NUM]['segmentation'].shape[0] // 2
# print(f'top right distance')
# print(math.sqrt(abs(x-mid_x)**2 + abs(y-mid_y)**2))
# print(math.sqrt(abs(x-mid_x)**2 + abs(y-mid_y)**2)< perm_limit2)
# print(f'area:\t')
# print( (sorted_masks[NUM2][NUM]['area']))
# print(sorted_masks[NUM2][NUM]['bbox'])
# plt.imshow(sorted_masks[NUM2][NUM]['segmentation'])