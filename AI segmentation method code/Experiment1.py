'''Testing for Metal Performance shader built-in Apple Silicon GPU. If works it will significantly reduce heating
issues. Code still takes 3mins per image for 32 points_per_side (SAM model parameter). Therefore 60mins for 20 images'''

'''Thanks to BachHoang, malsaidi93, adamfowler1, shunjapan and others for pointing out the change in code
required to solve mps problem with SAM.  
<https://github.com/facebookresearch/segment-anything/issues/94#issuecomment-1562127003>'''

''' Mean + Standard Deviation X
    Gaussian Blur X 
    Improcessing+otsu X
    All Images Contain Tumors X
    Full Image Used X 
    Mask Selection Method X
    Full Image Used X 
    Ellipse Drawn _/
    Weighted Addition 0.5 0.5
    Push pixels below mean
    Push Intensity 230
    Image Modality and Patient T1CE BRATS 2021 and 00002
    Number of Images Used 20'''

# IMPORT LIBRARIES: for Image Processing
import cv2 
import copy
import numpy as np
import matplotlib.pyplot as plt
#READ THE LARGEST IMAGE IN THE DATASET
image = cv2.imread('/Users/dr.rajeshsgokhale/Downloads/extensionForResearchcondaEnv/img00000-to-00005_t1ce/00002_78.png')
def some_val2(image_, centre, axes):
    # Define the center, axes lengths, and angle of the ellipse
    center = (85, 140)  # Adjust the coordinates as needed
    axes_length = (40, 30)  # Adjust the major and minor axes lengths as needed
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
    mean_plusstd = np.mean(masked_image[non_zero_mask])
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
            if np.mean(mean_ij[i][j])<pixel_value:
                # print(np.mean(mean_ij[i][j]))
                mean_ij[i][j] = 230

    ''' Weighted sum of orignal image (image_roi) with mean_ij allows to account for details in orignal
    image. The weight of image_roi is high (upto 0.95) to account for details'''

    #WEIGHTED SUM 
    image_roi =cv2.addWeighted(im, 0.5, mean_ij, 0.5, 0)

    # Checking if contour area fits the value assigned
    gray_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    
    return image_roi
RectangleCoordinates = ((66, 117), (120, 164))
EllipseArea = 1950
EllipseInfo_Center_Axes = ((93, 140), (27, 23))
ResultTuple =  (((66, 117), (120, 164)), 1950, ((93, 140), (27, 23)))
d_new = []
tup_temp = (in1, in2, in3, in4, area, centre, axes) =  66, 120, 117, 164, 1950, (93, 140), (27, 23)
d_new.append(tup_temp) 
# FIND THE GLOBAL INTENSITY FOR PROCESSING
mean_intensity_global = some_val2(image[117:164+1,66:120+1,:],(93, 140), (27, 23))
plt.imshow(some_val(image, mean_intensity_global))
import os

# Define the parent folder directory
parent_folder_dir = '/Users/dr.rajeshsgokhale/Downloads/extensionForResearchcondaEnv/img00000-to-00005_t1ce/'

# List all files in the parent folder
all_files = os.listdir(parent_folder_dir)

# Filter files that start with '00002' and have the '.png' extension
image_files = [parent_folder_dir+ file for file in all_files if file.startswith('00002') and file.endswith('.png')]

# Print the list of files
print(image_files)


# Show first few images 
n = len(image_files[:20])
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
    image_roi = some_val(image[in3:in4+1,in1:in2+1,:],meanintensity_)
    # mean_intensity_ = some_val2(image_roi, cen, axe)
    # image_roi = some_val(image_roi, mean_intensity_)
    imageHolder.append(image_roi)
# show images
n = len(imageHolder)
rows = int(np.ceil(n/4))
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
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
    print('Using MPS: True')

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

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
    if i ==1:
        break
    image = imageHolder[i]
    start_time = time.time()
    masks = mask_generator_.generate(image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time// 60)
    elapsed_seconds = int(elapsed_time % 60)
    print(f'Mask generated\t{i}\nTime taken\t{elapsed_minutes}min {elapsed_seconds}secs\nNumber of Annotations\t{len(masks)}')

    MasksSAM.append(masks)

#VISUALISE IMAGES
import supervision as sv

#GET MASK MASKANNOTATOR
mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)

#GET DETECTIONS FROM SAM_RESULT
detections1 = sv.Detections.from_sam(sam_result=MasksSAM[0])

#GET ANNOTATED IMAGE FROM DETECTIONS AND IMAGE_ROI
annotated_image1 = mask_annotator.annotate(scene=image_roi.copy(), detections=detections1)

#PLOT IMAGES IN A GRID
sv.plot_images_grid(
    #PUT SELECTED IMAGES IN A LIST
    images=[image_roi, annotated_image1],
    #NUMBER OF IMAGES 1ROW*2COLUMN 
    grid_size=(1, 2),
    # IMAGE TITLES
    titles=['source image', 'segmented image'])    


# plt.imshow(MasksSAM[0]['segmentation'])