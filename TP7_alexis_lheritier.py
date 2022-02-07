
import torch
import torchvision
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# load video
from imageai.Detection import ObjectDetection
import os
import numpy as np
import cv2

execution_path = os.getcwd()

# Data loading and visualization imports
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from matplotlib.pyplot import imshow, figure, subplots

# Model loading
from models.erfnet import Net as ERFNet
from models.lcnet import Net as LCNet

# utils
from functions import color_lanes, blend

# to cuda or not to cuda
if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'

# Descriptor size definition
DESCRIPTOR_SIZE = 64
# Maximum number of lanes the network has been trained with + background
NUM_CLASSES_SEGMENTATION = 5
# Maximum number of classes for classification
NUM_CLASSES_CLASSIFICATION = 3
# Image size
HEIGHT = 360
WIDTH = 640


def extract_descriptors(label, image):
    # avoids problems in the sampling
    eps = 0.01

    # The labels indices are not contiguous e.g. we can have index 1, 2, and 4 in an image
    # For this reason, we should construct the descriptor array sequentially
    inputs = torch.zeros(0, 3, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    # This is needed to keep track of the lane we are classifying
    mapper = {}
    classifier_index = 0

    # Iterating over all the possible lanes ids
    for i in range(1, NUM_CLASSES_SEGMENTATION):
        # This extracts all the points belonging to a lane with id = i
        single_lane = label.eq(i).view(-1).nonzero().squeeze()

        # As they could be not continuous, skip the ones that have no points
        if single_lane.numel() == 0 or len(single_lane.size()) == 0:
            continue

        # Points to sample to fill a squared desciptor
        sample = torch.zeros(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE)
        if torch.cuda.is_available():
            sample = sample.cuda()

        sample = sample.uniform_(0, single_lane.size()[0] - eps).long()
        sample, _ = sample.sort()

        # These are the points of the lane to select
        points = torch.index_select(single_lane, 0, sample)

        # First, we view the image as a set of ordered points
        descriptor = image.squeeze().view(3, -1)

        # Then we select only the previously extracted values
        descriptor = torch.index_select(descriptor, 1, points)

        # Reshape to get the actual image
        descriptor = descriptor.view(3, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)
        descriptor = descriptor.unsqueeze(0)

        # Concatenate to get a batch that can be processed from the other network
        inputs = torch.cat((inputs, descriptor), 0)

        # Track the indices
        mapper[classifier_index] = i
        classifier_index += 1

    return inputs, mapper


def line_detector(image_raw, image_with_object_detection):
    img = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img)

    im = im.resize((WIDTH, HEIGHT))
    im_tensor = ToTensor()(im)
    im_tensor = im_tensor.unsqueeze(0)

    # Creating CNNs and loading pretrained models
    segmentation_network = ERFNet(NUM_CLASSES_SEGMENTATION)
    classification_network = LCNet(NUM_CLASSES_CLASSIFICATION, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)

    segmentation_network.load_state_dict(torch.load('pretrained/erfnet_tusimple.pth', map_location=map_location))
    model_path = 'pretrained/classification_{}_{}class.pth'.format(DESCRIPTOR_SIZE, NUM_CLASSES_CLASSIFICATION)
    classification_network.load_state_dict(torch.load(model_path, map_location=map_location))

    segmentation_network = segmentation_network.eval()
    classification_network = classification_network.eval()

    if torch.cuda.is_available():
        segmentation_network = segmentation_network.cuda()
        classification_network = classification_network.cuda()

    # Inference on instance segmentation
    if torch.cuda.is_available():
        im_tensor = im_tensor.cuda()

    out_segmentation = segmentation_network(im_tensor)
    out_segmentation = out_segmentation.max(dim=1)[1]

    # Converting to numpy for visualization
    out_segmentation_np = out_segmentation.cpu().numpy()[0]
    out_segmentation_viz = np.zeros((HEIGHT, WIDTH, 3))

    for i in range(1, NUM_CLASSES_SEGMENTATION):
        rand_c1 = random.randint(1, 255)
        rand_c2 = random.randint(1, 255)
        rand_c3 = random.randint(1, 255)
        out_segmentation_viz = color_lanes(
            out_segmentation_viz, out_segmentation_np,
            i, (rand_c1, rand_c2, rand_c3), HEIGHT, WIDTH)

    # im_seg = blend(im, out_segmentation_viz)

    descriptors, index_map = extract_descriptors(out_segmentation, im_tensor)

    # Inference on descriptors
    classes = classification_network(descriptors).max(1)[1]
    print(index_map)
    print(classes)

    # Class visualization
    out_classification_viz = np.zeros((HEIGHT, WIDTH, 3))

    for i, lane_index in index_map.items():
        if classes[i] == 0:  # Continuous blue
            color = (255, 0, 0)
        elif classes[i] == 1:  # Dashed green
            color = (0, 255, 0)
        elif classes[i] == 2:  # Double-dashed red
            color = (0, 0, 255)
        else:
            raise ArithmeticError
        image_with_object_detection[out_segmentation_np == lane_index] = color
    # imshow(out_classification_viz.astype(int))
    # plt.show()

    return image_with_object_detection


full_video = cv2.VideoCapture("road.mp4")  # On récupère la vidéo à l'aide d'opencv
list_of_image = []
cpt_max = 0
while full_video.isOpened():
    ret, frame = full_video.read()  # On récupère chaque image de la vidéo
    if not ret:
        break
    if cpt_max % 10 == 0:
        list_of_image.append(frame)  # On stock dans un tableau une image sur 10
    cpt_max = cpt_max + 1

full_video.release()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "pretrained-yolov3.h5"))
detector.loadModel()

list_of_compute_image = []

for i in range(len(list_of_image)):
    image_compute = None
    print("image ", i)
    image_compute, detections = detector.detectObjectsFromImage(input_image=list_of_image[i], input_type="array",
                                                                output_type="array", minimum_percentage_probability=30)

    if image_compute is not None:
        list_of_compute_image.append(image_compute)
    else:
        print("ATTENTION ! Image numéro ", i, " non traitées.")


height, width, _ = list_of_compute_image[0].shape
size = (width, height)
print(size)
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('vehicles_detection.avi', fourcc, 10, size)
for i in range(len(list_of_compute_image)):
    out.write(list_of_compute_image[i])

out.release()

out = cv2.VideoWriter('road_detection.avi', fourcc, 10, size)
for i in range(len(list_of_compute_image)):
    line_compute = line_detector(list_of_image[i], list_of_compute_image[i])
    out.write(line_compute)

out.release()
