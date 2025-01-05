
import PIL
import torch
import numpy as np
import cv2
import torchvision.transforms.functional as TF

image_he = PIL.Image.open("val_image_he.jpg")
image_ihc = PIL.Image.open("val_image_ihc.jpg")

def pil_to_canny_edges_pil(pil_image, low_threshold=100, high_threshold=200):
    numpy_image = np.asarray(pil_image).astype(np.uint8)
    grayscale_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
    np_edges = cv2.Canny(grayscale_image, low_threshold, high_threshold)
    print(np_edges.mean())
    
    return PIL.Image.fromarray(np_edges)


pil_to_canny_edges_pil(image_he, 170, 256).save("val_image_he_canny.jpg")
pil_to_canny_edges_pil(image_ihc, 60, 200).save("val_image_ihc_canny.jpg")