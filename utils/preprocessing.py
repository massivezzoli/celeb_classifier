import cv2
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.transform import resize
from utils.load_data import *

def qualify_crop(entity, images_list, data_dir, face_cascade, eye_cascade, img_dims=100, detect_eyes=False):
    
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    face_images = []
    entity_list = []
    num_face = 0
    num_eyes = 0
    num_qual = 0

    #entity = df.entities[0]
    # images_list = os.listdir(data_dir + '/' + entity)
    #print(entity)
        
    for image in images_list:
        #print(image)
        image_name = data_dir + '/' + entity + '/' + image
        #print(image_name)
        img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_c = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) <= 1:
            for (x,y,w,h) in faces:
                    
                roi_gray = gray[y:y+h, x:x+w]
                #roi_color = img_c[y:y+h, x:x+w]
                    
                num_face = num_face + 1

                if detect_eyes:
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    if (len(eyes) != 2): 
                        continue

                    ex1,ey1,ew1,eh1 = eyes[0]
                    ex2,ey2,ew2,eh2 = eyes[1]
                    if np.abs( ey1 - ey2 ) > eh1 or np.abs( ex1 - ex2 ) < (ew1/2):
                        continue
                    num_eyes+=1
                
                new_img = img_c.copy()

                x,y,w,h = faces[0] 
                border = 0
                    
                new_img = cv2.resize(new_img[y-border:y+h-1+border, x-border:x+w-1+border],
                                         (img_dims, img_dims), interpolation = cv2.INTER_CUBIC)

                #crop_converted = crop_img
                face_images.append( new_img )
                entity_list.append( image )
    return face_images, entity_list

def normalize(images, mean_image, img_dims=100):
    x = np.linspace(-3.0, 3.0, img_dims)
    mu = 0.0
    sigma = 2.5

    z = (np.exp(np.negative(np.power(x - mu, 2.0) /
                       (2.0 * np.power(sigma, 2.0)))) *
         (1.0 / (sigma * np.sqrt(2.0 * 3.1415))))

    g2d = np.matmul(z.reshape(z.shape[0], 1), z.reshape(1, z.shape[0]))
    g2d = g2d  / g2d.max()

    normalized_images = []

    for input_img in list(images):
        # this_image = input_img - mean_image.reshape(img_dims, img_dims, 3)
        
        g2d_3 = np.dstack((g2d,g2d,g2d))
        masked_img = np.multiply(input_img, g2d_3)
        
        normalized_images.append(masked_img)

    normalized_images = np.array(normalized_images)
    
    return normalized_images