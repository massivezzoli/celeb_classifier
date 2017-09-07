import os
import re
import numpy as np
import pandas as pd
import cv2
import _pickle as pickle
# import matplotlib.pyplot as plt
from utils.preprocessing import *

def get_dataframe(data_dir, face_cascade, eye_cascade, num_entities=4, num_images=25, rnd_seed=1, img_dims=100):

    #Search for directories that start with 'm.'
    entities = [f for f in os.listdir(data_dir) if re.match(r'm\.*', f)]
    assert(num_entities <= len(entities))

    if rnd_seed:
        np.random.seed(rnd_seed)
    celebs = np.random.choice(entities, len(entities), replace=False)

    imgs = []
    accepted = []
    rejected = []

    print('Getting dataframe...')
    for name in celebs:
        samples = os.listdir(data_dir + name)

        if num_images <= len(samples):
            
            _, qualified_list = qualify_crop(name, samples, data_dir, face_cascade, eye_cascade, img_dims, detect_eyes=True)

            if num_images <= len(qualified_list):
                if rnd_seed:
                    np.random.seed(rnd_seed)
                qualified_list = np.random.choice(qualified_list, len(qualified_list), replace=False)
                qualified = qualified_list[:num_images]
                samples_list = [name]*len(qualified)
                pair = list(zip(samples_list, qualified))

                accepted.append(name)

                imgs = imgs + pair
            else:
                rejected.append(name)
        else:
            rejected.append(name)

        if len(accepted) == num_entities:
            break

    if len(accepted) != num_entities:
        print("Only {} entities are loaded.".format(len(accepted)))

    df = pd.DataFrame(imgs, columns=['entities', 'images'])
    print('Got dataframe.')

    return df

def get_all_images(df, data_dir, face_cascade, eye_cascade, retrain_path=None, corpus_dir=None, bottle_dir=None, img_dims=100, normalized=None, mean_image=None):

    all_images = []
    all_labels = []
    all_bottles = []

    print('Getting all images...')
    for entity in list(set(df.entities)):

        Xs, Ys, Zs = get_images(df, entity, data_dir, face_cascade, eye_cascade, retrain_path, corpus_dir, bottle_dir, img_dims, normalized, mean_image)

        all_images = all_images + list(Xs)
        all_labels = all_labels + Ys
        
        if bottle_dir != None:
            all_bottles = all_bottles + list(Zs)

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    if bottle_dir != None:
        all_bottles = np.array(all_bottles)
    else:
        all_bottles = None

    print('Got all images.')

    return all_images, all_labels, all_bottles

def get_images(df, entity, data_dir, face_cascade, eye_cascade, retrain_path=None, corpus_dir=None, bottle_dir=None, img_dims=100, normalized=None, mean_image=None):

    images_list = df[df.entities == entity].images

    face_images, face_images_list = qualify_crop(entity, images_list, data_dir, face_cascade, eye_cascade, img_dims)

    if normalized == True:
        face_images = normalize(face_images, mean_image, img_dims)

    Xs = np.array(face_images)
    Ys = [entity] * len(images_list)
    Zs = []
    
    if corpus_dir != None:
        
        destin_dir = os.path.join(corpus_dir, entity)

        if not os.path.isdir(destin_dir):
            os.makedirs(destin_dir)

        for idx, image in enumerate(face_images):
            if not os.path.exists(os.path.join(destin_dir, face_images_list[idx])):
                cv2.imwrite(os.path.join(destin_dir, face_images_list[idx]), image)
            # if not os.path.exists(os.path.join(destin_dir, face_images_list[idx]+'.pkl')):
            #     pickle.dump(image, open(os.path.join(destin_dir, face_images_list[idx]+'.pkl'), 'wb'))
    
    if bottle_dir != None:
        # if not os.path.isdir(os.path.join(bottle_dir, entity)):
        #     bazel_script = retrain_path+" --image_dir="+corpus_dir+" --bottleneck_dir="+bottle_dir
        #     os.system(bazel_script)
        for image in images_list:
            #image_name = os.path.join(bottle_dir, entity, image + '.pkl.txt')
            image_name = os.path.join(bottle_dir, entity, image + '.txt')
            with open(image_name, 'r') as bottleneck_file:
                bottleneck_string = bottleneck_file.read()
                bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
                Zs.append(bottleneck_values)

    return Xs, Ys, Zs