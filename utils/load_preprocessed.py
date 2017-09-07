from skimage.io import imread
import numpy as np
import os
import re

def load_preprocessed_data(corpus_dir, bottle_dir, img_dims, apply_gaussian=False):

    entities = [f for f in os.listdir(corpus_dir) if re.match(r'm\.*', f)]
    entities.sort()

    labels_accumulated = []
    image_np_list = []
    bottle_np_list = []
    for entity in entities:
        entity_path = os.path.join(corpus_dir, entity)
        images = [f for f in os.listdir(entity_path) if re.search(r'\.jpg$', f)]
        images.sort()
        #print("images:", images)

        # load bottle
        for image in images:
            image_np_list.append(imread(os.path.join(entity_path, image)))
            bottle_path = os.path.join(bottle_dir, entity, image+'.txt')
            #bottle_np_list.append(np.loadtxt(os.path.join(os.path.join(bottle_dir, entity), image+'.txt')))
            bottle_np_list.append(np.loadtxt(bottle_path, delimiter=','))

        # labels
        labels_accumulated += [entity]*len(images)


    if apply_gaussian == True:
        print("Applied Gaussian")
        Xs = gaussian(image_np_list, img_dims)
    else:
        Xs = np.array(image_np_list)

    Ys = np.array(labels_accumulated)
    Zs = np.array(bottle_np_list)
    return Xs, Ys, Zs


def gaussian(images, img_dims):
    x = np.linspace(-3.0, 3.0, img_dims)
    mu = 0.0
    sigma = 2.5

    z = (np.exp(np.negative(np.power(x - mu, 2.0) /
                       (2.0 * np.power(sigma, 2.0)))) *
         (1.0 / (sigma * np.sqrt(2.0 * 3.1415))))

    g2d = np.matmul(z.reshape(z.shape[0], 1), z.reshape(1, z.shape[0]))
    g2d = g2d  / g2d.max()
    g2d_3 = np.dstack((g2d,g2d,g2d))

    normalized_images_list = []

    for input_img in images:
        # this_image = input_img - mean_image.reshape(img_dims, img_dims, 3)

        masked_img = np.multiply(input_img, g2d_3)
        #masked_img /= 255   #scale between 0 and 1
        normalized_images_list.append(masked_img)

    normalized_images = np.array(normalized_images_list)

    return normalized_images
