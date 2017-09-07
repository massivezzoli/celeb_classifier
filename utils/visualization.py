import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.colors as colors

def montage(images, saveto='montage.png'):
    """Draw all images as a montage separated by 1 pixel borders.
    Also saves the file to the destination specified by `saveto`.
    Parameters
    ----------
    images : numpy.ndarray
        Input array to create montage of.  Array should be:
        batch x height x width x channels.
    saveto : str
        Location to save the resulting montage image.
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    else:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    # plt.imsave(arr=m, fname=saveto)
    return m

def montage_filters(W):
    """Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders.
    Parameters
    ----------
    W : Tensor
        Input tensor to create montage of.
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    W = np.reshape(W, [W.shape[0], W.shape[1], 1, W.shape[2] * W.shape[3]])
    n_plots = int(np.ceil(np.sqrt(W.shape[-1])))
    m = np.ones(
        (W.shape[0] * n_plots + n_plots + 1,
         W.shape[1] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < W.shape[-1]:
                m[1 + i + i * W.shape[0]:1 + i + (i + 1) * W.shape[0],
                  1 + j + j * W.shape[1]:1 + j + (j + 1) * W.shape[1]] = (
                    np.squeeze(W[:, :, :, this_filter]))
    return m

def color_montage(images, saveto='montage.png'):
   """Draw all images as a montage separated by 1 pixel borders.
   Also saves the file to the destination specified by `saveto`.
   Parameters
   ----------
   images : numpy.ndarray
       Input array to create montage of.  Array should be:
       batch x height x width x channels.
   saveto : str
       Location to save the resulting montage image.
   Returns
   -------
   m : numpy.ndarray
       Montage image.
   """
   if isinstance(images, list):
       images = np.array(images)
   img_h = images.shape[1]
   img_w = images.shape[2]
   n_plots = int(np.ceil(np.sqrt(images.shape[0])))
   m = np.ones(
       (images.shape[1] * n_plots + n_plots + 1,
        images.shape[2] * n_plots + n_plots + 1, 3), dtype = "uint8")

   for i in range(n_plots):
       for j in range(n_plots):
           this_filter = i * n_plots + j
           if this_filter < images.shape[0]:
               this_img = images[this_filter]
               m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                 1 + j + j * img_w:1 + j + (j + 1) * img_w,:] = this_img
   # plt.imsave(arr=m, fname=saveto)
   return m