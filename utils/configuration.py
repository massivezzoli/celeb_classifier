import os
import tensorflow as tf

face_cascade_path = './cv2/haarcascade_frontalface_default.xml'
eye_cascade_path = './cv2/haarcascade_eye.xml'

retrain_path = "~/tensorflow/bazel-bin/tensorflow/examples/image_retraining/retrain"
data_dir = "/mnt/data/images/"
#data_dir = "/home/data/images/"
trained_dir = "/mnt/data/corpi/aligned_multi_200_50/"
#trained_dir = "/home/data/gaussian_masked_10x50/"
#trained_dir = "/home/data/aligned/"
#trained_dir = 'home/data/incep_runs/corpus/celeb_cropped/'

corpus_dir = os.path.join(trained_dir, "corpus")
# bottle_dir = None
bottle_dir = os.path.join(trained_dir, "bottleneck")
# output_graph_path = os.path.join(trained_dir, "output_graph.pb")
# output_labels_path = os.path.join(trained_dir, "output_labels.txt")

#log_dir =  "/home/tensorlogs/massivezzoli"
#log_dir = "/home/logs"

mean_image_path = "./mean_image.txt"

#num_entities = 20
#num_images = 50
rnd_seed = 42
img_dims = 150

normalized = True

split = [0.7, 0.15, 0.15]

# Training
#learning_rate = 1e-2
#epochs = 151
#batch_size = 10

# Model
#FC_only = False
#activation_conv = tf.nn.relu
#activation_fc = tf.nn.sigmoid
#Conv parameters
#n_filters = [32, 32, 8]  #filter output sizes
#filter_sizes = [4, 4, 4]
#filter_strides = [[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1]]
#maxpool parameters
#pool = [1, 1, 1]
#psize = [[1,2,2,1], [1,2,2,1], [1,2,2,1]]
#pstrides = [[1,2,2,1], [1,2,2,1], [1,2,2,1]]

#dropout = 0.5
