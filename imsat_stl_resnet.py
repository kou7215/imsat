import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data
import numpy as np
import glob
import os
import argparse
import cv2
import tensorflow_hub as hub

parser = argparse.ArgumentParser()
# data args
parser.add_argument("--input_file_path", default='/home/konosuke-a/kaggle_datasets/stl10/train_images', help="path for input data directory")
parser.add_argument("--load_model", default=False, type=bool, help="train is False, test is True")
parser.add_argument("--load_model_path", default='./stl10_logs', help="path for checkpoint")
parser.add_argument("--save_dir", default='./stl10_logs', help="path for save the model and logs")
# train conditions args
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--y_dim", type=int, default=10, help="cluster size")
parser.add_argument("--Ip", type=int, default=1, help="Power iteration num")
parser.add_argument("--eps", type=float, default=8.0, help="hyper parameter scale of returned d")
parser.add_argument("--xi", type=float, default=10, help="hyper parameter scale of initialised d")
parser.add_argument("--lam", type=float, default=0.1, help="trade-off parameter for mutual information and smooth regularization")
parser.add_argument("--mu", type=float, default=4.0, help="trade-off parameter for entropy minimization and entropy maximization")
parser.add_argument("--epoch", type=int, default=100, help="epoch")
parser.add_argument("--print_loss_freq", type=int, default=500, help="print loss EPOCH frequency")
parser.add_argument("--gpu_config", default=-1, help="0: gpu0, 1: gpu1, -1: both gpu")
a = parser.parse_args()
seed = 1145141919

for k,v in a._get_kwargs():
    print(k, '=', v)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
elif a.gpu_config == '1':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'))

def Accuracy(y,label):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def He_initializer(y):
    with tf.name_scope('He_initializer'):
        # Initializer for ReLU
        stddev = lambda in_ch: np.sqrt(2/in_ch)
        if len(y.get_shape()) == 2:
            _, in_ch = [int(i) for i in y.get_shape()]
            return tf.random_normal_initializer(0.0, stddev(in_ch))
        elif len(y.get_shape()) == 4:
            _,h,w,ch = [int(i) for i in y.get_shape()]
            in_ch = h*w*ch
            return tf.random_normal_initializer(0.0, stddev(in_ch))

def batchnorm(input):
    with tf.variable_scope('batchnorm',reuse=tf.AUTO_REUSE):
        input = tf.identity(input)
        channels = input.get_shape()[-1]    # output channel size
        print(input.get_shape())
        offset = tf.get_variable("offset_{}".format(channels), [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale_{}".format(channels), [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = 0., 1.
        # MLP
        if len(input.get_shape()) == 2: # rank 2 tensor
            mean, variance = tf.nn.moments(input, axes=[0], keep_dims=False)
        # CNN
        if len(input.get_shape()) == 4: # rank 4 tensor
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

with tf.name_scope('LoadImage'):
    input_paths = glob.glob(os.path.join(a.input_file_path, "*.png"))
    sample_num = len(input_paths)
    iteration_num = int(sample_num/a.batch_size * a.epoch)
    path_queue = tf.train.string_input_producer(input_paths, shuffle=True,seed=seed)
    reader = tf.WholeFileReader()
    paths, contents = reader.read(path_queue)
    raw_input = tf.image.decode_jpeg(contents)
    x_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
    x_input.set_shape([96, 96, 3])
    height,width,ch = x_input.get_shape()
    x_input = preprocess(x_input)
    path_batch, x_batch = tf.train.batch([paths, x_input],
                                          batch_size=a.batch_size,
                                          allow_smaller_final_batch=a.load_model==True)
    x_batch = tf.image.resize_images(x_batch,[224,224])

def Get_normalized_vector(d):
    with tf.name_scope('get_normalized_vec'):
        d /= (1e-12 + tf.reduce_sum(tf.abs(d), axis=[1,2,3], keep_dims=True))
        d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.), axis=[1,2,3], keep_dims=True))
        return d

def KL_divergence(p, q):
    # KL = ∫p(x)log(p(x)/q(x))dx
    #    = ∫p(x)(log(p(x)) - log(q(x)))dx
    kld = tf.reduce_mean(tf.reduce_sum(p * (tf.log(p + 1e-14) - tf.log(q + 1e-14)), axis=[1]))
    return kld
    

module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1", trainable=False)
def CNN(x):
    x = module(x)
    # dropout_rate
    dropout_rate = 1.0
    if a.load_model is not True:
        dropout_rate = 0.8
    with tf.variable_scope('Dense',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('layer1'):
            # [n,2048] -> [n,512]
            w = tf.get_variable(name='w1',shape=[2048,512], dtype=tf.float32, initializer=He_initializer(x))
            b = tf.get_variable(name='b1',shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.nn.leaky_relu(batchnorm(tf.matmul(x,w) + b))
        with tf.variable_scope('layer2'):
            # [n,512] -> [n,10]
            out = tf.nn.dropout(out, dropout_rate)
            w = tf.get_variable(name='w2',shape=[512,10], dtype=tf.float32, initializer=He_initializer(out))
            b = tf.get_variable(name='b2',shape=[10], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.nn.softmax(batchnorm(tf.matmul(out,w) + b))
            return out

def Generate_perturbation(x):
    d = tf.random_normal(shape=tf.shape(x))
    for i in range(a.Ip):
        d = a.xi * Get_normalized_vector(d)
        p = CNN(x)
        q = CNN(x + d)
        d_kl = KL_divergence(p,q)
        grad = tf.gradients(d_kl, [d], aggregation_method=2)[0] # d(d_kl)/d(d)
        d = tf.stop_gradient(grad)
    return a.eps * Get_normalized_vector(d)
   
def Transform(x):
    with tf.name_scope('Transform'):
        with tf.name_scope('random_cropping'):
            # random cropping params
            _,h,w,ch = x.get_shape()
            CROP_SIZE = int(h)
            SCALE_SIZE = int(h+20)
            # random cropping
            crop_offset = tf.cast(tf.floor(tf.random_uniform([2], 0, SCALE_SIZE - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
            x = tf.image.resize_images(x, [SCALE_SIZE, SCALE_SIZE], method=tf.image.ResizeMethod.AREA)
            x = tf.image.crop_to_bounding_box(x, crop_offset[0], crop_offset[1], CROP_SIZE, CROP_SIZE)
        with tf.name_scope('random_flipping'):
            x = tf.image.random_flip_left_right(x)

        with tf.name_scope('random_brightness'):
            x = tf.image.random_brightness(x, max_delta=0.2)

        with tf.name_scope('random_contrast'):
            x = tf.image.random_contrast(x,lower=0.2,upper=1.8)
            return x

def Get_VAT_loss(x,r):
    with tf.name_scope('Get_VAT_loss'):
        p = tf.stop_gradient(CNN(x))
        q = CNN(x + r)
        loss = KL_divergence(p,q)
        return tf.identity(loss, name='vat_loss')

def Get_Affine_loss(x, x_r):
    with tf.name_scope('Get_Affine_loss'):
        p = tf.stop_gradient(CNN(x))
        q = CNN(x_r)
        loss = KL_divergence(p,q)
        return tf.identity(loss, name='affine_loss')

def Entropy(p):
    with tf.name_scope('Entropy'):
        if len(p.get_shape()) == 2:
            return -tf.reduce_sum(p * tf.log(p + 1e-14),axis=1) #[n,c] -> [n]
        else:
            return -tf.reduce_sum(p * tf.log(p + 1e-14))

# ----------------- Define Model -----------------
with tf.name_scope('Model'):
    with tf.name_scope('Generate_perturbation'):
        # generate perturbation
        r_adv = Generate_perturbation(x_batch)
        # add perturbation onto x for visualize on tensorboard
        x_batch_r = x_batch + r_adv
        x_batch_affine = Transform(x_batch)

    with tf.name_scope('CNN_outputs'):
        x_batch_out = CNN(x_batch)


# -------------------- Define Loss & ACC --------------------
with tf.name_scope('loss'):
    with tf.name_scope('matual_information_loss'):
        with tf.name_scope('marginal_entropy_loss'):
            # H(Y) = -Σ p(y) * log[p(y)]
            #      = -Σ[1/N * Σ p(y|x)] * log[1/N * Σp(y|x)]
            p_y = tf.reduce_mean(x_batch_out, axis=0)   # [n,c] -> [c]
            marginal_entropy_loss = Entropy(p_y)   # [c] -> []

        with tf.name_scope('conditional_entropy_loss'):
            # H(Y|X) = 1/N*Σ [Σ -p(y|x) * log[p(y|x)] ]
            c_entropy_loss = tf.reduce_mean(Entropy(x_batch_out), axis=0)   # [n,c] -> [n,] -> []

        matual_information_loss = marginal_entropy_loss - a.mu*c_entropy_loss
        
    with tf.name_scope('vat_loss'):
        vat_loss = Get_VAT_loss(x_batch, r_adv)

    with tf.name_scope('affine_loss'):
        affine_loss = Get_Affine_loss(x_batch, x_batch_affine)

    with tf.name_scope('total_loss'):
        losses = vat_loss + affine_loss - a.lam * matual_information_loss
#        losses = vat_loss - a.lam * matual_information_loss

# -------------------- Define Optimizers --------------
with tf.name_scope('Optimizer'):
    trainable_vars_list = [var for var in tf.trainable_variables()]
    print('trainable_vars_list')
    for var in trainable_vars_list:
        print(var)
    adam = tf.train.AdamOptimizer(0.0002,0.5)
    gradients_vars = adam.compute_gradients(losses, var_list=trainable_vars_list)
    train_op = adam.apply_gradients(gradients_vars)

# ---------------------------- Define Summaries ------------------------- #
with tf.name_scope('summary'):
    with tf.name_scope('Input_summaies'):
        tf.summary.image('x_batch', tf.image.convert_image_dtype(deprocess(x_batch), dtype=tf.uint8, saturate=True))
        tf.summary.image('x_batch_r', tf.image.convert_image_dtype(deprocess(x_batch_r), dtype=tf.uint8, saturate=True))
        tf.summary.image('x_batch_affine', tf.image.convert_image_dtype(deprocess(x_batch_affine), dtype=tf.uint8, saturate=True))

    with tf.name_scope('Loss_summary'):
        tf.summary.scalar('marginal_entropy_loss', marginal_entropy_loss)
        tf.summary.scalar('conditional_entropy_loss', c_entropy_loss)
        tf.summary.scalar('matual_information_loss', matual_information_loss)
        tf.summary.scalar('vat_loss', vat_loss)
        tf.summary.scalar('affine_loss', affine_loss)
        tf.summary.scalar('total_loss', losses)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + '/Variable_histogram', var)

    for grad, var in gradients_vars:
        tf.summary.histogram(var.op.name + '/Gradients', grad)

parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

# ------------------ Session -------------------
init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
with tf.Session(config=config) as sess:
    if a.load_model is not True:
        sess.run(init)
    
        # mkdir if not exist directory
        if not os.path.exists(a.save_dir): # NOT CHANGE
            os.mkdir(a.save_dir)
            os.mkdir(os.path.join(a.save_dir,'summary'))
            os.mkdir(os.path.join(a.save_dir,'model'))
        
        # remove old summary if already exist
        if tf.gfile.Exists(os.path.join(a.save_dir,'summary')):    # NOT CHANGE
            tf.gfile.DeleteRecursively(os.path.join(a.save_dir,'summary'))
        
        # merging summary & set summary writer
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(a.save_dir,'summary'), graph=sess.graph)
        
        # train
        tf.train.start_queue_runners(sess=sess)
        for step in range(iteration_num):
            # train operation
            sess.run(train_op)
            if step % a.print_loss_freq == 0:
                print('step', step)
                print('marginal entropy loss     : ', sess.run(marginal_entropy_loss))
                print('conditional entropy loss  : ', sess.run(c_entropy_loss))
                print('matual information loss   : ', sess.run(matual_information_loss))
                print('vat loss                  : ', sess.run(vat_loss))
                print('affine loss               : ', sess.run(affine_loss))
                print('total loss                : ', sess.run(losses))
                print()
                summary_writer.add_summary(sess.run(merged), step)
            if step % (a.epoch/5) == 0: 
                saver.save(sess, a.save_dir + "/model/model.ckpt")    
        saver.save(sess, a.save_dir + "/model/model.ckpt") 
    # test
    else:
        checkpoint = tf.train.latest_checkpoint(a.load_model_path + '/model')
        saver.restore(sess, checkpoint)
        print('Loaded model from {}'.format(a.load_model_path))
        tf.train.start_queue_runners(sess=sess)
        x_template = np.array([])
        clus_template = np.array([])
        for i in range(100):
            x_bch, cluster = sess.run((x_batch, x_batch_out))
            x_template = np.reshape(np.append(x_template, x_bch), [-1,96,96,3])
            clus_template = np.reshape(np.append(clus_template, cluster), [-1,10])
        
        # deprocess
        x_template = (x_template + 1) /2

        # load CIFAR10 for test
        print('cluster shape: ', cluster.shape)
        print('image shape: ', x_template.shape)
        cluster = np.argmax(clus_template,axis=1)
        print(cluster)
        CLUSTER_SHOW_NUM = 16
        clustering_results_tmp = np.full((int(96*a.y_dim),int(96*CLUSTER_SHOW_NUM),3), 255, dtype=np.uint8)
        cluster_count = np.zeros([a.y_dim],dtype=np.int32)
        for i in range(len(x_template)):
            if cluster_count.all() > CLUSTER_SHOW_NUM:
                break
            img = np.uint8(x_template[i]*255)
            img = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_CONSTANT,value=(0,0,0))
            img = cv2.resize(img, (96,96))
            clus = cluster[i]
            if cluster_count[clus] >= CLUSTER_SHOW_NUM:
                continue
            h, w = (int(96*clus), int(96*cluster_count[clus]))
            clustering_results_tmp[h:h+96, w:w+96,:] = img
            cluster_count[clus] += 1
        cv2.imwrite(a.save_dir + '/' + 'each_cluster_stl.jpg', cv2.cvtColor(clustering_results_tmp,cv2.COLOR_RGB2BGR))

