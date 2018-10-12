import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data
import numpy as np
import glob
import os
import argparse
import cv2
from munkres import Munkres, print_matrix

parser = argparse.ArgumentParser()
# data args
parser.add_argument("--load_model", default=False, type=bool, help="train is False, test is True")
parser.add_argument("--load_model_path", default='./cifar10_logs', help="path for checkpoint")
parser.add_argument("--save_dir", default='./cifar10_logs', help="path for save the model and logs")
# train conditions args
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--y_dim", type=int, default=10, help="cluster size")
parser.add_argument("--Ip", type=int, default=1, help="Power iteration num")
parser.add_argument("--eps", type=float, default=8.0, help="hyper parameter scale of returned d")
parser.add_argument("--xi", type=float, default=1e-6, help="hyper parameter scale of initialised d")
parser.add_argument("--lam", type=float, default=0.1, help="trade-off parameter for mutual information and smooth regularization")
parser.add_argument("--mu", type=float, default=4.0, help="trade-off parameter for entropy minimization and entropy maximization")
parser.add_argument("--epoch", type=int, default=100, help="epoch")
parser.add_argument("--print_loss_freq", type=int, default=500, help="print loss EPOCH frequency")
parser.add_argument("--gpu_config", default=-1, help="0: gpu0, 1: gpu1, -1: both gpu")
a = parser.parse_args()

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

#def Accuracy(y, label):
#    pred = tf.cast(tf.argmax(y,1), tf.float32)
#    t = tf.cast(tf.argmax(label,1), tf.float32)
#    with tf.variable_scope('mat', reuse=False):
#        mat = tf.get_variable(name='mat',shape=[a.y_dim, a.y_dim], initializer=tf.zeros_initializer(),trainable=False)
#    for i in range(a.y_dim):
#        for j in range(a.y_dim):
#            tf.assign(mat[i,j], tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(pred,i), tf.equal(t,j)),tf.float32)))
#    indice = Munkres().compute(-mat)
#    corresp = []
#    for i in range(a.y_dim):
#        corresp.append(indice[i][1])
#
#    pred_corresp = [corresp[int(predicted)] for predicted in pred]
#    acc = tf.reduce_mean(tf.equal(pred_corresp, t)) # TODO modify
#    return acc

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

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
    

def CNN(x):
    stride = 2
    # dropout_rate
    dropout_rate = 1.0
    if a.load_model is not True:
        dropout_rate = 0.5
    with tf.variable_scope('CNN',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('layer1'):
            # [n,32,32,1] -> [n,16,16,64]
            padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            w = tf.get_variable(name='w1',shape=[4,4,3,64], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b = tf.get_variable(name='b1',shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.nn.leaky_relu(batchnorm(tf.nn.conv2d(padded, w, [1,stride,stride,1], padding='VALID') + b), 0.2)
        with tf.variable_scope('layer2'):
            # [n,16,16,64] -> [n,8,8,128] ([n,8x8x128])
            padded = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            w = tf.get_variable(name='w2',shape=[4,4,64,128], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b = tf.get_variable(name='b2',shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.nn.leaky_relu(batchnorm(tf.nn.conv2d(padded, w, [1,stride,stride,1], padding='VALID') + b), 0.2)
        with tf.variable_scope('layer3'):
            # [n,8,8,128] -> [n,4,4,256] ([n,4x4x256])
            padded = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            w = tf.get_variable(name='w3',shape=[4,4,64,256], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b = tf.get_variable(name='b3',shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.nn.leaky_relu(batchnorm(tf.nn.conv2d(padded, w, [1,stride,stride,1], padding='VALID') + b), 0.2)
            out = tf.reshape(out, [-1,4*4*256])
        with tf.variable_scope('layer4'):
            # [n,8*8*128] -> [n,1024]
            w = tf.get_variable(name='w4',shape=[4*4*256,1024], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b = tf.get_variable(name='b4',shape=[1024], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.nn.leaky_relu(batchnorm(tf.matmul(out, w) + b), 0.2)
            # dropout
            out = tf.nn.dropout(out, dropout_rate)
        with tf.variable_scope('layer5'):
            # [n,1024] -> [n,10]
            w = tf.get_variable(name='w5',shape=[1024,10], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b = tf.get_variable(name='b5',shape=[10], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.nn.softmax(batchnorm(tf.matmul(out, w) + b))
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
    
def Get_VAT_loss(x,r):
    with tf.name_scope('Get_VAT_loss'):
        p = tf.stop_gradient(CNN(x))
        q = CNN(x + r)
        loss = KL_divergence(p,q)
        return tf.identity(loss, name='vat_loss')

def Entropy(p):
    with tf.name_scope('Entropy'):
        if len(p.get_shape()) == 2:
            return -tf.reduce_mean(p * tf.log(p + 1e-14), axis=1)
        elif len(p.get_shape()) == 1:
            return -tf.reduce_sum(p * tf.log(p + 1e-14))
        else:
            raise NotImplementedError

# ----------------- Define Model -----------------
with tf.name_scope('Model'):
    with tf.name_scope('Inputs'):
        # load inputs
        # for train
        train_x = tf.placeholder(shape=[a.batch_size,32,32,3], dtype=tf.float32, name='train_x')
        train_y = tf.placeholder(shape=[a.batch_size,10], dtype=tf.float32, name='train_y')
        # for validation
        val_x = tf.placeholder(shape=[None,32,32,3], dtype=tf.float32, name='val_x')
        val_y = tf.placeholder(shape=[None,10], dtype=tf.float32, name='val_y')
        # preprocess
        train_x = preprocess(train_x)
        val_x = preprocess(val_x)
    
    with tf.name_scope('Generate_perturbation'):
        # generate perturbation
        r_adv = Generate_perturbation(train_x)
        # add perturbation onto x for visualize on tensorboard
        train_x_r = train_x + r_adv

    with tf.name_scope('CNN_outputs'):
        # To calc acc
        train_x_out = CNN(train_x)
        val_x_out = CNN(val_x)

# -------------------- Define Loss & ACC --------------------
with tf.name_scope('loss'):
    with tf.name_scope('matual_information_loss'):
        with tf.name_scope('marginal_entropy_loss'):
            # H(Y) = -Σ p(y) * log[p(y)]
            #      = -Σ[1/N * Σ p(y|x)] * log[1/N * Σp(y|x)]
            h_y = tf.reduce_mean(train_x_out, axis=0)   # [n,c] -> [c]
            marginal_entropy_loss = Entropy(h_y)   # [n] -> []

        with tf.name_scope('conditional_entropy_loss'):
            # H(Y|X) = 1/N*Σ [Σ -p(y|x) * log[p(y|x)] ]
            c_entropy_loss = tf.reduce_mean(Entropy(train_x_out))   # [n,c] -> [n,] -> []

        matual_information_loss = marginal_entropy_loss - a.mu*c_entropy_loss
        
    with tf.name_scope('vat_loss'):
        vat_loss = Get_VAT_loss(train_x, r_adv)

    with tf.name_scope('total_loss'):
        losses = vat_loss - a.lam * matual_information_loss

with tf.name_scope('Accuracy'):
    with tf.name_scope('train_acc'):
        train_acc = Accuracy(train_x_out, train_y)

    with tf.name_scope('val_acc'):
        val_acc = Accuracy(val_x_out, val_y)


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
        tf.summary.image('train_x', tf.image.convert_image_dtype(deprocess(train_x), dtype=tf.uint8, saturate=True))
        tf.summary.image('train_x_r', tf.image.convert_image_dtype(deprocess(train_x_r), dtype=tf.uint8, saturate=True))

    with tf.name_scope('Loss_summary'):
        tf.summary.scalar('marginal_entropy_loss', marginal_entropy_loss)
        tf.summary.scalar('conditional_entropy_loss', c_entropy_loss)
        tf.summary.scalar('matual_information_loss', matual_information_loss)
        tf.summary.scalar('vat_loss', vat_loss)
        tf.summary.scalar('total_loss', losses)

    with tf.name_scope('Accuracy_summary'):
        tf.summary.scalar('train_acc', train_acc)
        tf.summary.scalar('val_acc', val_acc)
    
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
            os.mkdir(os.path.join(a.save_dir,'variables'))
            os.mkdir(os.path.join(a.save_dir,'model'))
        
        # remove old summary if already exist
        if tf.gfile.Exists(os.path.join(a.save_dir,'summary')):    # NOT CHANGE
            tf.gfile.DeleteRecursively(os.path.join(a.save_dir,'summary'))
        
        # merging summary & set summary writer
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(a.save_dir,'summary'), graph=sess.graph)
        
        # load CIFAR10 for validation
        (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = load_data()
        # one-hot
        y_train_cifar = np.identity(10)[y_train_cifar[:,0]]
        y_test_cifar  = np.identity(10)[y_test_cifar[:,0]]

        # train
        iteration_num = int(50000/a.batch_size * a.epoch)    # (iter_num) = (train sample size) * (epoch)
        iteration_num_per_epoch = int(50000/a.batch_size)    # (iter_num) = (train sample size) * (epoch)
        step = 0
        for epc in range(a.epoch):
            for bch in range(iteration_num_per_epoch):
                train_batch_idx = np.random.choice(len(x_train_cifar),a.batch_size,replace=False)
                test_batch_idx = np.random.choice(len(x_test_cifar),a.batch_size,replace=False)
                x_train_bch, y_train_bch = x_train_cifar[train_batch_idx]/255., y_train_cifar[train_batch_idx]
                x_test_bch, y_test_bch = x_test_cifar[test_batch_idx]/255., y_test_cifar[test_batch_idx]
                train_dict = {train_x:x_train_bch, train_y:y_train_bch}
                test_dict = {train_x:x_train_bch, train_y:y_train_bch, val_x:x_test_bch, val_y:y_test_bch}
                # train operation
                sess.run(train_op, feed_dict=train_dict)
                if step % a.print_loss_freq == 0:
                    print('step', step)
                    print('marginal entropy loss     : ', sess.run(marginal_entropy_loss, feed_dict=test_dict))
                    print('conditional entropy loss  : ', sess.run(c_entropy_loss, feed_dict=test_dict))
                    print('matual information loss   : ', sess.run(matual_information_loss, feed_dict=test_dict))
                    print('vat loss                  : ', sess.run(vat_loss, feed_dict=test_dict))
                    print('total loss                : ', sess.run(losses, feed_dict=test_dict))
                    print()
                    summary_writer.add_summary(sess.run(merged,feed_dict=test_dict), step)
                if step % (a.epoch/5) == 0: 
                    saver.save(sess, a.save_dir + "/model/model.ckpt")    
                step+=1
        saver.save(sess, a.save_dir + "/model/model.ckpt") 
    # test
    else:
        checkpoint = tf.train.latest_checkpoint(a.load_model_path + '/model')
        saver.restore(sess, checkpoint)
        print('Loaded model from {}'.format(a.load_model_path))
        # load CIFAR10 for test
        (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = load_data()
        x_test = x_test_cifar[:1000]/255.
        y_test = np.identity(10)[y_test_cifar[:,0]][:1000]
        test_dict = {val_x_vec:x_test_cifar, val_y:y_test_cifar}
        cluster = np.argmax(sess.run(val_x_out,feed_dict=test_dict),axis=1)
        CLUSTER_SHOW_NUM = 16
        clustering_results_tmp = np.full((int(32*a.y_dim),int(32*CLUSTER_SHOW_NUM),3), 255, dtype=np.uint8)
        cluster_count = np.zeros([a.y_dim],dtype=np.int32)
        for i in range(len(x_test_image)):
            if cluster_count.all() > CLUSTER_SHOW_NUM:
                break
            img = np.uint8(x_test_image[i]*255)
            img = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_CONSTANT,value=(255,255,255))
            img = cv2.resize(img, (32,32))
            clus = cluster[i]
            if cluster_count[clus] >= CLUSTER_SHOW_NUM:
                continue
            h, w = (int(28*clus), int(32*cluster_count[clus]))
            clustering_results_tmp[h:h+32, w:w+32,:] = img
            cluster_count[clus] += 1
        cv2.imwrite(a.save_dir + '/' + 'each_cluster_cifar10.jpg', cv2.cvtColor(clustering_results_tmp,cv2.COLOR_RGB2BGR))

