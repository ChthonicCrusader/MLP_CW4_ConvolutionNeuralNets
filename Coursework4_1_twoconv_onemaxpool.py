#!/disk/scratch/mlp/miniconda2/bin/python

import os
import datetime
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider
import time
print "Job Started"


assert 'MLP_DATA_DIR' in os.environ, (
    'An environment variable MLP_DATA_DIR must be set to the path containing'
    ' MLP data before running script.')
assert 'OUTPUT_DIR' in os.environ, (
    'An environment variable OUTPUT_DIR must be set to the path to write'
    ' output to before running script.')


# In[4]:

train_data = CIFAR10DataProvider('train',batch_size=50)
valid_data = CIFAR10DataProvider('valid',batch_size=50)
valid_inputs=valid_data.inputs
valid_targets=valid_data.to_one_of_k(valid_data.targets)
start_time = time.time()

# In[ ]:

print "Checkpoint 1: Data loaded successfully"
print "Start Time: "+str(start_time)

# In[16]:

# create arrays to store run train / valid set stats
num_epoch =2
#dropout = 0.2
train_accuracy = np.zeros(num_epoch)
train_error = np.zeros(num_epoch)
valid_accuracy = np.zeros(num_epoch)
valid_error = np.zeros(num_epoch)
conv1_layer = []# np.zeros(num_epoch)
conv2_layer = []# np.zeros(num_epoch)
pool1_layer =[]# np.zeros(num_epoch)
pool2_layer =[]# np.zeros(num_epoch)


#print 'Shape:%s'%str(conv1_layer.shape)
# In[8]:

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal([input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs


def conv2d(x, W, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return x


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def assign_weights(shape):
    initial = tf.truncated_normal(shape,stddev=2./(sum(shape)**0.5))
    return tf.Variable(initial)

def assign_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Create model
def conv_net(x):
    
    input_image = tf.reshape(x,[-1,32,32,3])
    print 'Input image dimension:'
    print input_image.get_shape()

    w_conv1 = assign_weights([5,5,3,32])
#    print 'conv1 weights:'
#    print w_conv1.get_shape()
    b_conv1 = assign_bias([32])
    h_conv1 = tf.nn.relu(conv2d(input_image,w_conv1)+b_conv1)
#    print 'hidden conv1 dimention'
#    print h_conv1.get_shape()

    #Second Layer:
    w_conv2 = assign_weights([5,5,32,200])
#    print 'weight conv2 dimension'
#    print w_conv2.get_shape()
    b_conv2 = assign_weights([200])
    h_conv2 = tf.nn.relu(conv2d(h_conv1,w_conv2)+b_conv2)
#    print 'conv2 hidden out dimention %s' %str(h_conv2.get_shape())
    h_pool2 = maxpool2d(h_conv2,k=4)
#    print 'h_pool dimension:%s'%str(h_pool2.get_shape())

    #Affine Layer
    w_fc1 = assign_weights([8*8*200,1000])
    b_fc1 = assign_bias([1000])
    h_pool2_flat = tf.reshape(h_pool2,[-1,8*8*200])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)


    #output layer
    w_fc2 = assign_weights([1000,10])
    b_fc2 = assign_bias([10])
    out = tf.matmul(h_fc1,w_fc2) + b_fc2
    return h_conv1,h_conv2,h_pool2,out


# In[11]:
print "Loading Model"
#defining model
with tf.name_scope('data'):
    inputs = tf.placeholder(tf.float32,[None,train_data.inputs.shape[1]], 'inputs')
    targets = tf.placeholder(tf.float32, [None,train_data.num_classes], 'targets')
#    keep_prob = tf.placeholder(tf.float32)
with tf.name_scope('model'):
    h_conv1,h_conv2,h_pool2,pred = conv_net(inputs)
with tf.name_scope('error'):
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(error)
with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[13]:

print "Adding Summaries"
#Add summary
tf.summary.scalar('error',error)
tf.summary.scalar('accuracy',accuracy)
summary_op = tf.summary.merge_all()


# In[14]:

# create objects for writing summaries and checkpoints during training
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = os.path.join(os.environ['OUTPUT_DIR'], timestamp)
checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
train_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'train-summaries'))
valid_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'valid-summaries'))
###saver = tf.train.Saver()


# In[17]:
print "Training Loop"
# create session and run training loop
sess = tf.Session()
sess.run(tf.global_variables_initializer())
step = 0
for e in range(num_epoch):
    epoch_time = time.time()
    print("Epoch started:{0}".format(e))
    for b, (input_batch, target_batch) in enumerate(train_data):
        # do train step with current batch
        conv1,conv2,pool2,_, summary, batch_error, batch_acc = sess.run(
            [h_conv1,h_conv2,h_pool2,train_step, summary_op, error, accuracy],
            feed_dict={inputs: input_batch, targets: target_batch})
        # add symmary and accumulate stats
        train_writer.add_summary(summary, step)
        train_error[e] += batch_error
        train_accuracy[e] += batch_acc
        step += 1
    # normalise running means by number of batches
    print type(conv1)
    print conv1.shape
    aa=np.array(conv1,dtype=np.float64)
    print type(aa)
    conv1_layer.append(conv1)
    conv2_layer.append(conv2)
    pool2_layer.append(pool2)
    train_error[e] /= train_data.num_batches
    train_accuracy[e] /= train_data.num_batches
    # evaluate validation set performance
    valid_summary, valid_error[e], valid_accuracy[e] = sess.run(
        [summary_op, error, accuracy],
        feed_dict={inputs: valid_inputs, targets: valid_targets})
    valid_writer.add_summary(valid_summary, step)
    # checkpoint model variables
    ###saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), step)
    # write stats summary to stdout
    print('Epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
          .format(e + 1, train_error[e], train_accuracy[e]))
    print('          err(valid)={0:.2f} acc(valid)={1:.2f}'
          .format(valid_error[e], valid_accuracy[e]))
    print('Time Taken:{0:.2f} mins'.format((epoch_time-time.time())/60))

# In[26]:

# close writer and session objects
train_writer.close()
valid_writer.close()
sess.close()


# In[27]:
end_time=time.time()-start_time
print "End Time"
print end_time/60
print "Saving Accuracies"
# save run stats to a .npz file
np.savez_compressed(
    os.path.join(exp_dir, 'exp_run_cnn_onepool_'+str(num_epoch)+'.npz'),
    train_error=train_error,
    train_accuracy=train_accuracy,
    valid_error=valid_error,
    valid_accuracy=valid_accuracy,
    conv1_layer=np.asarray(conv1_layer, dtype='float64'),
    conv2_layer=np.asarray(conv2_layer, dtype='float64'),
    pool2_layer=np.asarray(pool2_layer, dtype='float64'),
    time_taken=np.asarray([end_time], dtype='float64')
)
print('Success')
