import numpy as np
import tensorflow as tf

# loss functions -------------------------------------------
def cross_entropy(data_dict):
    logits = data_dict['logits']
    labels = data_dict['labels']
    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return loss_map

def balance_cross_entropy(data_dict):
    loss_map = cross_entropy(data_dict)
    weight_map = balance_weight_map(data_dict)
    return loss_map * weight_map

def feedback_cross_entropy(data_dict, alpha=3, beta=100):
    loss_map = cross_entropy(data_dict)
    weight_map = feedback_weight_map(data_dict, alpha, beta)
    return loss_map * weight_map

def mse(logits, labels):
    if np.ndim(labels) < 2:
        labels = np.expand_dims(labels, -1) 
    loss_map = tf.reduce_mean(tf.square(logits-labels), -1)
    return loss_map


def dice_coefficient(data_dict):
    logits = tf.cast(data_dict['logits'], tf.float32)
    labels = tf.cast(data_dict['labels'], tf.float32)
    axis = tuple(range(1, len(labels.shape) - 1)) if len(labels.shape) > 2 else -1
    pred = tf.nn.softmax(logits)
    # pred = tf.one_hot(tf.argmax(logits, -1), labels.shape[-1])
    
    intersection = tf.reduce_sum(pred * labels, axis)
    sum_ = tf.reduce_sum(pred + labels, axis)
    dice = 1 - 2 * intersection / sum_
    return dice

def balanced_dice_coefficient(data_dict):
    labels = tf.cast(data_dict['labels'], tf.float32)
    dice_loss = dice_coefficient(data_dict)
    axis = tuple(range(np.ndim(labels) - 1)) if np.ndim(labels) > 2 else -1
    c = 1/(np.sum(1/(np.sum(labels, axis=axis))))
    balanced_weight = c/(np.sum(labels, axis=axis))
    dice = dice_loss * balanced_weight
    return dice

# @tf.function
def spatially_constrained_loss_orgs(data_dict, sigma=0.5, uncertainty_map=None, NoConfs=None):

    orgs = tf.cast(data_dict['orgs'], tf.float32)
    logits = tf.cast(data_dict['logits'], tf.float32)
    kernal_size = 3
    ndim = len(logits.shape)
    if type(kernal_size) is int:
        kernal_size = [1] + [kernal_size,] * (ndim-2) + [1]
    elif type(kernal_size) is list:
        kernal_size = [1] + kernal_size + [1]
    strides = [1,] * ndim
    rates = [1,] * ndim
    if uncertainty_map is None:
        probs = tf.nn.softmax(logits)
        confs = tf.reduce_max(probs, -1, keepdims=True)
    else:
        probs = tf.nn.softmax(logits)
        uncertainty = tf.expand_dims(uncertainty_map, axis=-1)
        confs = 1 - uncertainty
        confs = tf.cast(confs, tf.float32)

    arg_preds = tf.cast(tf.expand_dims(tf.argmax(probs, -1), -1), tf.float32)

    h,w,c=orgs[0].shape
    if c==3:
        orgs = orgs[...,0]*0.2989 + orgs[...,1]*0.5870 + orgs[...,2]*0.1140
        orgs = tf.expand_dims(orgs, axis=-1)


    if ndim == 4:
        p_zmask = tf.image.extract_patches(tf.ones(confs.shape), kernal_size, strides, rates, padding='SAME')
        p_confs = tf.image.extract_patches(confs, kernal_size, strides, rates, padding='SAME')
        p_orgs = tf.image.extract_patches(orgs, kernal_size, strides, rates, padding='SAME')
        p_preds = tf.image.extract_patches(arg_preds, kernal_size, strides, rates, padding='SAME')
    elif ndim == 5:
        p_zmask = tf.extract_volume_patches(tf.ones(confs.shape), kernal_size, strides, padding='SAME')
        p_confs = tf.extract_volume_patches(confs, kernal_size, strides, padding='SAME')
        p_preds = tf.extract_volume_patches(arg_preds, kernal_size, strides, padding='SAME')
        p_orgs = tf.extract_volume_patches(orgs, kernal_size, strides, padding='SAME')

    p_exp = tf.exp(-tf.square(orgs - p_orgs) / (2 * sigma ** 2))
    p_exp = p_zmask * p_exp
    p_mask = 2 * tf.cast(tf.math.equal(arg_preds, p_preds), tf.float32) - 1

    u_ij = p_exp * p_mask

    P_ij = confs * p_confs
    if NoConfs is None:
        F_ij = u_ij * P_ij
    else:
        F_ij = u_ij
    F_i = (tf.reduce_sum(F_ij, -1) - tf.reshape(confs**2, confs.shape[:-1])) / (tf.reduce_sum(p_exp, -1) - 1 + 1e-9)
    sc_loss_map = 1 - F_i

    return sc_loss_map


# ----------------------------------------------------------

# weight maps ----------------------------------------------

def balance_weight_map(data_dict, epsilon=1e-9):
    labels = data_dict['labels']
    axis = tuple(range(np.ndim(labels) - 1)) if np.ndim(labels) > 1 else -1
    c = 1/(np.sum(1/(epsilon + np.sum(labels, axis=axis))))
    weight_map = np.sum(labels * np.tile(c/(epsilon + np.sum(labels, axis=axis, keepdims=True)), list(labels.shape[0:-1]) + [1]), axis=-1)
    return weight_map

def feedback_weight_map(data_dict, alpha=3, beta=100):
    logits = data_dict['logits']
    labels = data_dict['labels']
    probs = tf.nn.softmax(logits, -1)
    p = np.sum(probs * labels, axis=-1)
    weight_map = np.exp(-np.power(p, beta)*np.log(alpha))
    return weight_map 

# ----------------------------------------------------------