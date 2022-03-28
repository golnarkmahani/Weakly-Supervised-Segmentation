import numpy as np
import sklearn.metrics as skm
import scipy.spatial.distance as sci
import SimpleITK as sitk
import medpy.metric as medmetric
# evaluation -----------------------------------------------
def softmax(prob_map, axis=-1):
    e = np.exp(prob_map - np.max(prob_map))
    return e / np.sum(e, axis, keepdims=True)
    
def cross_entropy(pred, gt, epsilon=1e-9):
    axis = tuple(range(np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    ce = -np.sum(gt * np.log(pred + epsilon), axis) / pred.shape[0]
    return ce

def mean_squared_error(pred, gt):
    if np.ndim(gt) < 2:
        gt = np.expand_dims(gt, -1) 
    mse = np.mean(np.square(pred - gt), -1)
    return mse

def true_positive(pred, gt):
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    return np.sum(np.logical_and(pred == gt, gt == 1), axis)

def true_negative(pred, gt):
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    return np.sum(np.logical_and(pred == gt, gt == 0), axis)

def false_positive(pred, gt):
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    return np.sum(np.logical_and(pred != gt, pred == 1), axis)

def false_negative(pred, gt):
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    return np.sum(np.logical_and(pred != gt, pred == 0), axis)

def precision(pred, gt, epsilon=1e-9):
    tp = true_positive(pred, gt)
    fp = false_positive(pred, gt)
    return tp / (tp + fp + epsilon)

def recall(pred, gt, epsilon=1e-9):
    tp = true_positive(pred, gt)
    fn = false_negative(pred, gt)
    return tp / (tp + fn + epsilon)

def sensitivity(pred, gt, epsilon=1e-9):
    return recall(pred, gt, epsilon)

def specificity(pred, gt, epsilon=1e-9):
    tn = true_negative(pred, gt)
    fp = false_positive(pred, gt)
    return tn / (tn + fp + epsilon)

def accuracy(pred, gt):
    """ equal(pred, gt) / all(pred, gt)
        (tp + tn) / (tp + tn + fp + fn)
    """
    axis = tuple(range(1, np.ndim(pred)))# if np.ndim(pred) > 1 else -1
    return np.mean(np.equal(pred, gt), axis)

def dice_coefficient(pred, gt, epsilon=1e-9):
    """ 2 * intersection(pred, gt) / (pred + gt) 
        2 * tp / (2*tp + fp + fn)
    """
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    intersection = np.sum(pred * gt, axis)
    sum_ = np.sum(pred + gt, axis)
    return 2 * intersection / (sum_ + epsilon)

def iou(pred, gt, epsilon=1e-9):
    """ intersection(pred, gt) / union(pred, gt)
        tp / (tp + fp + fn)
    """
    axis = tuple(range(1, np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    intersection = np.sum(pred * gt, axis)
    union = np.sum(pred + gt, axis) - intersection
    return intersection / (union + epsilon)

def hausdorff_2d_old(pred, gt):
    HDis = []
    for i in range(0, pred.shape[0]):
        HDis.append(max(sci.directed_hausdorff(pred[i, ..., 1], gt[i, ..., 1])[0], sci.directed_hausdorff(gt[i, ..., 1], pred[i, ..., 1])[0]))
    return HDis

def hausdorff_2d(pred, gt):
    #The hausdorff with stk package
    HDis = []
    for i in range(0, pred.shape[0]):
        if np.sum(pred[i,...,1]) == 0:
            #HDis.append(-1)
            continue
        if np.sum(gt[i,...,1]) == 0:
            #HDis.append(-1)
            continue
        labelPred = sitk.GetImageFromArray(pred[i, ..., 1], isVector=False)
        labelTrue = sitk.GetImageFromArray(gt[i, ..., 1], isVector=False)
        hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
        hausdorffcomputer.Execute(labelTrue, labelPred)
        HDis.append(hausdorffcomputer.GetHausdorffDistance())

    return HDis


def assd_metric(pred, gt):
    assdmet =[]
    for i in range(0, pred.shape[0]):
        assdmet.append(medmetric.assd(pred[i, ..., 1],gt[i, ..., 1]))
    return assdmet

def auc(pred, gt):
    pass
# ----------------------------------------------------------


