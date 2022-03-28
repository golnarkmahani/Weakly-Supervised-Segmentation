import argparse
import scipy.io as sio
import tensorflow as tf
from utils import util as U
from core.trainer_tf import Trainer
from core.data_provider import DataProvider
from core.data_processor import SimpleImageProcessor

from models.model_ucrf2d import UNCBBOXCRFBGMODEL
from nets_tf.unet2d import UNet2D
from core.learning_rate import StepDecayLearningRate

parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('-bs', '--batch_size', type=int, default=10, help='batch size')
parser.add_argument('-mbs', '--minibatch_size', type=int, default=5, help='mini-batch size')
parser.add_argument('-ebs', '--eval_batch_size', type=int, default=2, help='mini-batch size')
parser.add_argument('-ef', '--eval_frequency', type=int, default=1, help='frequency of evaluation within training')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('-logt', '--logits_times', type=int, default=1, help='number of logits for loss calc')
parser.add_argument('-out', '--output_path', type=str, default='[path]')
args = parser.parse_args()


# Dataset
skin_mat = sio.loadmat("../skin_mat/1820_250_524_Skin_256_256_70_10_20_sets.mat")
train_set = ['../2D_ISIC_2018_256_rect_lab/' + x for x in skin_mat['train']]
valid_set = ['../2D_ISIC_2018_256_rect_lab/' + x for x in skin_mat['val']]
test_set = ['../2D_ISIC_2018_256_rect_lab/' + x for x in skin_mat['test']]



output_path = args.output_path

org_suffix = '.jpg'
lab_suffix = '_segmentation_rect_lab.png'
old_lab_suffix = '_segmentation.png'
rect_lab_suffix = '_segmentation_rect_lab.png'

pre = {org_suffix: ['zero-mean', ('channelcheck', 3)],
       lab_suffix: [('one-hot', 2), ('channelcheck', 2)],
       old_lab_suffix: [('one-hot', 2), ('channelcheck', 2)],
       rect_lab_suffix: [('one-hot', 2), ('channelcheck', 2)]}

processor = SimpleImageProcessor(pre=pre)

train_provider = DataProvider(train_set, [org_suffix, lab_suffix, old_lab_suffix, rect_lab_suffix],
                        is_pre_load=False,
                        is_shuffle=True,
                        processor=processor)


validation_provider = DataProvider(valid_set, [org_suffix, old_lab_suffix, old_lab_suffix, rect_lab_suffix],
                        is_pre_load=False,
                        processor=processor)

# model test
unet = UNet2D(n_class=2, n_layer=5, root_filters=16, use_bn=True, use_res=True)


# stepfunc
unc_dict = {'dropout': 0.1, 'threshold': 0.1, 't_stochastic': 20}

# CRF parameters
crf_dict = {'lambda': 1, 'sigma': 1.5}


model = UNCBBOXCRFBGMODEL(unet,
                        org_suffix,
                        lab_suffix,
                        old_lab_suffix,
                        rect_lab_suffix,
                        dropout=0.1,
                        unc_dict=unc_dict,
                        crf_dict = crf_dict,
                        is_stepfunc=True)

trainer = Trainer(model)

lr = StepDecayLearningRate(learning_rate=args.learning_rate,
                               decay_step=10,
                               decay_rate=0.8,
                               data_size=train_provider.size,
                               batch_size=args.batch_size)

optimizer = tf.keras.optimizers.Adam(lr)
update_gt = None
train_eval_dict, valid_eval_dict = trainer.train(train_provider, validation_provider,
                                   epochs=args.epochs,
                                   batch_size=args.batch_size,
                                   mini_batch_size=args.minibatch_size,
                                   output_path=output_path,
                                   optimizer=optimizer,
                                   learning_rate=lr,
                                   eval_frequency=args.eval_frequency,
                                   is_save_train_imgs=False,
                                   is_save_valid_imgs=True)


# eval test & pre load test
model._y_suffix = old_lab_suffix
model._y_rect_suffix = old_lab_suffix
pre = {org_suffix: ['zero-mean', ('channelcheck', 3)],
       old_lab_suffix: [('one-hot', 2), ('channelcheck', 2)]}

processor = SimpleImageProcessor(pre=pre)

test_provider = DataProvider(test_set, [org_suffix, old_lab_suffix],
                        is_pre_load=False,
                        processor=processor)

trainer.restore(output_path + '/ckpt/final')
eval_dcit = trainer.eval(test_provider, batch_size=args.eval_batch_size, cal_unc=False)
with open(output_path + '/test_eval.txt', 'a+') as f:
    f.write('final    :' + U.dict_to_str(eval_dcit) + '\n')
sio.savemat(args.output_path + '/final_results.mat', eval_dcit)






