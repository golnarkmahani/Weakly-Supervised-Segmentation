import numpy as np
import tensorflow as tf
from utils import eval_methods as EM
from utils import loss_tf as LF
from utils import util as U
from utils.process_methods import one_hot
from models.model import Model


class UNCBBOXCRFBGMODEL(Model):
    def __init__(self,
                 net,
                 x_suffix,
                 y_suffix,
                 y_old_suffix,
                 y_rect_suffix,
                 m_suffix=None,
                 dropout=0,
                 loss_function={'cross-entropy': 1.},
                 unc_dict=None,
                 crf_dict=None,
                 is_stepfunc=False):
        super().__init__(net)
        self._x_suffix = x_suffix
        self._y_suffix = y_suffix
        self._y_old_suffix = y_old_suffix
        self._y_rect_suffix = y_rect_suffix
        self._m_suffix = m_suffix
        self._unc_dict = unc_dict
        self._crf_dict = crf_dict
        self.dropout = dropout
        self._loss_function = loss_function


        self._loss_dict = {'cross-entropy': LF.cross_entropy,
                           'balance-cross-entropy': LF.balance_cross_entropy,
                           'feedback-cross-entropy': LF.feedback_cross_entropy,
                           'dice': LF.dice_coefficient,
                           'balance-dice': LF.balanced_dice_coefficient,
                           'sc-loss-original': LF.spatially_constrained_loss_orgs}


        self._is_stepfunc = is_stepfunc


    def get_grads(self, data_dict):
        xs = data_dict[self._x_suffix]


        with tf.GradientTape() as tape:


            logits = self.net(xs,  self.dropout, True, False)
            features_before_last_conv = None

            # get uncertainty and calculate the uncertainty weight map
            if self._unc_dict is not None:
                uncertainty = self._uncertainty_mc_integration(xs, self.dropout, self._unc_dict['t_stochastic'])
                if self._is_stepfunc:
                    uncertainty_weight = self._uncertainty_weight_map_stepfunction(uncertainty, self._unc_dict['threshold'])
                else:
                    uncertainty_weight = self._uncertainty_weight_map(uncertainty, self._unc_dict['alpha'], self._unc_dict['beta'])
            else:
                uncertainty_weight = None


            loss = self._get_loss(logits, data_dict, uncertainty_weight, self._crf_dict, features_before_last_conv)
        grads = tape.gradient(loss, self.net.trainable_variables)
        return grads

    def eval(self, data_dict, **kwargs):
        xs = data_dict[self._x_suffix]
        ys = data_dict[self._y_suffix]
        yolds = data_dict[self._y_old_suffix]


        if self._crf_dict['version']=='feature':
            logits, features_before_last_conv = self.net(xs, 0., False, True)
        else:
            logits = self.net(xs, 0., False, False)
            features_before_last_conv = None

        loss = self._get_loss(logits, data_dict, features_before_last_conv= features_before_last_conv)
        loss = tf.reduce_mean(loss, range(1, loss.ndim))
        loss = [loss] if loss.ndim == 0 else loss

        prob = tf.nn.softmax(logits, -1)
        pred = one_hot(np.argmax(prob, -1), list(range(ys.shape[-1])))

        acc = EM.accuracy(pred, ys)
        dice = EM.dice_coefficient(pred, ys)
        iou = EM.iou(pred, ys)
        precision = EM.precision(pred, ys)
        recall = EM.recall(pred, ys)
        sensitivity = EM.sensitivity(pred, ys)
        specificity = EM.specificity(pred, ys)
        hausdorff = EM.hausdorff_2d(pred, ys)

        uc_map = self._uncertainty_mc_integration(xs, 0.1, 2)

        eval_results = {'loss': loss,
                        'acc': acc,
                        'dice': dice,
                        'iou': iou,
                        'precision': precision,
                        'recall': recall,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'hausdorff': hausdorff
                        }

        # TODO: Write this part more efficient later

        need_imgs = kwargs.get('need_imgs', None)
        if need_imgs is not None:
            eval_results.update({'imgs': self._get_imgs_eval(xs, ys, yolds, prob, uc_map)})

        cal_unc = kwargs.get('cal_unc', False)
        if cal_unc:
            eval_results.update({'uc_map': uc_map})
            eval_results.update({'prob_map': prob})
            eval_results.update({'org_map': xs})
            eval_results.update({'gt_map': ys})
            eval_results.update({'pred_map': pred})

        need_logits = kwargs.get('need_logits', False)
        if need_logits:
            eval_results.update({'logits': logits})

        need_preds = kwargs.get('need_preds', False)
        if need_preds:
            eval_results.update({'pred': pred})

        return eval_results

    def predict(self, data_dict):
        logits = self.net(data_dict[self._x_suffix], feature_last_conv = False)
        prob = tf.nn.softmax(logits, -1)
        return prob

    def _get_loss(self, logits, data_dict, uncertainty_weight=None, crf_dict=None, features_before_last_conv=None):

        loss_data_dict = {'orgs': data_dict[self._x_suffix],
                          'labels': data_dict[self._y_suffix],
                          'masks': None,
                          'logits': logits}
        if features_before_last_conv is not None:
            loss_data_dict.update({'featurevector': features_before_last_conv})

        loss_map = self._loss_dict['cross-entropy'](loss_data_dict)
        # to apply the uncertainty weight map to total loss
        if uncertainty_weight is not None:
            y_rect = data_dict[self._y_rect_suffix]
            combined_weight = np.maximum(y_rect[..., 0], uncertainty_weight)
            loss_map = loss_map * combined_weight
        else:
            y_rect = data_dict[self._y_rect_suffix]
            loss_map = loss_map*y_rect[..., 0]

        # crf loss added
        if crf_dict is not None:
            loss_map += crf_dict['lambda'] * self._loss_dict['sc-loss-original'](loss_data_dict, sigma=crf_dict['sigma'])

        return tf.reduce_mean(loss_map, range(1, loss_map.ndim))


    def _uncertainty_mc_integration_train(self, segmentation_score, number_of_times):
        # The average score contains the probability score of each class
        segmentation_score = segmentation_score / number_of_times
        # The uncertainty will be calculated using the entropy of the segmentation score
        score_for_log = np.where(segmentation_score < 1.0e-10, 1, segmentation_score).astype(np.float64)
        is_small = np.where (np.any(segmentation_score< 1.0e-10, axis=-1),0.,1.).astype(np.float64)
        uncertainty_tmp = -np.sum(score_for_log * np.log2(score_for_log), axis=-1).astype(np.float64)
        uncertainty = uncertainty_tmp * is_small
        return uncertainty

    def _uncertainty_mc_integration(self, data, dropout, t_stochastic):
        # T stochastic Forward passes to calculate uncertainty after each iteration
        segmentation_score = None
        for _ in range(t_stochastic):
            logits = self.net(data, dropout, False, feature_last_conv = False)
            prob = tf.nn.softmax(logits, -1)
            segmentation_score = prob if segmentation_score is None else segmentation_score + prob

        uncertainty = self._uncertainty_mc_integration_train(segmentation_score, t_stochastic)
        return uncertainty

    def _uncertainty_weight_map(self, uncertainty, alpha=0.1, beta=3):
        uncertainty_weight = np.power(alpha, (np.power(np.abs(1 - uncertainty), beta)))
        return uncertainty_weight


    def _uncertainty_weight_map_stepfunction(self, uncertainty, threshold):
        uncertainty_weight = np.where(uncertainty > threshold, 0, 1)

        return uncertainty_weight

    def _get_imgs_eval(self, xs, ys, yolds, prob, unc):
        img_dict = {}
        n_class = ys.shape[-1]
        for i in range(n_class):
            img = U.combine_2d_imgs_from_tensor([xs, yolds[..., i], ys[..., i], prob[..., i]])
            img_dict.update({'class %d' % i: img})

        unc_jet = U.unc_to_heat_map(unc)
        argmax_ys = np.argmax(ys, -1)
        argmax_yolds = np.argmax(yolds, -1)
        argmax_prob = np.argmax(prob, -1)
        img = U.combine_2d_imgs_from_tensor([xs, argmax_yolds, argmax_ys, argmax_prob, unc_jet])
        img_dict.update({'argmax': img})

        return img_dict
