import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def dict_append(old_dict, new_dict):
    if new_dict is None:
        return old_dict
    if old_dict is None or not old_dict:
        for key in new_dict:
            old_dict[key] = []

    for key in old_dict:
        assert key in new_dict, 'No key "{}" in old dict!'.format(key)
        old_dict[key].append(new_dict[key])

    return old_dict

def dict_concat(old_dict, new_dict, axis=0):
    if new_dict is None:
        return old_dict
    if old_dict is None or not old_dict:
        old_dict = new_dict
    else:
        for key in old_dict:
            assert key in new_dict, 'No key "{}" in old dict!'.format(key)
            old_v = old_dict[key]
            new_v = new_dict[key]
            old_v = [old_v] if np.ndim(old_v) == 0 else old_v
            new_v = [new_v] if np.ndim(new_v) == 0 else new_v
            old_dict[key] = np.concatenate((old_v, new_v), axis)

    return old_dict


def dict_add(old_dict, new_dict):
    if new_dict is None:
        return old_dict
    if old_dict is None:
        old_dict = new_dict
    else:
        for key in new_dict:
            old_dict[key] += new_dict[key]
    return old_dict

def dict_list2arr(d):
    for key in d:
        d[key] = np.array(d[key])

def dict_to_str(evaluation_dict, axis=0):
    if evaluation_dict is None or not evaluation_dict:
        return ''
    o_s = ''
    for key in evaluation_dict:
        value = np.array(evaluation_dict.get(key))
        if value.size >= 2:
            mean = np.mean(value, axis) #[1:]
        else:
            mean = np.mean(value, axis)
        if type(mean) in [int, float, np.float32, np.float64]:
            mean = [mean]
        if np.ndim(mean) > 1:
            continue
        mean = ['%.4f'%m for m in mean]
        o_s += '%s: '%key
        for s in mean:
            o_s += '%s '%s
        o_s += '  '
    return o_s


def recale_array(array, nmin=0, nmax=1, tmin=0, tmax=255, dtype=np.uint8):
    array = np.array(array)
    if nmin is None:
        nmin = np.min(array)
    array = array - nmin
    if nmax is None:
        nmax = np.max(array) + 1e-9
    array = array / nmax
    array = (array * (tmax - tmin)) - tmin
    return array.astype(dtype)

def gray2rgb(img):
    return np.stack((img,)*3, axis=-1)

def combine_2d_imgs_from_tensor(img_list):
    imgs = []
    combined = None
    for i, im in enumerate(img_list):
        assert len(im.shape) == 3 or len(im.shape) == 4 and im.shape[-1] in [1, 3], \
        'Only accept gray or rgb 2d images with shape [n, x, y] or  [n, x, y, c], where c = 1 (gray) or 3 (rgb).'
        if im.shape[-1] != 3:
            if len(im.shape) == 4:
                im = im[..., 0]
            im = gray2rgb(im)

        if i == 0:
            im = recale_array(im, nmin=None, nmax=None)
        else:
            im = recale_array(im)
        im = im.reshape(-1, im.shape[-2], im.shape[-1])
        imgs.append(im)
    combined = np.concatenate(imgs, 1)
    return combined

def plot_learning_curve(train_data, valid_data, number_epochs, path):
    # create a color palette
    palette = plt.get_cmap('Set1')
    plt.style.use('seaborn-darkgrid')

    class_number = 1
    train_mean_dice = []
    valid_mean_dice = []
    train_mean_loss = []
    valid_mean_loss = []
    for epoch in range(0, number_epochs):
        train_mean_dice.append(np.mean(train_data['dice'][epoch, ..., class_number]))
        valid_mean_dice.append(np.mean(valid_data['dice'][epoch, ..., class_number]))

        train_mean_loss.append(np.mean(train_data['loss'][epoch]))
        valid_mean_loss.append(np.mean(valid_data['loss'][epoch]))

    plt.plot(range(1, number_epochs+1), train_mean_dice, label='Train')
    plt.plot(range(1, number_epochs+1), valid_mean_dice, label='Validation')
    plt.legend(loc='lower right', bbox_to_anchor=(0, 1.02, 2, 0.2), mode='expand')
    plt.title('baseline_res --perfect label', loc='right', fontsize=12, fontweight='bold', color=palette(9))
    plt.xlabel("Epoch#")
    plt.ylabel("Dice")
    plt.savefig(path + "baseline_ResUnet_perfectlabel_dice.png")
    plt.close()

    plt.plot(range(1, 31), train_mean_loss, label='Train')
    plt.plot(range(1, 31), valid_mean_loss, label='Validation')
    plt.legend(loc='lower right', bbox_to_anchor=(0, 1.02, 2, 0.2), mode='expand')
    plt.title('baseline_res --perfect label', loc='right', fontsize=12, fontweight='bold', color=palette(9))
    plt.xlabel("Epoch#")
    plt.ylabel("Loss")
    plt.savefig(path + "baseline_ResUnet_perfectlabel_loss.png")
    plt.close()

    return

def unc_to_heat_map(unc):
    heatmap_unc = []
    for i in range(unc.shape[0]):
        GBR_heatmap = cv2.applyColorMap((unc[i, ...] * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_unc.append(cv2.cvtColor(GBR_heatmap, cv2.COLOR_BGR2RGB))

    return np.array(heatmap_unc)