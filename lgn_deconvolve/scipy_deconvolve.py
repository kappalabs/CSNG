import os
import pickle

import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from scipy import fftpack


def convolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft * psf_fft)))


def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    x = star_fft / psf_fft
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(x)))


data_dir = os.path.join("..", "datasets")
neuron_position_file = os.path.join(data_dir, 'lgn_convolved_imagenet_val_greyscale110x110_resize110_cropped.pickle')
with open(neuron_position_file, 'rb') as f:
    dataset = pickle.load(f)

stimuli = dataset['stim'][0]
response = dataset['resp'][0]
filter = dataset['filter']

filter_pad = np.pad(filter, int((110-60)/2), mode='constant')

# star_conv = convolve(star, psf)
# star_deconv = deconvolve(star_conv, psf)

star_conv = convolve(stimuli, filter_pad)
star_deconv = deconvolve(star_conv, filter_pad)
# star_deconv = deconvolve(response, filter_pad)

f, axes = plt.subplots(2,2)
axes[0,0].imshow(stimuli)
axes[0,1].imshow(filter_pad)
axes[1,0].imshow(np.real(star_conv))
axes[1,1].imshow(np.real(star_deconv))
plt.show()
