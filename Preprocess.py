import numpy as np
from Util import fft_filter_prior_input, standardize_numpy, fft_filter, samples_multi, normalize_img
import torch
import torch.nn.functional as F
import cv2


class PreProcess(torch.nn.Module):
    def __init__(self, sample_num):
        super(PreProcess, self).__init__()
        self.sample_n = sample_num

    def forward(self, inputs):
        _, _, h = inputs.shape
        inputs = inputs.reshape(-1, 1, 512, 512)
        inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
        inputs = F.interpolate(inputs, [1024, 1024], mode='bicubic')
        inputs = inputs.numpy().reshape(-1, 1024, 1024)
        samples = samples_multi(inputs=inputs, sample_num=self.sample_n, block_size=8, pos_noise=True)
        f = fft_filter(samples)
        f_s = standardize_numpy(f)
        return inputs, samples, f_s


# imitate sampling process
data_dir = "data"
test_dir = "test"
sample_num = 3
data = np.load('data/test/stem_gt.npy')
data = standardize_numpy(data)
pre_process = PreProcess(sample_num)
inter, samples, f = pre_process(data)
inter_dir = 'data/test/stem_interpolate.npy'
samples_test_dir = 'data/test/stem_samples_%d.npy' % sample_num
f_test_dir = 'data/test/stem_f_%d.npy' % sample_num
np.save(inter_dir, inter)
np.save(samples_test_dir, samples)
np.save(f_test_dir, f)

# frequency initial reconstruction
data = np.load("./%s/%s/stem_samples_%d.npy" % (data_dir, test_dir, 3))
stack = np.sum(data, axis=1)
f_res = fft_filter_prior_input(data)
f_res = standardize_numpy(f_res)
f_res = (stack + f_res) / 2
np.save("./%s/%s/stem_f_%d_prior_input.npy" % (data_dir, test_dir, 3), f_res)
