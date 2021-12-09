import numpy as np
import random
import torch
import cv2
import torchvision
import os
import pytorch_ssim
from torch.autograd import Variable


def add_gaussian_noise(raw, mu, sigma):
    noise = np.random.normal(loc=mu, scale=sigma, size=raw.shape)
    return raw + noise


def add_poisson_noise(raw):
    row, column = raw.shape
    for i in range(row):
        for j in range(column):
            if raw[i, j]:
                raw[i, j] = np.random.poisson(raw[i, j])
    return raw


def add_poisson_noise_intensity(raw, intensity):
    row, column = raw.shape
    for i in range(row):
        for j in range(column):
            if raw[i, j]:
                ratio = np.random.poisson(intensity) / intensity
                raw[i, j] *= ratio
    return raw


def compressing_sampling(raw, input_dim, output_dim):
    raw = raw.reshape(-1)
    zero_ones = np.random.randint(0, 2, (output_dim, input_dim))
    out = np.matmul(zero_ones, raw)
    return out


def sparse_sampling(raw, mask, pos_noise=False):
    if not pos_noise:
        return torch.multiply(raw, mask)


def sparse_sampling_numpy(raw, mask, pos_noise=False):
    if not pos_noise:
        return np.multiply(raw, mask)
    else:
        sample = np.multiply(raw, mask)
        b, _, _ = sample.shape
        res = []
        for i in range(b):
            res.append(add_poisson_noise_intensity(sample[i], 10))
        return np.stack(res, axis=0)


def random_mask_generate(input_dim, ones_num):
    new_arr = np.zeros(input_dim*input_dim)
    new_arr[:ones_num] = 1
    np.random.shuffle(new_arr)
    re_arr = new_arr.reshape(input_dim, input_dim)
    return re_arr


def small_block_mask_generate(block_size, input_size=512, drift_x=0, drift_y=0, fixed=True):
    if fixed:
        num, remain = input_size // block_size, input_size % block_size
        small_mask = np.zeros((block_size, block_size))
        small_mask[drift_x, drift_y] = 1
        new_arr = np.tile(small_mask, (num, num))
        if remain:
            new_arr = np.column_stack((new_arr, new_arr[:, -remain:]))
            new_arr = np.row_stack((new_arr, new_arr[-remain:, :]))
    else:
        new_arr = np.zeros((input_size, input_size))
        for i in range(0, input_size, block_size):
            for j in range(0, input_size, block_size):
                m = random.randint(0, block_size-1)
                n = random.randint(0, block_size-1)
                new_arr[i+m][j+n] = 1
    return new_arr


def generate_gaussian(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    s = sigma**2
    val_sum = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            kernel[i, j] = np.exp(-(x**2+y**2)/(s*2))
            val_sum += kernel[i, j]
    kernel = kernel / val_sum
    return kernel


def image_shrink(inputs, scale):
    b, c, x, y = inputs.shape
    output = torch.zeros((b, c, x // scale, y // scale))
    for i in range(x // scale):
        for j in range(y // scale):
            center = scale // 2
            output[:, :, i, j] = inputs[:, :, i * scale + center, j * scale + center]
    return output


def tensor_cat(inputs, dim=1):
    x, y, z, m, n = inputs.shape
    output = None
    row = None
    for i in range(y):
        if not i % 8:
            if output is None:
                output = row
            else:
                output = torch.cat([output, row], dim=3)
            row = None
        if row is None:
            row = inputs[:, i:i + 1, :, :, :]
        else:
            row = torch.cat([row, inputs[:, i:i + 1, :, :, :]], dim=4)

    output = torch.cat([output, row], dim=3)
    return output


def crop_into_64(inputs):
    b, c, x, y = inputs.shape
    inputs = inputs.view(-1, 1, c, x, y)
    output_list = []
    for i in range(8):
        for j in range(8):
            m, n = i*64, j*64
            output_list.append(inputs[:, :, :, m:m+64, n:n+64])
    output = torch.cat(output_list, dim=1)
    return output


def normalize_numpy(inputs, min_=0, max_=1):
    x_max = np.max(inputs, axis=(1, 2)).reshape(-1, 1, 1)
    x_min = np.min(inputs, axis=(1, 2)).reshape(-1, 1, 1)
    if min_ == 0:
        inputs = (inputs - x_min) / (x_max - x_min)
    if min_ == -1:
        a = 0.5 * (x_max - x_min)
        b = 0.5 * (x_max + x_min)
        inputs = (inputs - b) / a
    return inputs


def standardize_tensor(inputs):
    s, m = torch.std_mean(inputs, dim=(1, 2))
    s = s.view(-1, 1, 1)
    m = m.view(-1, 1, 1)
    output = (inputs - m) / s
    output_data = torch.where(torch.isnan(output), torch.full_like(output, 0), output)
    return output_data


def standardize_numpy(inputs, constraint=False):
    s = np.std(inputs, axis=(1, 2))
    m = np.mean(inputs, axis=(1, 2))
    m = m.reshape(-1, 1, 1)
    s = s.reshape(-1, 1, 1)
    output = (inputs - m) / s
    if constraint:
        output = np.where(output > -4, output, -4)
        output = np.where(output < 4, output, 4)
    return output


def snr(ori, mod):
    ori_square = np.sum(np.square(ori))
    diff_square = np.sum(np.square(ori - mod))
    return 10*np.log10((ori_square / diff_square))


def psnr(ori, mod):
    diff_square = np.mean(np.square(ori - mod))
    return 10*np.log10(1 / diff_square)


def tv_loss(inputs, beta=2):
    b, _, m, n = inputs.shape
    # inputs = torch.squeeze(inputs)
    inputs = inputs.view(b, m, n)
    if beta == 2:
        # b, m, n = inputs.shape
        count_div = (m - 1) * (n - 1)
        m_tv = torch.square((inputs[:, 1:, :] - inputs[:, :m-1, :])).sum()
        n_tv = torch.square(inputs[:, :, 1:] - inputs[:, :, :n-1]).sum()
    return (m_tv + n_tv) / (count_div * b)


def modify_state_dict(pretrained_dict, model_dict):
    for k, v in pretrained_dict.items():
        new_k = k[:6] + '.init_recon' + k[6:]
        model_dict[new_k] = v
    return model_dict


def stem_sparse_sampling(inputs, block_size=4, mode='fixed', pos_noise=False):
    b, h, w = inputs.shape
    sample_list = []
    if mode == 'fixed':
        for i in range(4):
            mask = small_block_mask_generate(block_size=block_size, input_size=h, drift_x=i, drift_y=np.random.randint(block_size))
            sample_list.append(sparse_sampling_numpy(inputs, mask, pos_noise))
    elif mode == 'block_random':
        for i in range(4):
            mask = small_block_mask_generate(block_size=block_size, input_size=h, drift_x=i, drift_y=np.random.randint(block_size), fixed=False)
            sample_list.append(sparse_sampling_numpy(inputs, mask, pos_noise))
    elif mode == 'random':
        for i in range(4):
            shrink = block_size * block_size
            mask = random_mask_generate(h, np.square(h) // shrink)
            sample_list.append(sparse_sampling_numpy(inputs, mask, pos_noise))
    return np.stack(sample_list, axis=1)


def samples_multi(inputs, sample_num, input_size=1024, block_size=8, pos_noise=True):
    sample_list = []
    for i in range(4):
        new_arr = np.zeros((input_size, input_size))
        for i in range(0, input_size, block_size):
            for j in range(0, input_size, block_size):
                new_block = np.zeros(8 * 8)
                new_block[:sample_num] = 1
                np.random.shuffle(new_block)
                re_block = new_block.reshape(8, 8)
                new_arr[i:i+8, j:j+8] = re_block
        mask = new_arr
        sample_list.append(sparse_sampling_numpy(inputs, mask, pos_noise))
    return np.stack(sample_list, axis=1)


def upsample(inputs):
    res = []
    for img in inputs:
        up_img = cv2.resize(img, (1024, 1024), cv2.INTER_CUBIC)
        res.append(up_img)
    return np.stack(res, axis=0)


def fft_filter(inputs, n=200):
    # sum_frame = inputs.sum(axis=0)
    b, s, h, w = inputs.shape
    total_res = []
    for l in range(b):
        f_res = []
        for k in range(s):
            frame = inputs[l, k, :, :]
            f = np.fft.fft2(frame)
            fshift = np.fft.fftshift(f)
            f_res.append(fshift)
        f_sum = np.stack(f_res, axis=0).sum(axis=0)
        show_f = np.log(np.abs(f_sum))
        mask = np.zeros((h, w))
        row, col = h // 2, w // 2
        flat_indices = np.argpartition((show_f * -1).ravel(), n)[:n+1]
        row_indices, col_indices = np.unravel_index(flat_indices, (h, w))
        for i, j in zip(row_indices, col_indices):
            i_dis = i - row
            j_dis = j - col
            if np.square(i_dis) + np.square(j_dis) <= 40000:
                mask[i, j] = 1
        # for i in range(200):
        #     for j in range(200):
        #         mask[i, j] = 1
        filter_res = f_sum * mask
        f_i = np.fft.ifftshift(filter_res)
        new_img = np.fft.ifft2(f_i)
        new_img = np.abs(new_img)
        total_res.append(new_img)
    return np.stack(total_res, axis=0)


def fft_filter_single_stack(inputs, n=20):
    b, s, h, w = inputs.shape
    total_res = []
    for l in range(b):
        f_res = []
        for k in range(s):
            frame = inputs[l, k, :, :]
            f = np.fft.fft2(frame)
            fshift = np.fft.fftshift(f)
            show_f = np.log(np.abs(fshift))
            mask = np.zeros((h, w))
            row, col = h // 2, w // 2
            for i in range(h):
                for j in range(w):
                    i_dis = i - row
                    j_dis = j - col
                    if np.square(i_dis) + np.square(j_dis) <= 40000:
                        mask[i, j] = 1
            show_f = show_f * mask
            flat_indices = np.argpartition((show_f * -1).ravel(), n)[:n + 1]
            row_indices, col_indices = np.unravel_index(flat_indices, (h, w))
            for i, j in zip(row_indices, col_indices):
                mask[i, j] = 1
            filter_res = fshift * mask
            f_res.append(filter_res)
        f_sum = np.stack(f_res, axis=0).sum(axis=0)
        f_i = np.fft.ifftshift(f_sum)
        new_img = np.fft.ifft2(f_i)
        new_img = np.abs(new_img)
        total_res.append(new_img)
    return np.stack(total_res, axis=0)


def fft_filter_multi_scale_stack(inputs, n=50):
    b, s, h, w = inputs.shape
    total_res = []
    for l in range(b):
        multi_res = []
        for m in range(4):
            f_res = []
            for k in range(s):
                frame = inputs[l, k, :, :]
                f = np.fft.fft2(frame)
                fshift = np.fft.fftshift(f)
                f_res.append(fshift)
            f_sum = np.stack(f_res, axis=0).sum(axis=0)
            show_f = np.log(np.abs(f_sum))
            mask = np.zeros((h, w))
            row, col = h // 2, w // 2
            for i in range(h):
                for j in range(w):
                    i_dis = i - row
                    j_dis = j - col
                    # pre = np.square(10*m)
                    pos = np.square(50*m + 100)
                    if np.square(i_dis) + np.square(j_dis) <= pos:
                        mask[i, j] = 1
            show_f = show_f * mask
            n_f = n * (m + 1)
            flat_indices = np.argpartition((show_f * -1).ravel(), n_f)[:n_f + 1]
            row_indices, col_indices = np.unravel_index(flat_indices, (h, w))
            mask = np.zeros((h, w))
            for i, j in zip(row_indices, col_indices):
                mask[i, j] = 1
            filter_res = f_sum * mask
            f_i = np.fft.ifftshift(filter_res)
            new_img = np.fft.ifft2(f_i)
            new_img = np.abs(new_img)
            multi_res.append(new_img)
        multi_sum = np.stack(multi_res, axis=0)
        total_res.append(multi_sum)
    return np.stack(total_res, axis=0)


def fft_filter_prior_input(inputs, n=40):
    # sum_frame = inputs.sum(axis=0)
    b, s, h, w = inputs.shape
    total_res = []
    for l in range(b):
        f_res = []
        for k in range(s):
            frame = inputs[l, k, :, :]
            f = np.fft.fft2(frame)
            fshift = np.fft.fftshift(f)
            f_res.append(fshift)
        f_sum = np.stack(f_res, axis=0).sum(axis=0)
        show_f = np.log(np.abs(f_sum))
        mask = np.zeros((h, w))
        row, col = h // 2, w // 2
        for i in range(h):
            for j in range(w):
                i_dis = i - row
                j_dis = j - col
                if np.square(i_dis) + np.square(j_dis) <= 150**2:
                    mask[i, j] = 1
        # for i in range(200):
        #     for j in range(200):
        #         mask[i, j] = 1
        filter_res = f_sum * mask
        f_i = np.fft.ifftshift(filter_res)
        new_img = np.fft.ifft2(f_i)
        new_img = np.abs(new_img)
        total_res.append(new_img)
    return np.stack(total_res, axis=0)


def compute_weight(dist, h):
    return torch.exp((-dist) / h)


def compute_dist(x, y):
    return torch.mean(torch.square(x - y))


def img2patch(inputs, patch_size=(16, 16), stride=(8, 8)):
    _, _, h, _ = inputs.shape
    # stride = patch_size // 2
    p_h, p_w = patch_size
    s_h, s_w = stride
    h_num = h // s_h - (p_h // s_h) + 1
    w_num = h // s_w - (p_w // s_w) + 1
    patch_dict = {}
    # patch_list = []
    for i in range(h_num):
        for j in range(w_num):
            patch_dict[(i, j)] = inputs[:, :, i*s_h:i*s_h+p_h, j*s_w:j*s_w+p_w]
            # patch_list.append((inputs[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size], (i, j)))
    # return patch_list
    return patch_dict


def patch2img(patch_dict, patch_size=(16, 16), stride=(8, 8)):
    row = None
    one_row = None
    img = None
    one_img = None
    p_h, p_w = patch_size
    s_h, s_w = stride
    o_h = p_h - s_h
    o_w = p_w - s_w
    patch_dict = sorted(patch_dict.items(), key=lambda x: (x[0][0], x[0][1]))
    for index, p in patch_dict:
        # print(index)
        # print(p)
        # print('================================')
        if row is None:
            row = p.clone()
            one_row = torch.ones(p.shape)
            continue
        elif index[1] == 0:
            if img is None:
                img = row
                one_img = one_row
            else:
                img[:, :, -o_h:, :] = img[:, :, -o_h:, :] + row[:, :, :o_h, :]
                img = torch.cat([img, row[:, :, -s_h:, :]], dim=2)
                one_img[:, :, -o_h:, :] = one_img[:, :, -o_h:, :] + one_row[:, :, :o_h, :]
                one_img = torch.cat([one_img, one_row[:, :, -s_h:, :]], dim=2)
            row = p.clone()
            one_row = torch.ones(p.shape)
            continue
        else:
            row[:, :, :, -o_w:] = row[:, :, :, -o_w:] + p[:, :, :, :o_w]
            # print(index)
            # print(p)
            # print('================================')
            row = torch.cat([row, p[:, :, :, -s_w:]], dim=3)
            one_row[:, :, :, -o_w:] = one_row[:, :, :, -o_w:] + torch.ones(p[:, :, :, :o_w].shape)
            one_row = torch.cat([one_row, torch.ones(p[:, :, :, -s_w:].shape)], dim=3)
    img[:, :, -o_h:, :] = img[:, :, -o_h:, :] + row[:, :, :o_h, :]
    img = torch.cat([img, row[:, :, -s_h:, :]], dim=2)
    one_img[:, :, -o_h:, :] = one_img[:, :, -o_h:, :] + one_row[:, :, :o_h, :]
    one_img = torch.cat([one_img, one_row[:, :, -s_h:, :]], dim=2)
    del row
    del one_row
    res = img / one_img
    # return res, one_img
    del one_img
    return res


def Knn_patch(inputs, k=4, patch_size=(16, 16), stride=(8, 8), region_size=(128, 128), fusion=False):
    p_h, p_w = patch_size
    re_h, re_w = region_size
    s_h, s_w = stride
    b, c, h, w = inputs.shape
    # inputs = inputs.view(b, 1, h, w)
    patch_dict = img2patch(inputs, patch_size, stride)

    # interest_num = re_h // p_h
    i_h = re_h // (s_h * 2) - (p_h // 2 // s_h) + 1
    i_w = re_w // (s_w * 2) - (p_h // 2 // s_h) + 1
    patch_num_h = h // s_h - (p_h // s_h)
    patch_num_w = h // s_w - (p_w // s_w)
    for index in patch_dict.keys():
        dist = []
        region_index = [(index[0]+m, index[1]+n) for n in range(i_h) for m in range(i_w)]
        region_index.extend([(index[0]-m, index[1]-n) for n in range(i_h) for m in range(i_w)])
        region_index = filter(lambda x: 0 <= x[0] <= patch_num_h and 0 <= x[1] <= patch_num_w, region_index)
        for t in region_index:
            if patch_dict[t].shape[1] == c:
                dist.append((compute_dist(patch_dict[index], patch_dict[t]), t))
            else:
                dist.append((compute_dist(patch_dict[index], patch_dict[t][:, :c, :, :]), t))
        dist = sorted(dist, key=lambda x: x[0])
        tmp = []
        if fusion:
            total_w = 0
            for i in range(k+1):
                indice = dist[i][1]
                near_patch = patch_dict[indice]
                weight = compute_weight(dist[i][0], 1).cpu()
                # weight = compute_weight(dist[i][0], 1/2).cpu()
                total_w += weight
                if near_patch.shape[1] == 1:
                    tmp.append(near_patch * weight)
                else:
                    tmp.append(near_patch[:, :c, :, :] * weight)

            tmp = [i / total_w for i in tmp]
            patch_dict[index] = torch.cat(tmp, dim=1)
        else:
            for i in range(1, k+1):
                indice = dist[i][1]
                near_patch = patch_dict[indice]
                if near_patch.shape[1] == 1:
                    tmp.append(near_patch)
                else:
                    tmp.append(near_patch[:, :c, :, :])
            patch_dict[index] = torch.cat(tmp, dim=1)
    if fusion:
        for k in patch_dict.keys():
            patch_dict[k] = (torch.sum(patch_dict[k][:, :, :, :], dim=1)).view(1, 1, p_h, p_w)
        return patch2img(patch_dict, patch_size, stride)
    return patch2img(patch_dict, patch_size, stride)


def Knn_patch_feature(inputs, feature, k=4, patch_size=(16, 16), region_size=(128, 128)):
    p_h, p_w = patch_size
    re_h, re_w = region_size
    b, c, h, w = inputs.shape
    patch_dict = img2patch(inputs, p_h)
    feature_dict = img2patch(feature, p_h)
    interest_num = re_h // p_h
    patch_num = h * 2 // p_h - 2
    for index in patch_dict.keys():
        dist = []
        region_index = [(index[0]+m, index[1]+n) for n in range(interest_num) for m in range(interest_num)]
        region_index.extend([(index[0]-m, index[1]-n) for n in range(interest_num) for m in range(interest_num)])
        region_index = filter(lambda x: 0 <= x[0] <= patch_num and 0 <= x[1] <= patch_num, region_index)
        for t in region_index:
            if patch_dict[t].shape[1] == c:
                dist.append((compute_dist(feature_dict[index], feature_dict[t]), t))
            else:
                dist.append((compute_dist(feature_dict[index], feature_dict[t][:, :c, :, :]), t))
        dist = sorted(dist, key=lambda x: x[0])
        tmp = []
        total_weight = 0
        for i in range(k+1):
            indice = dist[i][1]
            near_patch = patch_dict[indice]
            weight = compute_weight(dist[i][0], 1).cpu()
            total_weight += weight
            if near_patch.shape[1] == 1:
                tmp.append(near_patch * weight)
            else:
                tmp.append(near_patch[:, :c, :, :] * weight)
        tmp = [i / total_weight for i in tmp]
        patch_dict[index] = torch.cat(tmp, dim=1)
    return patch2img(patch_dict)


def cal_psnr_ssim():
    generate_dir = './test/'
    raw = np.load('./stem_test_gt_nostd.npy')
    psnr_res = dict()
    ssim_res = dict()
    length = len(raw)
    for file in os.listdir(generate_dir):
        if 'Z' in file:
            data = np.load(generate_dir + file)
            psnr_tmp = 0
            for i in range(length):
                raw_img = raw[i]
                raw_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
                img = data[i]
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                psnr_tmp += psnr(raw_img, img)
            psnr_res[file[:-4]] = psnr_tmp / length
            psnr_tmp = 0

            recons_data = normalize_numpy(data).reshape(-1, 1, 512, 512)
            raw_data = normalize_numpy(raw).reshape(-1, 1, 512, 512)
            img1 = Variable(torch.from_numpy(raw_data))
            img2 = Variable(torch.from_numpy(recons_data))

            if torch.cuda.is_available():
                img1 = img1.cuda()
                img2 = img2.cuda()

            ssim_loss = pytorch_ssim.SSIM(window_size=11)
            ssim_res[file[:-4]] = ssim_loss(img1, img2).item()

    print(psnr_res)
    print(ssim_res)

    with open('./test/Metric.txt', 'a') as f:
        for k in psnr_res.keys():
            f.write(k + '\n')
            f.write('psnr: %.4f \t' % psnr_res[k])
            f.write('ssim: %.4f \n' % ssim_res[k])
    return


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def normalize_img(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img
