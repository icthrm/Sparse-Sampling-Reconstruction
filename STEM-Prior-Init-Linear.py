import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from Util import tv_loss, stem_sparse_sampling, normalize_numpy, fft_filter, standardize_numpy, generate_gaussian, get_parameter_number
from argparse import ArgumentParser
import cv2
from collections import defaultdict
import time
from torch.autograd import Variable

parser = ArgumentParser(description="Ours")

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--feature_mode', type=str, default='loG', help='feature mode')
parser.add_argument('--block_size', type=int, default=8, help='block size')

parser.add_argument('--archi_flag', action='store_true', default=False, help='model architecture')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--train_dir', type=str, default='stem_train', help='training data name')
parser.add_argument('--test_dir', type=str, default='test', help='testing data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--test', action='store_true', default=False, help='train or test')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
feature_mode = args.feature_mode
block_size = args.block_size

model_flag = args.archi_flag
gpu_list = args.gpu_list
model_dir = args.model_dir
data_dir = args.data_dir
log_dir = args.log_dir
test_dir = args.test_dir
train_dir = args.train_dir
test_flag = args.test

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0")

n_input_dim = 1024
n_input = n_input_dim * n_input_dim
n_output_dim = 512
n_output = n_output_dim * n_output_dim
n_train_num = 637
batch_size = 4
epoch_num = end_epoch
layer_num = 5

if not test_flag:
    Train_data_name = './%s/%s.npy' % (data_dir, train_dir)
    Train_raw = np.load(Train_data_name)
    Train_raw = standardize_numpy(Train_raw)


class Reshape_Concat_Adap(torch.autograd.Function):
    blocksize = 0

    def __init__(self, block_size):
        # super(Reshape_Concat_Adap, self).__init__()
        Reshape_Concat_Adap.blocksize = block_size

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)

        data = torch.clone(input_.data)
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        output = torch.zeros((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                              int(w_ * Reshape_Concat_Adap.blocksize), int(h_ * Reshape_Concat_Adap.blocksize))).cuda()

        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = data[:, :, i, j]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                # data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                                            Reshape_Concat_Adap.blocksize, Reshape_Concat_Adap.blocksize))
                # print data_temp.shape
                output[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize] += data_temp

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        input_ = torch.clone(inp.data)
        grad_input = torch.clone(grad_output.data)

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        output = torch.zeros((b_, c_, w_, h_)).cuda()
        output = output.view(b_, c_, w_, h_)
        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = grad_input[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                            j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, i, j] += torch.squeeze(data_temp)

        return Variable(output)


def My_Reshape_Adap(input, blocksize):
    return Reshape_Concat_Adap(blocksize).apply(input)


class ResidualModule(torch.nn.Module):
    def __init__(self):
        super(ResidualModule, self).__init__()
        self.conv1 = nn.Parameter(init.kaiming_normal_(torch.Tensor(32, 64, 1, 1)))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Parameter(init.kaiming_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Parameter(init.kaiming_normal_(torch.Tensor(64, 32, 1, 1)))
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, inputs):
        x = F.conv2d(inputs, self.conv1)
        x = F.leaky_relu(x, 0.2)
        x = self.bn1(x)
        x = F.conv2d(x, self.conv2, padding=1)
        x = F.leaky_relu(x, 0.2)
        x = self.bn2(x)
        x = F.conv2d(x, self.conv3)
        x = F.leaky_relu(x, 0.2)
        x = self.bn3(x)
        return x + inputs


class EncoderDecoder(torch.nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.x_vst = nn.Parameter(init.kaiming_normal_(torch.Tensor(4, 4, 1, 1)))
        self.vst_x_bias = nn.Parameter(torch.Tensor(4).fill_(0.))
        self.bn_x_vst = nn.BatchNorm2d(4)
        self.shallow_x_1 = nn.Parameter(init.kaiming_normal_(torch.Tensor(8, 4, 7, 7)))
        self.bn_x_s_1 = nn.BatchNorm2d(8)
        self.shallow_x_2 = nn.Parameter(init.kaiming_normal_(torch.Tensor(8, 8, 7, 7)))
        self.bn_x_s_2 = nn.BatchNorm2d(8)
        self.shallow_x_3 = nn.Parameter(init.kaiming_normal_(torch.Tensor(16, 8, 5, 5)))
        self.bn_x_s_3 = nn.BatchNorm2d(16)
        self.shallow_x_4 = nn.Parameter(init.kaiming_normal_(torch.Tensor(16, 16, 3, 3)))
        self.bn_x_s_4 = nn.BatchNorm2d(16)

        self.shallow_1 = nn.Parameter(init.kaiming_normal_(torch.Tensor(8, 1, 7, 7)))
        self.bn_s_1 = nn.BatchNorm2d(8)
        self.shallow_2 = nn.Parameter(init.kaiming_normal_(torch.Tensor(8, 8, 7, 7)))
        self.bn_s_2 = nn.BatchNorm2d(8)
        self.shallow_3 = nn.Parameter(init.kaiming_normal_(torch.Tensor(16, 8, 5, 5)))
        self.bn_s_3 = nn.BatchNorm2d(16)
        self.shallow_4 = nn.Parameter(init.kaiming_normal_(torch.Tensor(16, 16, 5, 5)))
        self.bn_s_4 = nn.BatchNorm2d(16)
        self.shallow_5 = nn.Parameter(init.kaiming_normal_(torch.Tensor(32, 16, 3, 3)))
        self.bn_s_5 = nn.BatchNorm2d(32)
        self.shallow_6 = nn.Parameter(init.kaiming_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bn_s_6 = nn.BatchNorm2d(32)

        self.downpool = nn.MaxPool2d(kernel_size=block_size, stride=block_size)
        self.deep1_1 = nn.Parameter(init.kaiming_normal_(torch.Tensor(8, 4, 3, 3)))
        self.bn_d_1 = nn.BatchNorm2d(8)
        self.deep1_2 = nn.Parameter(init.kaiming_normal_(torch.Tensor(16, 8, 3, 3)))
        self.bn_d_2 = nn.BatchNorm2d(16)
        self.deep1_3 = nn.Parameter(init.kaiming_normal_(torch.Tensor(32, 16, 3, 3)))
        self.bn_d_3 = nn.BatchNorm2d(32)

        res_layer = []
        self.LayerNo = layer_num
        for i in range(self.LayerNo):
            res_layer.append(ResidualModule())
        self.repeat_residual = nn.ModuleList(res_layer)

        self.up_1 = nn.Parameter(init.kaiming_normal_(torch.Tensor(32, 64, 3, 3)))
        self.bn_up_1 = nn.BatchNorm2d(32)
        self.up_2 = nn.Parameter(init.kaiming_normal_(torch.Tensor(32, 64, 3, 3)))
        self.bn_up_2 = nn.BatchNorm2d(32)
        self.up_3 = nn.Parameter(init.kaiming_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bn_up_3 = nn.BatchNorm2d(32)
        self.up_4 = nn.Parameter(init.kaiming_normal_(torch.Tensor(32, 32, 1, 1)))
        self.up_4_bias = nn.Parameter(torch.Tensor(32).fill_(0.))
        self.bn_up_4 = nn.BatchNorm2d(32)
        self.upsample = nn.PixelShuffle(2)

        self.fusion_1 = nn.Parameter(init.kaiming_normal_(torch.Tensor(8, 24, 3, 3)))
        self.bn_f_1 = nn.BatchNorm2d(8)
        self.fusion_2 = nn.Parameter(init.kaiming_normal_(torch.Tensor(1, 8, 3, 3)))
        self.bn_f_2 = nn.BatchNorm2d(1)

        self.fusion_2_1 = nn.Parameter(init.kaiming_normal_(torch.Tensor(8, 8, 3, 3)))
        self.fusion_2_2 = nn.Parameter(init.kaiming_normal_(torch.Tensor(1, 8, 3, 3)))
        self.fusion_2_3 = nn.Parameter(init.kaiming_normal_(torch.Tensor(1, 1, 1, 1)))

    def forward(self, inputs, prior_input):  # (b, 4, 1024, 1024)

        inputs_shallow = F.conv2d(inputs, self.x_vst, bias=self.vst_x_bias, padding=0)
        inputs_shallow = F.leaky_relu(inputs_shallow, 0.2)
        inputs_shallow = self.bn_x_vst(inputs_shallow)

        inputs_shallow = F.conv2d(inputs_shallow, self.shallow_x_1, padding=3)
        inputs_shallow = F.leaky_relu(inputs_shallow, 0.2)
        inputs_shallow = self.bn_x_s_1(inputs_shallow)
        inputs_shallow = F.conv2d(inputs_shallow, self.shallow_x_2, padding=3)
        inputs_shallow = F.leaky_relu(inputs_shallow, 0.2)  # (b, 8, 512, 512)
        inputs_shallow = self.bn_x_s_2(inputs_shallow)

        inputs_shallow = F.conv2d(inputs_shallow, self.shallow_x_3, padding=2)
        inputs_shallow = F.leaky_relu(inputs_shallow, 0.2)
        inputs_shallow = self.bn_x_s_3(inputs_shallow)
        inputs_shallow = F.conv2d(inputs_shallow, self.shallow_x_4, padding=1)
        inputs_shallow = F.leaky_relu(inputs_shallow, 0.2)  # (b, 16, 256, 256)
        inputs_shallow = self.bn_x_s_4(inputs_shallow)

        x_shallow = F.conv2d(prior_input, self.shallow_1, padding=3)
        x_shallow = F.leaky_relu(x_shallow, 0.2)
        x_shallow = self.bn_s_1(x_shallow)
        x_shallow_1 = F.conv2d(x_shallow, self.shallow_2, stride=2, padding=3)
        x_shallow_1 = F.leaky_relu(x_shallow_1, 0.2)  # (b, 8, 512, 512)
        x_shallow_1 = self.bn_s_2(x_shallow_1)

        x_shallow_2 = F.conv2d(x_shallow_1, self.shallow_3, padding=2)
        x_shallow_2 = F.leaky_relu(x_shallow_2, 0.2)
        x_shallow_2 = self.bn_s_3(x_shallow_2)
        x_shallow_2 = F.conv2d(x_shallow_2, self.shallow_4, stride=2, padding=2)
        x_shallow_2 = F.leaky_relu(x_shallow_2, 0.2)  # (b, 16, 256, 256)
        x_shallow_2 = self.bn_s_4(x_shallow_2)

        x_shallow_3 = F.conv2d(x_shallow_2, self.shallow_5, stride=2, padding=1)
        x_shallow_3 = F.leaky_relu(x_shallow_3, 0.2)
        x_shallow_3 = self.bn_s_5(x_shallow_3)

        x_shallow_4 = F.conv2d(x_shallow_3, self.shallow_6, stride=2, padding=1)
        x_shallow_4 = F.leaky_relu(x_shallow_4, 0.2)  # (b, 32, 64, 64)
        x_shallow_4 = self.bn_s_6(x_shallow_4)

        x_deep = self.downpool(inputs)  # (b, 4, 128, 128)
        x_deep = F.conv2d(x_deep, self.deep1_1, padding=1)
        x_deep = F.leaky_relu(x_deep, 0.2)
        x_deep = self.bn_d_1(x_deep)
        x_deep = F.conv2d(x_deep, self.deep1_2, padding=1)
        x_deep = F.leaky_relu(x_deep, 0.2)
        x_deep = self.bn_d_2(x_deep)

        x_deep = F.conv2d(x_deep, self.deep1_3, stride=2, padding=1)  # (b, 64, 64, 64)
        x_deep = F.leaky_relu(x_deep, 0.2)
        x_deep = self.bn_d_3(x_deep)

        x_deep = torch.cat([x_deep, x_shallow_4], dim=1)

        for i in range(self.LayerNo):
            x_deep = self.repeat_residual[i](x_deep)  # (b, 64, 64, 64)

        x_up = F.interpolate(x_deep, [128, 128], mode='nearest')
        x_up = F.conv2d(x_up, self.up_1, padding=1)
        x_up = F.leaky_relu(x_up, 0.2)
        x_up = self.bn_up_1(x_up)

        x_up = torch.cat([x_up, x_shallow_3], dim=1)
        x_up = F.interpolate(x_up, [256, 256], mode='nearest')
        x_up = F.conv2d(x_up, self.up_2, padding=1)
        x_up = F.leaky_relu(x_up, 0.2)
        x_up = self.bn_up_2(x_up)
        x_up = F.conv2d(x_up, self.up_3, padding=1)
        x_up = F.leaky_relu(x_up, 0.2)
        x_up = self.bn_up_3(x_up)

        # x_up = torch.cat([x_up, x_shallow_2], dim=1)
        x_up = F.interpolate(x_up, [512, 512], mode='nearest')
        x_up = F.conv2d(x_up, self.up_4, bias=self.up_4_bias)  # (b, 32, 512, 512)
        x_up = F.leaky_relu(x_up, 0.2)
        x_up = self.bn_up_4(x_up)

        x_final = self.upsample(x_up)  # (b, 8, 1024, 1024)
        if model_flag:

            x_final = torch.cat([inputs_shallow, x_final], dim=1)
            x_final = F.conv2d(x_final, self.fusion_1, padding=1)
            x_final = F.leaky_relu(x_final, 0.2)
            x_final = self.bn_f_1(x_final)
            x_final = F.conv2d(x_final, self.fusion_2, padding=1)
            x_final = F.leaky_relu(x_final, 0.2)
            x_final = self.bn_f_2(x_final)

            x_output = F.interpolate(x_final, [512, 512], mode='nearest')
        else:
            x_final = F.conv2d(x_final, self.fusion_1, stride=2, padding=1)
            x_final = F.leaky_relu(x_final, 0.2)
            x_final = F.conv2d(x_final, self.fusion_2_1, padding=1)
            x_final = F.leaky_relu(x_final, 0.2)
            x_final = F.conv2d(x_final, self.fusion_2_2, padding=1)
            x_final = F.leaky_relu(x_final, 0.2)
            x_final = F.conv2d(x_final, self.fusion_2_3, padding=0)
            x_output = x_final
        return x_final, x_output


class Extract(torch.nn.Module):
    def __init__(self, mode):
        super(Extract, self).__init__()
        self.sobel_1 = nn.Parameter(data=torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).repeat(4, 1, 1, 1), requires_grad=False)
        self.sobel_2 = nn.Parameter(data=torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).repeat(4, 1, 1, 1), requires_grad=False)
        self.loG = nn.Parameter(data=torch.Tensor([[[[0, 0, -1, 0, 0],
                                                     [0, -1, -2, -1, 0],
                                                     [-1, -2, 16, -2, -1],
                                                     [0, -1, -2, -1, 0],
                                                     [0, 0, -1, 0, 0]]]]), requires_grad=False)
        self.loG4 = nn.Parameter(data=torch.Tensor([[[[0, 0, -1, 0, 0],
                                                     [0, -1, -2, -1, 0],
                                                     [-1, -2, 16, -2, -1],
                                                     [0, -1, -2, -1, 0],
                                                     [0, 0, -1, 0, 0]]]]).repeat(4, 1, 1, 1), requires_grad=False)
        self.feature_mode = mode

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        if self.feature_mode == 'sobel':
            x_direct = F.conv2d(inputs, self.sobel_1, padding=1, groups=4)
            y_direct = F.conv2d(inputs, self.sobel_2, padding=1, groups=4)
            # feature = (torch.abs(x_direct + y_direct) + torch.abs(x_direct - y_direct)) / 2
            feature = torch.abs(x_direct) + torch.abs(y_direct)
        elif self.feature_mode == 'loG':
            if c == 1:
                feature = F.conv2d(inputs, self.loG, padding=2)
                feature = torch.where(feature > 0, torch.full_like(feature, 1), torch.full_like(feature, 0))
            else:
                feature = F.conv2d(inputs, self.loG4, padding=2, groups=4)
                feature = torch.where(feature > 0, torch.full_like(feature, 1), torch.full_like(feature, 0))
        return feature


class Ours(torch.nn.Module):
    def __init__(self, blocksize=64, subrate=0.05):
        super(Ours, self).__init__()
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(4, int(np.round(blocksize * blocksize * subrate)), blocksize, stride=blocksize,
                                  padding=0, bias=False)
        # upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), blocksize * blocksize, 1, stride=1,
                                    padding=0)
        self.recon = EncoderDecoder()
        self.extract1 = Extract('sobel')
        self.extract = Extract(feature_mode)

    def forward(self, inputs, prior):
        inputs = inputs.view(-1, 4, 1024, 1024)
        prior_input = self.sampling(inputs)
        prior_input = self.upsampling(prior_input)
        prior_input = My_Reshape_Adap(prior_input, self.blocksize)
        prior = prior.view(-1, 1, 1024, 1024)
        x_final, x_output = self.recon(inputs, prior_input)
        if model_flag:
            x_final_fea = self.extract(x_final)
            prior_fea = self.extract(prior)
            return x_final_fea, prior_fea, x_output


class RandomDataset(Dataset):
    def __init__(self, raw, data, f, length):
        self.raw = raw
        self.data = data
        self.f = f
        self.len = length

    def __getitem__(self, index):
        raw_data = self.raw[index, :, :]
        sample_frames = self.data[index, :, :, :]
        f = self.f[index, :, :]
        return torch.Tensor(raw_data).float(), torch.Tensor(sample_frames).float(), torch.Tensor(f).float()

    def __len__(self):
        return self.len


model = Ours()
model = nn.DataParallel(model)
model = model.to(device)

if not test_flag:

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    adjusted_lr_schedule = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    model_dir = "./%s/STEM_Z_Prior_Init_Linear_Block_%d_lr_%.4f_interpolate_%s_%s" % (model_dir, block_size, learning_rate, str(model_flag), feature_mode)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = "./%s/STEM_Z_Prior_Init_Linear_Block_%d_lr_%.4f_interpolate_%s_%s" % (log_dir, block_size, learning_rate, str(model_flag), feature_mode)

    if start_epoch > 0:
        pre_model_dir = model_dir
        model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

    ground_truth_dir = './%s/%s.npy' % (data_dir, 'stem_train_median_filter_std')

    sample_dir_3 = './%s/%s.npy' % (data_dir, 'stem_samples_3')
    f_dir_3 = './%s/%s.npy' % (data_dir, 'stem_f_3')

    f_input_dir = './%s/%s.npy' % (data_dir, 'stem_samples_3_prior_input')

    Ground_truth = np.load(ground_truth_dir)
    Train_Samples = np.load(sample_dir_3)
    Train_F = np.load(f_dir_3)

    rand_loader = DataLoader(dataset=RandomDataset(Ground_truth, Train_Samples, Train_F, n_train_num), batch_size=batch_size, num_workers=4,
                             shuffle=True)

    for epoch_i in range(start_epoch, end_epoch+1):
        avg_loss = defaultdict(int)
        model.train()
        for data_raw, data_samples, data_f, data_f_in in rand_loader:

            raw = data_raw.to(device)
            samples = data_samples.to(device)
            prior = data_f.to(device)

            x_final_fea, x_prior_fea, x_output = model(samples, prior)
            loss_discrepancy = torch.mean(torch.square(x_output - raw))
            gamma_1 = torch.Tensor([0.01]).to(device)
            loss_reg = tv_loss(x_output)
            gamma_2 = torch.Tensor([0.2]).to(device)
            loss_prior = torch.mean(torch.square(x_final_fea - x_prior_fea))
            loss_all = loss_discrepancy + torch.mul(gamma_1, loss_reg) + torch.mul(gamma_2, loss_prior)
            # loss_all = loss_discrepancy + torch.mul(gamma_2, loss_prior)

            optimizer.zero_grad()
            loss_all.backward()

            optimizer.step()

            output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f,  Reg Loss: %.4f,  Prior Loss: %.4f\n" % \
                          (epoch_i, epoch_num, loss_all.item(), loss_discrepancy.item(), loss_reg.item(), loss_prior.item())

            print(output_data)

            avg_loss['loss_all'] += loss_all.item()
            avg_loss['loss_discrepancy'] += loss_discrepancy.item()
            avg_loss['loss_reg'] += loss_reg.item()
            avg_loss['loss_prior'] += loss_prior.item()

        for k, v in avg_loss.items():
            avg_loss[k] = (v * batch_size) / n_train_num

        log_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f,  Reg Loss: %.4f,  Prior Loss: %.4f\n" % \
                      (epoch_i, epoch_num, avg_loss['loss_all'], avg_loss['loss_discrepancy'], avg_loss['loss_reg'], avg_loss['loss_prior'])
        log_file = open(log_dir, 'a')
        log_file.write(log_data)
        log_file.close()

        for k, v in avg_loss.items():
            avg_loss[k] = 0

        if epoch_i % 5 == 0:
            torch.save(model.state_dict(), "%s/net_params_%d.pkl" % (model_dir, epoch_i))

        adjusted_lr_schedule.step()

else:

    feature_mode = 'loG'
    model_flag = True
    store_model_dir = "./%s/STEM_Z_Prior_Init_Linear_Block_%d_lr_%.4f_interpolate_%s_%s" % (model_dir, block_size, learning_rate, str(model_flag), feature_mode)
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (store_model_dir, 200)))

    sample_num = 3
    samples_test_dir = "./%s/%s/stem_samples_%d.npy" % (data_dir, test_dir, sample_num)
    f_test_dir = "./%s/%s/stem_f_%d.npy" % (data_dir, test_dir, sample_num)

    for l in range(1, 7):
        samples_test_dir = "./%s/%s/stem_samples_%d.npy" % (data_dir, test_dir, l)
        f_test_dir = "./%s/%s/stem_f_%d.npy" % (data_dir, test_dir, l)

        print(str(l) + 'reconstrcut')

        with torch.no_grad():
            model.eval()
            res = []

            samples = np.load(samples_test_dir)
            f = np.load(f_test_dir)

            for i in range(samples.shape[0]):

                img = samples[i]
                s = np.std(img)
                m = np.mean(img)

                x_samples = torch.Tensor(samples[i]).float()
                x_prior = torch.Tensor(f[i]).float()

                x_samples = x_samples.to(device)
                x_prior = x_prior.to(device)

                x_output_fea, prior_fea, x_output = model(x_samples, x_prior)
                x_res = x_output.cpu().data.numpy()
                x_res = np.squeeze(x_res)
                x_res = x_res * s + m
                res.append(x_res)
            save_dir = "./%s/%s/STEM-Prior-Init-Z-linear-%d.npy" % (data_dir, test_dir, sample_num)
            np.save(save_dir, np.stack(res, axis=0))
