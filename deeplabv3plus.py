
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import queue
import collections
import threading
from torch.nn.modules.batchnorm import _BatchNorm
from model_utils import unetConv2, DSConv, GhostConv, SELayer, MSCA, DSDilatedConv

class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)

class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5

class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = DSConv(inplanes, planes, kernel_size=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = DSConv(planes, planes, kernel_size=3, stride=stride,
                               padding=1*atrous, dilation=atrous, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = DSConv(planes, planes * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3 = SynchronizedBatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

bn_mom = 0.0003
class ResNet_Atrous(nn.Module):

    def __init__(self, block, layers, atrous=None, os=16):
        super(ResNet_Atrous, self).__init__()
        stride_list = None
        if os == 8:
            stride_list = [2,1,1]
        elif os == 16:
            stride_list = [2,2,1]
        else:
            raise ValueError('resnet_atrous.py: output stride=%d is not supported.'%os) 
            
        self.inplanes = 64
        self.conv1 = DSConv(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = SynchronizedBatchNorm2d(64, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=stride_list[0])
        self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=stride_list[1], atrous=16//os)
        self.layer4 = self._make_layer(block, 512, 512, layers[3], stride=stride_list[2], atrous=[item*16//os for item in atrous])
        self.layers = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self):
        return self.layers

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1]*blocks
        elif isinstance(atrous, int):
            atrous_list = [atrous]*blocks
            atrous = atrous_list
        if stride != 1 or inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                DSConv(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, atrous=atrous[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes*block.expansion, planes, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.layers.append(x)
        x = self.layer2(x)
        self.layers.append(x)
        x = self.layer3(x)
        self.layers.append(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        #x = self.layer6(x)
        #x = self.layer7(x)
        self.layers.append(x)

        return x
    
class ASPP(nn.Module):
	
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				DSConv(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				DSConv(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate,bias=True),
				SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				DSConv(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate,bias=True),
				SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				DSConv(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate,bias=True),
				SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = DSConv(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = SynchronizedBatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)
		self.conv_cat = nn.Sequential(
				DSConv(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)
#		self.conv_cat = nn.Sequential(
#				DSConv(dim_out*4, dim_out, 1, 1, padding=0),
#				SynchronizedBatchNorm2d(dim_out),
#				nn.ReLU(inplace=True),		
#		)
	def forward(self, x):
		[b,c,row,col] = x.size()
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row,col), None, 'bilinear', True)
		
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
		result = self.conv_cat(feature_cat)
		return result

def resnet152_atrous(pretrained=False, os=16, **kwargs):
    """Constructs a atrous ResNet-152 model."""
    model = ResNet_Atrous(Bottleneck, [3, 8, 36, 3], atrous=[1,2,1], os=os, **kwargs)
    return model

def build_backbone(backbone_name, pretrained=True, os=16):
	if backbone_name == 'res152_atrous':
		net = resnet152_atrous(pretrained=False, os=os)
		return net
	else:
		raise ValueError('backbone.py: The backbone named %s is not supported yet.'%backbone_name)

class deeplabv3plus(nn.Module):
    def __init__(self):
        super(deeplabv3plus, self).__init__()
        self.backbone = None		
        self.backbone_layers = None
        input_channel = 512
        aspp_outdim = 128
        MODEL_OUTPUT_STRIDE = 16
        TRAIN_BN_MOM = 0.0003
        MODEL_SHORTCUT_DIM = 16
        MODEL_SHORTCUT_KERNEL = 1
  
        self.aspp = ASPP(dim_in=input_channel, dim_out=aspp_outdim, rate=16//MODEL_OUTPUT_STRIDE, bn_mom = TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.3)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=MODEL_OUTPUT_STRIDE//4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
				DSConv(indim, MODEL_SHORTCUT_DIM, MODEL_SHORTCUT_KERNEL, 1, padding=MODEL_SHORTCUT_KERNEL//2,bias=True),
				SynchronizedBatchNorm2d(MODEL_SHORTCUT_DIM, momentum=TRAIN_BN_MOM),
				nn.ReLU(inplace=True),		
		)		
        self.cat_conv = nn.Sequential(
				DSConv(aspp_outdim+MODEL_SHORTCUT_DIM, aspp_outdim, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(aspp_outdim, momentum=TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.3),
				DSConv(aspp_outdim, aspp_outdim, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(aspp_outdim, momentum=TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
        self.cls_conv = DSConv(aspp_outdim, 19, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone('res152_atrous', os=MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total parameters counts deep: {:,} \n".format(total_params))

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp,feature_shallow],1)
        result = self.cat_conv(feature_cat) 
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result
