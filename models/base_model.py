import os
import torch
import torch.nn as nn
from collections import OrderedDict
import models.networks as networks


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device(
            'cuda') if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []
        # self.isTrain = opt.isTrain
        # if opt.resize_or_crop != 'scale_width':
        #     torch.backends.cudnn.benchmark = True

    def setup(self, opt):
        # TODO may not need it no optimizer
        if opt.isTrain:
            self.schedulers = [networks.getScheduler(
                optimizer, opt) for optimizer in self.optimizers]
    
        if not opt.isTrain or opt.continue_train:
            self.loadNetworks(opt.epoch)
        self.print_networks(opt.verbose)

    def setInput(self, input):
        self.input = input

    def forward(self):
        pass

    def isTrain(self):
        # TODO don't know if need this
        return True

    def setRequiresGrad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad  # to avoid computation

    def test(self):
        """
        Used in test time, wrapping `forward` in no_grad() so we don't save
        Intermediate steps for backprop
        """
        # TODO don't know if need this wrapper
        with torch.no_grad():
            self.forward()

    def getImagePaths(self):
        """
        Get image paths
        """
        return self.image_paths

    def optimizeParameters(self):
        # TODO  don't know if need this
        pass

    def updateLearningRate(self):
        # TODO may not need it no optimizer
        """
        Update learning rate (called once every epoch)
        """
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print(f'learning rate = {lr:.7f}')

    def getCurrentVisuals(self):
        """
        Return visualization images.
        train.py will display these images, and save the images to a html
        """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def getCurrentLosses(self):
        """
        Return traning losses/errors.
        train.py will print out these errors as debugging information

        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def eval(self):
        """
        Make models eval mode during test time
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def saveNetworks(self, epoch):
        """
        Save models to the disk
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda()
                    net = torch.nn.DataParallel(net)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def loadNetworks(self, epoch):
        """
        Load models from the disk
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print(f'loading the model from {load_path}')
                state_dict = torch.load(
                    load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # need to copy keys here because we mutate in loop
                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(
                        state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """
        Print network information
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(
                    f'[Network %{name}] Total number \
                of parameters : {num_params/1e6:.3f} M')
        print('-----------------------------------------------')
