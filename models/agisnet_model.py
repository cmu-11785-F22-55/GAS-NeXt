import torch
import torch.nn as nn
import networks
from base_model import BaseModel
from vgg import VGG19
import os


class AGISNetModel(BaseModel):
    def __init__(self, opt):
        super(AGISNetModel, self).__init__()
        if opt.isTrain:
            assert(opt.batch_size % 2 == 0)

        self.loss_names = ['G_L1', 'G_L1_B', 'G_CX', 'G_CX_B', 'G_GAN', 'G_GAN_B', 'D', 'D_B', 'G_L1_val', 'G_L1_B_val', 'local_adv']
        self.loss_G_L1_val = 0.0
        self.loss_G_L1_B_val = 0.0

        self.model_names = ['G', 'D', 'D_B', 'D_local']
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'real_C', 'real_C_l', 'fake_C'] #TODO: necessary?

        
