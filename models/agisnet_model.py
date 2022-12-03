import os
import random
from collections import OrderedDict

import torch
import torch.nn as nn

from .networks import CXLoss, GANLoss, define_D, define_G, get_scheduler
from .vgg import VGG19


class AGISNetModel(nn.Module):
    def __init__(self, opt):
        super(AGISNetModel, self).__init__(opt)
        if opt.isTrain:
            assert opt.batch_size % 2 == 0, f"batch size {opt.batch_size} is not even."

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device("cuda") if self.gpu_ids else torch.device("cpu")
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

        self.loss_names = [
            "G_L1",
            "G_L1_B",
            "G_CX",
            "G_CX_B",
            "G_GAN",
            "G_GAN_B",
            "D",
            "D_B",
            "G_L1_val",
            "G_L1_B_val",
            "local_adv",
        ]
        self.loss_G_L1_val = 0.0
        self.loss_G_L1_B_val = 0.0

        self.visual_names = [
            "real_A",
            "real_B",
            "fake_B",
            "real_C",
            "real_C_l",
            "fake_C",
        ]  # TODO: necessary?

        # TODO: use_D etc. necessary?
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D_B = opt.isTrain and opt.lambda_GAN_B > 0.0
        use_local_D = opt.isTrain and opt.lambda_local_D > 0.0

        self.model_names = ["G"]  # TODO: necessary?
        self.netG = define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            self.opt.nencode,
            norm=opt.norm,
            nl=opt.nl,
            use_dropout=opt.use_dropout,
            init_type=opt.init_type,
            gpu_ids=self.gpu_ids,
            upsample=opt.upsample,
        )
        self.models = [self.netG]

        D_output_nc = (
            (opt.input_nc + opt.output_nc) if opt.conditional_D else opt.output_nc
        )
        use_sigmoid = opt.gan_mode == "dcgan"
        if use_D:
            self.model_names += ["D"]
            self.netD = define_D(
                D_output_nc,
                opt.ndf,
                netD=opt.netD,
                norm=opt.norm,
                nl=opt.nl,
                use_sigmoid=use_sigmoid,
                init_type=opt.init_type,
                gpu_ids=self.gpu_ids,
            )
            self.models.append(self.netD)

        if use_D_B:
            self.model_names += ["D_B"]
            self.netD_B = define_D(
                D_output_nc,
                opt.ndf,
                netD=opt.netD,
                norm=opt.norm,
                nl=opt.nl,
                use_sigmoid=use_sigmoid,
                init_type=opt.init_type,
                gpu_ids=self.gpu_ids,
            )
            self.models.append(self.netD_B)

        if use_local_D:
            self.model_names += ["D_local"]
            self.netD_local = define_D(
                D_output_nc,
                opt.ndf,
                netD=opt.netD,
                norm=opt.norm,
                nl=opt.nl,
                use_sigmoid=use_sigmoid,
                init_type=opt.init_type,
                gpu_ids=self.gpu_ids,
            )
            self.models.append(self.netD_local)

        if opt.isTrain:
            self.criterionGAN = GANLoss(mse_loss=not use_sigmoid).to(self.device)
            self.criterionL1 = nn.L1Loss(reduction="none")
            self.criterionL1_reduce = nn.L1Loss()

            self.criterionCX = CXLoss(sigma=0.5).to(self.device)
            self.vgg19 = VGG19().to(self.device)
            self.vgg19.load_model(self.opt.vgg)
            self.vgg19.eval()
            self.vgg_layers = ["conv3_3", "conv4_2"]

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers = [self.optimizer_G]

            if use_D:
                self.optimizer_D = torch.optim.Adam(
                    self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizers.append(self.optimizer_D)

            if use_D_B:
                self.optimizer_D_B = torch.optim.Adam(
                    self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizers.append(self.optimizer_D_B)

            if use_local_D:
                self.optimizer_Dlocal = torch.optim.Adam(
                    self.netD_local.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizers.append(self.optimizer_Dlocal)

    def set_input(self, input):
        self.real_A = input["A"].to(self.device)  # A is the base font
        self.real_B = input["B"].to(self.device)  # B is the gray shape
        # B_G is for GAN, label == 1: B_G == B, else B_G == Shapes[rand]
        self.real_B_G = input["B_G"].to(self.device)
        self.real_C = input["C"].to(self.device)  # C is the color font
        # C_G is for GAN, label == 1: C_G == C, else C_G == Colors[rand]
        self.real_C_G = input["B_G"].to(self.device)
        # C_l is the C * label, useful to visual
        self.real_C_l = input["C_l"].to(self.device)
        # label == 1 means the image is in the few set
        self.label = input["label"].to(self.device)

        self.real_Bases = input["Bases"].to(self.device)
        self.real_Shapes = input["Shapes"].to(self.device)
        # Colors is reference color characters
        self.real_Colors = input["Colors"].to(self.device)

        self.blur_Colors = input["blur_Colors"].to(self.device)

    def train(self):
        for model in self.models:
            model.train()

    def validate(self):
        with torch.no_grad():
            self.fake_C, self.fake_B = self.netG(self.real_A, self.real_Colors)
            self.loss_G_L1_val = torch.nn.functional.l1_loss(self.fake_C, self.real_C)
            self.loss_G_L1_B_val = torch.nn.functional.l1_loss(self.fake_B, self.real_B)
            return (
                self.real_A,
                self.fake_B,
                self.real_B,
                self.fake_C,
                self.real_C,
                self.loss_G_L1_B_val,
                self.loss_G_L1_val,
            )

    def test(self):
        with torch.no_grad():
            self.fake_C, self.fake_B = self.netG(self.real_A, self.real_Colors)
            test_l1_loss = torch.nn.functional.l1_loss(self.fake_C, self.real_C)
            return (
                self.real_A,
                self.fake_B,
                self.real_B,
                self.fake_C,
                self.real_C,
                test_l1_loss,
            )

    def forward(self):
        # generate fake_C
        self.fake_C, self.fake_B = self.netG(self.real_A, self.real_Colors)
        # vgg
        self.vgg_fake_C = self.vgg19(self.fake_C)
        self.vgg_real_C = self.vgg19(self.real_C)
        self.vgg_fake_B = self.vgg19(self.fake_B)
        self.vgg_real_B = self.vgg19(self.real_B)

        # local blocks
        (
            self.fake_C_blocks,
            self.real_color_blocks,
            self.blur_color_blocks,
        ) = self.generate_random_block(self.fake_C, self.real_Colors, self.blur_Colors)

        if self.opt.conditional_D:  # TODO: conditional_D necessary?
            self.fake_data_B = torch.cat([self.real_A, self.fake_B], 1)
            self.real_data_B = torch.cat([self.real_A, self.real_B_G], 1)
            self.fake_data_C = torch.cat([self.real_A, self.fake_C], 1)
            self.real_data_C = torch.cat([self.real_A, self.real_C_G], 1)
        else:
            self.fake_data_B = self.fake_B
            self.real_data_B = self.real_B_G
            self.fake_data_C = self.fake_C
            self.real_data_C = self.real_C_G

    def backward_D(self, netD, real, fake, blur=None):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        # blur
        loss_D_blur = 0.0
        if blur is not None:
            pred_blur = netD(blur)
            loss_D_blur, _ = self.criterionGAN(pred_blur, False)

        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real + loss_D_blur
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real, loss_D_blur]

    def backward_G_GAN(self, fake, net=None, ll=0.0):
        if ll > 0.0:
            pred_fake = net(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_G(self):
        # 1. G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(
            self.fake_data_C, self.netD, self.opt.lambda_GAN
        )
        self.loss_G_GAN_B = self.backward_G_GAN(
            self.fake_data_B, self.netD_B, self.opt.lambda_GAN_B
        )

        # 2. reconstruction |fake_C-real_C| |fake_B-real_B|
        self.loss_G_L1 = 0.0
        self.loss_G_L1_B = 0.0
        if self.opt.lambda_L1 > 0.0:
            L1 = torch.mean(
                torch.mean(
                    torch.mean(self.criterionL1(self.fake_C, self.real_C), dim=1), dim=1
                ),
                dim=1,
            )
            self.loss_G_L1 = torch.sum(self.label * L1) * self.opt.lambda_L1
            L1_B = torch.mean(
                torch.mean(
                    torch.mean(self.criterionL1(self.fake_B, self.real_B), dim=1), dim=1
                ),
                dim=1,
            )
            self.loss_G_L1_B = torch.sum(self.label * L1_B) * self.opt.lambda_L1_B

        # 3. contextual loss
        self.loss_G_CX = 0.0
        self.loss_G_CX_B = 0.0
        if self.opt.lambda_CX > 0.0:
            for layer in self.vgg_layers:
                # symmetric contextual loss
                _, cx_batch = self.criterionCX(
                    self.vgg_real_C[layer], self.vgg_fake_C[layer]
                )
                self.loss_G_CX += torch.sum(cx_batch * self.label) * self.opt.lambda_CX
                _, cx_B_batch = self.criterionCX(
                    self.vgg_real_B[layer], self.vgg_fake_B[layer]
                )
                self.loss_G_CX_B += (
                    torch.sum(cx_B_batch * self.label) * self.opt.lambda_CX_B
                )

        # 4. local adv loss
        self.loss_local_adv = 0.0
        if self.opt.lambda_local_D > 0.0:
            self.loss_local_adv = self.backward_G_GAN(
                self.fake_C_blocks, self.netD_local, self.opt.lambda_local_D
            )

        self.loss_G = (
            self.loss_G_GAN
            + self.loss_G_GAN_B
            + self.loss_G_L1
            + self.loss_G_L1_B
            + self.loss_G_CX
            + self.loss_G_CX_B
            + self.loss_local_adv
        )

        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        # update D
        if self.opt.lambda_GAN > 0.0:
            self.setRequiresGrad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(
                self.netD, self.real_data_C, self.fake_data_C
            )
            self.optimizer_D.step()

        if self.opt.lambda_GAN_B > 0.0:
            self.setRequiresGrad(self.netD_B, True)
            self.optimizer_D_B.zero_grad()
            self.loss_D_B, self.losses_D_B = self.backward_D(
                self.netD_B, self.real_data_B, self.fake_data_B
            )
            self.optimizer_D_B.step()

        if self.opt.lambda_local_D > 0.0:
            self.setRequiresGrad(self.netD_local, True)
            self.optimizer_Dlocal.zero_grad()
            self.loss_Dlocal, self.losses_Dlocal = self.backward_D(
                self.netD_local,
                self.real_color_blocks,
                self.fake_C_blocks,
                self.blur_color_blocks,
            )
            self.optimizer_Dlocal.step()

    def update_G(self):
        # update dual net G
        self.setRequiresGrad(self.netD, False)
        self.setRequiresGrad(self.netD_B, False)
        # TODO: not required to set require grad for netD_local false?
        # self.set_requires_grad(self.netD_local, False)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G()
        self.update_D()

    def pretrain_D_local(self):
        self.forward()
        self.update_D()

    def generate_random_block(self, input, target, blurs):
        batch_size, _, height, width = target.size()  # B X 3*nencode X 64 X 64
        target_tensor = target.data
        block_size = self.opt.block_size
        img_size = self.opt.fineSize  # 64

        inp_blk = list()
        tar_blk = list()
        blr_blk = list()

        for b in range(batch_size):
            for i in range(self.opt.block_num):
                rand_idx = random.randint(0, self.opt.nencode - 1)
                x = random.randint(0, height - block_size - 1)
                y = random.randint(0, width - block_size - 1)
                target_random_block = torch.tensor(
                    target_tensor[
                        b,
                        rand_idx * 3 : (rand_idx + 1) * 3,
                        x : x + block_size,
                        y : y + block_size,
                    ],
                    requires_grad=False,
                )
                target_blur_block = torch.tensor(
                    blurs[
                        b,
                        rand_idx * 3 : (rand_idx + 1) * 3,
                        x : x + block_size,
                        y : y + block_size,
                    ],
                    requires_grad=False,
                )
                if i == 0:
                    target_blocks = target_random_block
                    blur_blocks = target_blur_block
                else:
                    target_blocks = torch.cat([target_blocks, target_random_block], 0)
                    blur_blocks = torch.cat([blur_blocks, target_random_block], 0)

                x1 = random.randint(0, img_size - block_size)
                y1 = random.randint(0, img_size - block_size)
                input_random_block = torch.tensor(
                    input.data[b, :, x1 : x1 + block_size, y1 : y1 + block_size],
                    requires_grad=False,
                )
                if i == 0:
                    input_blocks = input_random_block
                else:
                    input_blocks = torch.cat([input_blocks, target_random_block], 0)

            input_blocks = torch.unsqueeze(input_blocks, 0)
            target_blocks = torch.unsqueeze(target_blocks, 0)
            blur_blocks = torch.unsqueeze(blur_blocks, 0)
            inp_blk.append(input_blocks)
            tar_blk.append(target_blocks)
            blr_blk.append(blur_blocks)

        inp_blk = torch.cat(inp_blk)
        tar_blk = torch.cat(tar_blk)
        blr_blk = torch.cat(blr_blk)

        return inp_blk, tar_blk, blr_blk

    def setup(self, opt):
        # TODO may not need it no optimizer
        if opt.isTrain:
            self.schedulers = [
                get_scheduler(optimizer, opt) for optimizer in self.optimizers
            ]

        # if not self.isTrain or opt.continue_train:
        #     self.load_networks(opt.epoch)
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
        lr = self.optimizers[0].param_groups[0]["lr"]
        print(f"learning rate = {lr:.7f}")

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
                errors_ret[name] = float(getattr(self, "loss_" + name))
        return errors_ret

    def eval(self):
        """
        Make models eval mode during test time
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def saveNetworks(self, epoch):
        """
        Save models to the disk
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net" + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda()
                    net = torch.nn.DataParallel(net)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    # load models from the disk
    def loadNetworks(self, epoch):
        """
        Load models from the disk
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pth" % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print(f"loading the model from {load_path}")
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # need to copy keys here because we mutate in loop
                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(
                        state_dict, net, key.split(".")
                    )
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """
        Print network information
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(
                    f"[Network %{name}] Total number \
                of parameters : {num_params/1e6:.3f} M"
                )
        print("-----------------------------------------------")
