import models.ftgan_networks as ftgan_networks
import torch
import random

from .base_model import BaseModel


class GASModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values
        parser.set_defaults(norm="batch", netG="FTGAN_MLAN", dataset_mode="font")

        if is_train:
            parser.set_defaults(
                # batch_size=256, pool_size=0, gan_mode="hinge", netD="basic_64"
                batch_size=128,
                pool_size=0,
                gan_mode="hinge",
                netD="basic_64",
            )
            parser.add_argument(
                "--lambda_L1", type=float, default=100.0, help="weight for L1 loss"
            )
            parser.add_argument(
                "--lambda_style", type=float, default=1.0, help="weight for style loss"
            )
            parser.add_argument(
                "--lambda_content",
                type=float,
                default=1.0,
                help="weight for content loss",
            )
            parser.add_argument(
                "--dis_2", default=True, help="use two discriminators or not"
            )
            parser.add_argument(
                "--use_local_D", default=True, help="use local discriminator"
            )
            parser.add_argument(
                "--lambda_local", type=float, default=0.1, help="weight for local loss"
            )
            parser.add_argument(
                "--num_block", type=int, default=4, help="number of random blocks"
            )
            parser.add_argument("--block_size", type=int, default=32, help="block size")
            parser.add_argument("--use_spectral_norm", default=True)
        return parser

    def __init__(self, opt):
        """Initialize the font_translator_gan class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.style_channel = opt.style_channel
        # used for generating blocks

        if self.isTrain:
            # For training G + D
            self.num_block = opt.num_block
            self.block_size = opt.block_size
            self.num_dis = int(opt.dis_2 + opt.use_local_D + 1)
            self.visual_names = ["gt_images", "generated_images"] + [
                "style_images_{}".format(i) for i in range(self.style_channel)
            ]
            if self.num_dis == 3:
                # 3 discriminator: content + style + local
                self.model_names = ["G", "D_content", "D_style", "D_local"]
                self.loss_names = ["G_GAN", "G_L1", "D_content", "D_style", "D_local"]
            elif self.num_dis == 2:
                # 2 discriminator: content + style
                self.model_names = ["G", "D_content", "D_style"]
                self.loss_names = ["G_GAN", "G_L1", "D_content", "D_style"]
            else:
                # 1 discriminator: all in one
                self.model_names = ["G", "D"]
                self.loss_names = ["G_GAN", "G_L1", "D"]
        else:
            # For test and evaluation only G
            self.visual_names = ["gt_images", "generated_images"]
            self.model_names = ["G"]
        # define networks (both generator and discriminator)
        self.netG = ftgan_networks.define_G(
            self.style_channel + 1,  # number of style image + 1 content image
            1,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

        if self.isTrain:
            # define discriminators;
            # conditional GANs need to take both input and output images;
            # Therefore, #channels for D is input_nc + output_nc
            if self.num_dis == 3:
                self.netD_local = ftgan_networks.define_D(
                    2 * self.num_block,
                    opt.ndf,
                    opt.netD,
                    opt.n_layers_D,
                    opt.norm,
                    opt.init_type,
                    opt.init_gain,
                    self.gpu_ids,
                    use_spectral_norm=opt.use_spectral_norm,
                )
                self.netD_content = ftgan_networks.define_D(
                    2,  # content + fake/gt image
                    opt.ndf,
                    opt.netD,
                    opt.n_layers_D,
                    opt.norm,
                    opt.init_type,
                    opt.init_gain,
                    self.gpu_ids,
                    use_spectral_norm=opt.use_spectral_norm,
                )
                self.netD_style = ftgan_networks.define_D(
                    self.style_channel + 1,  # style images + fake/gt image
                    opt.ndf,
                    opt.netD,
                    opt.n_layers_D,
                    opt.norm,
                    opt.init_type,
                    opt.init_gain,
                    self.gpu_ids,
                    use_spectral_norm=opt.use_spectral_norm,
                )
            elif self.num_dis == 2:
                self.netD_content = ftgan_networks.define_D(
                    2,  # content + fake/gt image
                    opt.ndf,
                    opt.netD,
                    opt.n_layers_D,
                    opt.norm,
                    opt.init_type,
                    opt.init_gain,
                    self.gpu_ids,
                    use_spectral_norm=opt.use_spectral_norm,
                )
                self.netD_style = ftgan_networks.define_D(
                    self.style_channel + 1,  # style images + fake/gt image
                    opt.ndf,
                    opt.netD,
                    opt.n_layers_D,
                    opt.norm,
                    opt.init_type,
                    opt.init_gain,
                    self.gpu_ids,
                    use_spectral_norm=opt.use_spectral_norm,
                )
            else:
                self.netD = ftgan_networks.define_D(
                    self.style_channel + 2,
                    opt.ndf,
                    opt.netD,
                    opt.n_layers_D,
                    opt.norm,
                    opt.init_type,
                    opt.init_gain,
                    self.gpu_ids,
                    use_spectral_norm=opt.use_spectral_norm,
                )

        if self.isTrain:
            # define loss functions
            self.lambda_L1 = opt.lambda_L1
            self.criterionGAN = ftgan_networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            if self.num_dis == 3:
                self.lambda_style = opt.lambda_style
                self.lambda_content = opt.lambda_content
                self.lambda_local = opt.lambda_local
                self.optimizer_D_content = torch.optim.Adam(
                    self.netD_content.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizer_D_style = torch.optim.Adam(
                    self.netD_style.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizer_D_local = torch.optim.Adam(
                    self.netD_style.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizers.append(self.optimizer_D_content)
                self.optimizers.append(self.optimizer_D_style)
                self.optimizers.append(self.optimizer_D_local)
            elif self.num_dis == 2:
                self.lambda_style = opt.lambda_style
                self.lambda_content = opt.lambda_content
                self.optimizer_D_content = torch.optim.Adam(
                    self.netD_content.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizer_D_style = torch.optim.Adam(
                    self.netD_style.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizers.append(self.optimizer_D_content)
                self.optimizers.append(self.optimizer_D_style)
            else:
                self.optimizer_D = torch.optim.Adam(
                    self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizers.append(self.optimizer_D)

    def set_input(self, data):
        # ground truth image
        self.gt_images = data["gt_images"].to(self.device)
        # content image
        self.content_images = data["content_images"].to(self.device)
        # style image
        self.style_images = data["style_images"].to(self.device)
        if not self.isTrain:
            self.image_paths = data["image_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.generated_images = self.netG((self.content_images, self.style_images))
        if self.isTrain and self.num_dis == 3:
            self.fake_blocks, self.style_blocks, self.gt_blocks = self.cut_patch(
                self.generated_images, self.style_images, self.gt_images
            )

    def compute_gan_loss_D(self, real_images, fake_images, netD):
        # Fake
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real = torch.cat(real_images, 1)
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    def compute_gan_loss_G(self, fake_images, netD):
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True, True)
        return loss_G_GAN

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        if self.num_dis == 3:
            self.loss_D_local = self.compute_gan_loss_D(
                [self.style_blocks, self.gt_blocks],
                [self.style_blocks, self.fake_blocks],
                self.netD_local,
            )
            self.loss_D_content = self.compute_gan_loss_D(
                [self.content_images, self.gt_images],
                [self.content_images, self.generated_images],
                self.netD_content,
            )
            self.loss_D_style = self.compute_gan_loss_D(
                [self.style_images, self.gt_images],
                [self.style_images, self.generated_images],
                self.netD_style,
            )
            self.loss_D = (
                self.lambda_content * self.loss_D_content
                + self.lambda_style * self.loss_D_style
                + self.lambda_local * self.loss_D_local
            )
        elif self.num_dis == 2:
            self.loss_D_content = self.compute_gan_loss_D(
                [self.content_images, self.gt_images],
                [self.content_images, self.generated_images],
                self.netD_content,
            )
            self.loss_D_style = self.compute_gan_loss_D(
                [self.style_images, self.gt_images],
                [self.style_images, self.generated_images],
                self.netD_style,
            )
            self.loss_D = (
                self.lambda_content * self.loss_D_content
                + self.lambda_style * self.loss_D_style
            )
        else:
            self.loss_D = self.compute_gan_loss_D(
                [self.content_images, self.style_images, self.gt_images],
                [self.content_images, self.style_images, self.generated_images],
                self.netD,
            )

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.num_dis == 3:
            self.loss_G_content = self.compute_gan_loss_G(
                [self.content_images, self.generated_images], self.netD_content
            )
            self.loss_G_style = self.compute_gan_loss_G(
                [self.style_images, self.generated_images], self.netD_style
            )
            self.loss_G_local = self.compute_gan_loss_G(
                [self.style_blocks, self.fake_blocks], self.netD_local
            )
            self.loss_G_GAN = (
                self.lambda_content * self.loss_G_content
                + self.lambda_style * self.loss_G_style
                + self.lambda_local * self.loss_G_local
            )
        elif self.num_dis == 2:
            self.loss_G_content = self.compute_gan_loss_G(
                [self.content_images, self.generated_images], self.netD_content
            )
            self.loss_G_style = self.compute_gan_loss_G(
                [self.style_images, self.generated_images], self.netD_style
            )
            self.loss_G_GAN = (
                self.lambda_content * self.loss_G_content
                + self.lambda_style * self.loss_G_style
            )
        else:
            self.loss_G_GAN = self.compute_gan_loss_G(
                [self.content_images, self.style_images, self.generated_images],
                self.netD,
            )

        # Second, G(A) = B
        self.loss_G_L1 = (
            self.criterionL1(self.generated_images, self.gt_images) * self.opt.lambda_L1
        )
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        if self.num_dis == 3:
            self.set_requires_grad([self.netD_content, self.netD_style, self.netD_local], True)
            self.optimizer_D_content.zero_grad()
            self.optimizer_D_style.zero_grad()
            self.optimizer_D_local.zero_grad()
            self.backward_D()
            self.optimizer_D_content.step()
            self.optimizer_D_style.step()
            self.optimizer_D_local.step()

        elif self.num_dis == 2:
            self.set_requires_grad([self.netD_content, self.netD_style], True)
            self.optimizer_D_content.zero_grad()
            self.optimizer_D_style.zero_grad()
            self.backward_D()
            self.optimizer_D_content.step()
            self.optimizer_D_style.step()
        else:
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights
        # update G
        if self.num_dis == 3:
            self.set_requires_grad([self.netD_content, self.netD_style, self.netD_local], False)
        elif self.num_dis == 2:
            self.set_requires_grad([self.netD_content, self.netD_style], False)
        else:
            self.set_requires_grad(
                self.netD, False
            )  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def compute_visuals(self):
        if self.isTrain:
            self.netG.eval()
            with torch.no_grad():
                self.forward()
            for i in range(self.style_channel):
                setattr(
                    self,
                    "style_images_{}".format(i),
                    torch.unsqueeze(self.style_images[:, i, :, :], 1),
                )
            self.netG.train()
        else:
            pass

    def cut_patch(self, fake_images, real_images, gt_images):
        # B, num_image, H(64), W(64)
        batch_size, num_style, H, W = real_images.shape
        fake_blk, real_blk, gt_blk = [], [], []
        block_size = self.block_size
        num_block = self.num_block
        for b in range(batch_size):
            for i in range(num_block):
                style_idx = random.randint(0, num_style-1)
                x = random.randint(0, H - block_size - 1)
                y = random.randint(0, W - block_size - 1)

                real_random_block = real_images.clone().detach().requires_grad_(False)
                real_random_block = real_random_block[b, style_idx, x:x+block_size, y:y+block_size].unsqueeze(0)

                fake_random_block = fake_images.clone().detach().requires_grad_(False)
                fake_random_block = fake_random_block[b, 0, x:x+block_size, y:y+block_size].unsqueeze(0)

                gt_random_block = gt_images.clone().detach().requires_grad_(False)
                gt_random_block = gt_random_block[b, 0, x:x+block_size, y:y+block_size].unsqueeze(0)

                if i == 0:
                    real_blocks = real_random_block
                    fake_blocks = fake_random_block
                    gt_blocks = gt_random_block
                else:
                    real_blocks = torch.cat([real_blocks, real_random_block], dim=0)
                    fake_blocks = torch.cat([fake_blocks, fake_random_block], dim=0)
                    gt_blocks = torch.cat([gt_blocks, gt_random_block], dim=0)
            real_blk.append(real_blocks.unsqueeze(0))
            fake_blk.append(fake_blocks.unsqueeze(0))
            gt_blk.append(gt_blocks.unsqueeze(0))

        real_blk = torch.cat(real_blk)
        fake_blk = torch.cat(fake_blk)
        gt_blk = torch.cat(gt_blk)
        return fake_blk, real_blk, gt_blk

