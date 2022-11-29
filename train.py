from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images

import copy
import wandb
import time

def train_step(model, dataset, opt, total_steps, wandb_run):
    model.train()
    batch_size = opt.batch_size

    for i, data in enumerate(dataset):
        iter_start_t = time.time()
        total_steps += batch_size

        model.setInput(data)
        if not opt.isTrain:
            continue
        model.optimize_parameters()

        # Removed display_freq

        if total_steps % opt.print_freq == 0:
            losses = model.getCurrentLosses()
            t = (time.time() - iter_start_t) / batch_size

            info = {"epoch": epoch, "step": total_steps, "time": t}
            for loss_key, loss_val in losses.items():
                info[loss_key] = loss_val

            if opt.use_wandb:
                wandb_run.log(info)

            print(info)
        
        if total_steps % opt.save_latest_freq == 0:
            print("saving model at epoch {} total_steps {}".format(epoch, total_steps))
            model.saveNetworks('latest')
    
    return total_steps

def eval_step(model, dataset, opt, epoch, wandb_run=None):
    model.eval()
    l1_B_loss = 0.0
    l1_C_loss = 0.0
    n = 0

    for i, data in enumerate(dataset):
        model.setInput(data)
        real_in, fake_out_B, real_out_B, fake_out, real_out, loss_B, loss_C = model.validate()
        l1_B_loss += loss_B
        l1_C_loss += loss_C
        n += 1

        ABC_path = data['ABC_path']
        print("ABC_path len", len(ABC_path))

        for idx in range(len(ABC_path)):
            names = ['real', 'fake', 'real_B', 'fake_B']
            images = []
            for img_out in [real_out, fake_out, real_out_B, fake_out_B]:
                images.append(img_out[idx].unsqueeze(0))
            images_path = str(epoch) + '_' + ABC_path[idx].split('/')[-1].split('.')[0]
            save_images(images, names, images_path, opt=opt, aspect_ratio=1.0, width=opt.fineSize)

    l1_B_loss /= n
    l1_C_loss /= n
    info = {
        "epoch": epoch,
        "val_l1_B_loss": l1_B_loss, 
        "val_l1_C_loss": l1_C_loss
    }

    if opt.use_wandb:
        wandb_run.log(info)
    print(info)


if __name__ == '__main__':
    # Parse Train Options
    opt = TrainOptions().parse()
    print(opt)

    # DataLoader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    if opt.validate_freq > 0:
        val_opt = copy.deepcopy(opt)
        val_opt.phase = 'val'
        val_opt.serial_batches = True  # no shuffle
        val_data_loader = CreateDataLoader(val_opt)
        val_dataset = val_data_loader.load_data()
        val_dataset_size = len(val_data_loader)
        print('#validation images = %d' % val_dataset_size)
    
    model = create_model(opt)
    model.setup(opt)

    # WandB Config
    run = None
    if opt.use_wandb:
        wandb.login(key=opt.wandb_key) #API Key is in your wandb account, under settings (wandb.ai/settings)
        opt_dict = vars(opt)
        run = wandb.init(
            name = "train",
            reinit = True,
            project = "gasnet",
            config = opt_dict
        )

    total_steps = 0
    batch_size = opt.batch_size

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_t = time.time()
        
        total_steps = train_step(model, dataset, opt, total_steps, run)
        
        if epoch % opt.save_epoch_freq == 0:
            print("saving model AFTER epoch {} total_steps {}".format(epoch, total_steps))
            model.saveNetworks('latest')
            model.saveNetworks(epoch)

        if opt.validate_freq > 0 and epoch % opt.validate_freq == 0:
            eval_step(model, val_dataset, val_opt, epoch, run)
        
        t = time.time() - epoch_start_t
        print("End of epoch {}/{} \t Time Taken: {} sec".format(epoch, opt.niter + opt.niter_decay, t))

        model.updateLearningRate()
