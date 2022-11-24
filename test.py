from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util import html
from util.visualizer import save_images

import wandb
from itertools import islice
import os

def test_step(model, dataset, opt):
    l1_loss = 0.0
    n = 0

    # TO DO: remove webpage
    web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
    webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class = %s' % (opt.name, opt.phase, opt.name))

    for i, data in enumerate(islice(dataset, opt.num_test)):
        print('process input image {}/{}'.format(i, opt.num_test))

        model.set_input(data)
        
        real_in, fake_out_B, real_out_B, fake_out, real_out, l1_loss = model.test()
        l1_loss += l1_loss.item()
        n += 1

        names = ['real', 'fake']
        images = [real_out, fake_out]
        ABC_path = data['ABC_path'][0]
        img_path = ABC_path.split('/')[-1].split('.')[0]
        
        save_images(images, names, img_path, webpage=webpage, width=opt.fineSize)
    
    webpage.save()
    l1_loss /= n
    return l1_loss

if __name__ == '__main__':
    # Parse Test Options
    opt = TestOptions().parse()
    print(opt)

    # TO DO: Merge with test_options.py
    opt.num_threads = 1   # test code only supports num_threads=1
    opt.batch_size = 1   # test code only supports batch_size=1
    opt.serial_batches = True  # no shuffle

    # DataLoader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#testing images = %d' % dataset_size)

    # TO DO: interface with create_model
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    print("Loading model {}".format(opt.model))

    # WandB Config
    run = None
    # if opt.use_wandb:
    #     wandb.login(key=opt.wandb_key) #API Key is in your wandb account, under settings (wandb.ai/settings)
    #     opt_dict = vars(opt)
    #     run = wandb.init(
    #         name = "test",
    #         reinit = True,
    #         project = "gasnet",
    #         config = opt_dict
    #     )
    
    test_loss = test_step(model, dataset, opt)

    l1_loss_file = os.path.join(opt.results_dir, opt.phase, "l1_loss.txt")
    with open(l1_loss_file, "w") as f:
        f.write(str(test_loss))