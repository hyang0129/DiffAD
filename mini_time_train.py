import argparse
import logging
import math

import torch
from tensorboardX import SummaryWriter

import core.logger as Logger
import data as Data
import model as Model
from tqdm import tqdm

import core.metrics as Metrics
import numpy as np

from mini_model import get_mini_model

# train model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/smap_time_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)

    logger.info('Initial Dataset Finished')

    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']


    logger.info('Generating Minimodel Training Set')

    params = {
        'opt': opt,
        'logger': logger,
        'row_num': train_set.row_num,
        'col_num': train_set.col_num
    }

    device = 'cuda'


    import os
    save_to = f'{os.path.basename(args.config).split(".")[0]}.npz'


    if os.path.exists(save_to):
        logger.info('Existing NPZ archive exists')

        recon_data = np.load(save_to)
    else:
        raise FileNotFoundError

    from torch.utils.data import Dataset
    class GeneratedData(Dataset):
        def __init__(self, recon_data):
            self.recon_data = {k:v for k,v in recon_data.items()}

            try:
                _ = self.recon_data['label']
            except KeyError:
                self.recon_data['label'] = self.recon_data['differ']

        def __len__(self):
            return len(self.recon_data['HR'])

        def __getitem__(self, index):
            return {'HR': self.recon_data['HR'], 'SR': self.recon_data['SR'], 'differ': self.recon_data['differ'] ,  'label': self.recon_data['label'] }


    train_set = GeneratedData(recon_data)
    train_loader = Data.create_dataloader(train_set, dataset_opt, phase='train')

    sample_data = train_set[0]
    channels = sample_data['HR'].shape[-1]

    for k,v in sample_data.items():
        print(k, v.shape )

    print(len(train_set))

    reg_models, loss_fn, reg_optim, reg_base = get_mini_model(input_channels=channels * 2)



    # for i in tqdm(range(n_epoch), desc = 'epochs'):
    #     for ii, train_data in pbar := tqdm(enumerate(train_loader)):
    #         # # go from B1TC to BTC
    #         inp = torch.squeeze(torch.cat([train_data['HR'], train_data['SR']], dim = -1))
    #         inp = inp.to(device)
            #
            #
            # targets = targets.to(device)
            #
            # print(targets.shape)
            #
            # predictions = [torch.squeeze(m(inp)) for m in reg_models]
            #
            # print(predictions[0].shape)
            #
            # losses = [loss_fn(pred, targets) for pred in predictions]
            #
            # for i, loss in enumerate(losses):
            #     reg_optim[i].zero_grad()
            #     loss.backward()
            #     reg_optim[i].step()
            #
            # epoch_losses.append(torch.mean(torch.stack(losses, -1)).detach().cpu())
            #
            # pbar.set_description(f'''
            # Training
            # Loss : {float(torch.mean(torch.stack(epoch_losses)) ):.2f}
            # ''' )





