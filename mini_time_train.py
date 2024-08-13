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
            return {'HR': self.recon_data['HR'][index], 'SR': self.recon_data['SR'][index],
                    'differ': self.recon_data['differ'][index] ,  'label': self.recon_data['label'][index] }


    train_set = GeneratedData(recon_data)
    train_loader = Data.create_dataloader(train_set, dataset_opt, phase='train')

    sample_data = train_set[0]
    channels = sample_data['HR'].shape[-1]

    for k,v in sample_data.items():
        print(k, v.shape )

    print(len(train_set))

    reg_models, loss_fn, reg_optim, reg_base = get_mini_model(input_channels=channels * 2)

    # def evaluate():
    test_save_to = f'{os.path.basename(args.config).split(".")[0]}.npz'.replace('train', 'test')

    test_recon_data = np.load(test_save_to)

    test_set = GeneratedData(test_recon_data)
    test_loader = Data.create_dataloader(test_set, dataset_opt, phase='test')

    import json

    with open(args.config.replace('train', 'test')) as f:
        test_config = json.load(f)

    start_label = test_config['model']['beta_schedule']['test']['start_label']
    end_label = test_config['model']['beta_schedule']['test']['end_label']
    step_label = test_config['model']['beta_schedule']['test']['step_label']
    step_t = test_config['model']['beta_schedule']['test']['step_t']
    strategy_params = {
        'start_label': start_label,
        'end_label': end_label,
        'step_label': step_label,
        'step_t': step_t
    }

    import core.metrics as Metrics
    import pandas as pd
    all_datas = pd.DataFrame(
        {
            'label' : np.reshape(test_recon_data['label'], -1),
            'differ' : np.reshape(test_recon_data['differ'], -1),
        }
                             )
    best_f1, precision, recall  = Metrics.relabeling_strategy(all_datas, strategy_params, return_all=True)

    print(f'Original Scores : P: {precision:.4f}, R: {recall:.4f}, F1: {best_f1:.4f} ')


    for i in tqdm(range(n_epoch), desc = 'epochs'):
        epoch_losses = []
        epoch_model_var = []
        epoch_time_var = []


        for ii, train_data in (pbar := tqdm(enumerate(train_loader), mininterval=0.5)):
            # # go from B1TC to BTC
            inp = torch.squeeze(torch.cat([train_data['HR'], train_data['SR']], dim = -1))
            inp = inp.to(device)

            diffs = train_data['differ']
            diffs = diffs.to(device)

            # print(targets.shape)

            pred_diffs = [torch.squeeze(m(inp)) for m in reg_models]

            # print(predictions[0].shape)

            losses = [loss_fn(pred, diffs) for pred in pred_diffs]

            for i, loss in enumerate(losses):
                reg_optim[i].zero_grad()
                loss.backward()
                reg_optim[i].step()

            epoch_losses.append(torch.mean(torch.stack(losses, -1)).detach().cpu())
            epoch_model_var.append(var_over_models.mean().cpu())
            epoch_time_var.append(var_over_timesteps_of_var_over_models.mean().cpu())

            with torch.no_grad():
                # print(torch.var(torch.var(torch.concat(pred_diffs, dim = -1), dim = -1), dim=-1) .shape   )

                var_over_models = torch.var(torch.concat(pred_diffs, dim = -1), dim = -1)
                var_over_timesteps_of_var_over_models = torch.var(var_over_models, dim = 1)

            pbar.set_description(f'''
            Training 
            Loss : {float(torch.mean(torch.stack(epoch_losses)) ):.2f}  
            model_var : {float(torch.mean(torch.stack(epoch_model_var))):.2f}  
            time_var : {float(torch.mean(torch.stack(epoch_time_var))):.2f}
            ''' )


        print(f'''
        Training
        Loss : {float(torch.mean(torch.stack(epoch_losses))):.2f}  
        model_var : {float(torch.mean(torch.stack(epoch_model_var))):.2f}  
        time_var : {float(torch.mean(torch.stack(epoch_time_var))):.2f}
        ''')


        vars = []

        with torch.no_grad():

            for ii, train_data in (pbar := tqdm(enumerate(test_loader), mininterval=0.5)):

                inp = torch.squeeze(torch.cat([train_data['HR'], train_data['SR']], dim = -1))
                inp = inp.to(device)
                pred_diffs = [torch.squeeze(m(inp)) for m in reg_models]

                var_over_models = torch.var(torch.concat(pred_diffs, dim=-1), dim=-1).cpu()

                vars.append(var_over_models)

        vars = np.array(torch.cat(vars, dim = 0))

        all_datas = pd.DataFrame(
            {
                'label': np.reshape(vars, -1),
                'differ': np.reshape(test_recon_data['differ'], -1),
            }
        )
        best_f1, precision, recall = Metrics.relabeling_strategy(all_datas, strategy_params, return_all=True)


        # for