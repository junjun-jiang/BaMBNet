import argparse
import os
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import util
from data.create_coc_dataset import TestDataset
from models.kernel_de_bparam_net import KernelEDNet
import option.options as option
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Defocus Deblur: Path to option ymal file.')
test_args = parser.parse_args()

opt = option.parse(test_args.opt, is_train=False)
util.mkdir_and_rename(opt['path']['results_root'])  # rename results folder if exists
util.mkdirs((path for key, path in opt['path'].items() if not key == 'results_root'
                     and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)

logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

torch.backends.cudnn.deterministic = True
# convert to NoneDict, which returns None for missing keys
opt = option.dict_to_nonedict(opt)


dataset_opt = opt['dataset']['test']
test_dataset = TestDataset(dataset_opt)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        num_workers=dataset_opt['workers'], pin_memory=True)
logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_dataset)))

# create model
model = KernelEDNet()

# resume for test
device_id = torch.cuda.current_device()
resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
logger.info('Resuming training from epoch: {}.'.format(
    resume_state['epoch']))
model.load_state_dict(resume_state['state_dict'])


model = model.cuda()

# testing
max_steps = len(test_loader)

torch.cuda.empty_cache()
model.eval()
for test_data in tqdm(test_loader):
    with torch.no_grad():
        l_img, r_img, root_name = test_data
        l_img = l_img.cuda()
        r_img = r_img.cuda()
        x = torch.cat((l_img, r_img), dim=1)
        recover = model(x)[0]
        # Save ground truth
        img_dir = opt['path']['test_images']
        recover_img = (recover.abs().squeeze().cpu()/25.0 * 65535.0).int()
        recover_img = recover_img.permute(1, 2, 0).numpy().astype(np.uint16)
        save_img_path_gtr = os.path.join(img_dir,
                                        "{:s}_recover.png".format(root_name[0][0]))
        recover_img.save(save_img_path_gtr)
        cv2.imwrite(save_img_path_gtr, recover_img)

