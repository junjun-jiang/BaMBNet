from torch.utils.data import Dataset
import os
import torch
import torchvision.transforms.functional as TF
import cv2


class TestDataset(Dataset):
    def __init__(self, testopt):
        super(TestDataset, self).__init__()
        path = testopt['dataroot']

        left_imgs = os.path.join(os.path.expanduser(path), testopt["left_name"])
        right_imgs = os.path.join(os.path.expanduser(path), testopt["right_name"])
        combine_imgs = os.path.join(os.path.expanduser(path), testopt["combine_name"])

        left_imgs = [os.path.join(left_imgs, os_dir) for os_dir in os.listdir(left_imgs)]
        right_imgs = [os.path.join(right_imgs, os_dir) for os_dir in os.listdir(right_imgs)]
        combine_imgs = [os.path.join(combine_imgs, os_dir) for os_dir in os.listdir(combine_imgs)]
        left_imgs.sort()
        right_imgs.sort()
        combine_imgs.sort()

        self.uegt_imgs = {}
        for l_img, r_img, c_img in zip(left_imgs, right_imgs, combine_imgs):
            self.uegt_imgs[c_img] = [l_img, r_img]

        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.uegt_imgs)

    def __getitem__(self, index):
        c_img, (l_img, r_img) = self.uegt_imgs[index]
        c_img_name = c_img
        l_img = cv2.imread(l_img, -1)
        r_img = cv2.imread(r_img, -1)
        gt_img = cv2.imread(c_img, -1)
        l_img = torch.tensor(l_img / 65535.0).float().permute(2, 0, 1)
        r_img = torch.tensor(r_img / 65535.0).float().permute(2, 0, 1)
        gt_img = torch.tensor(gt_img / 65535.0).float().permute(2, 0, 1)

        if l_img.size(1) > l_img.size(2):
            l_img = TF.rotate(l_img, 90)
            r_img = TF.rotate(r_img, 90)
            gt_img = TF.rotate(gt_img, 90)

        return l_img, r_img, gt_img, os.path.basename(c_img_name).split('.')
