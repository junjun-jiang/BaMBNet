from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import json
import torch
import torchvision.transforms.functional as TF
import cv2


class TrainDataset(Dataset):
    def __init__(self, trainopt):
        super(TrainDataset, self).__init__()
        path = trainopt['dataroot']
        self.image_size = trainopt['image_size']
        trainpairs_forreading = str(trainopt['trainpairs'])
        if not os.path.exists(trainpairs_forreading):
            left_imgs = os.path.join(os.path.expanduser(path), trainopt["left_name"])
            right_imgs = os.path.join(os.path.expanduser(path), trainopt["right_name"])
            combine_imgs = os.path.join(os.path.expanduser(path), trainopt["combine_name"])

            left_imgs = [os.path.join(left_imgs, os_dir) for os_dir in os.listdir(left_imgs)]
            right_imgs = [os.path.join(right_imgs, os_dir) for os_dir in os.listdir(right_imgs)]
            combine_imgs = [os.path.join(combine_imgs, os_dir) for os_dir in os.listdir(combine_imgs)]
            left_imgs.sort()
            right_imgs.sort()
            combine_imgs.sort()

            self.uegt_imgs = {}
            for l_img, r_img, c_img in zip(left_imgs, right_imgs, combine_imgs):
                self.uegt_imgs[c_img] = [l_img, r_img]
            with open(trainpairs_forreading, 'w') as f:
                json.dump(self.uegt_imgs, f)
        else:
            with open(trainpairs_forreading, 'r') as f:
                self.uegt_imgs = json.load(f)
        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.uegt_imgs)

    def random_augmentation(self, under):
        w, h = under.size
        w_start = w - self.image_size
        h_start = h - self.image_size

        random_w = 1 if w_start <= 1 else torch.randint(low=1, high=w_start, size=(1, 1)).item()
        random_h = 1 if h_start <= 1 else torch.randint(low=1, high=h_start, size=(1, 1)).item()
        return random_w, random_h

    def __getitem__(self, index):
        c_img, (l_img, r_img) = self.uegt_imgs[index]
        l_img = cv2.imread(l_img, -1)
        r_img = cv2.imread(r_img, -1)
        l_img = torch.tensor(l_img / 65535.0).float().permute(2, 0, 1)
        r_img = torch.tensor(r_img / 65535.0).float().permute(2, 0, 1)
        if l_img.size(2) < l_img.size(1):
            l_img = TF.rotate(l_img, 90)
            r_img = TF.rotate(r_img, 90)

        return l_img, r_img


class ValDataset(Dataset):
    def __init__(self, valopt):
        super(ValDataset, self).__init__()
        self.rotation =  transforms.RandomRotation([0, 180])
        path = valopt['dataroot']

        left_imgs = os.path.join(os.path.expanduser(path), valopt["left_name"])
        right_imgs = os.path.join(os.path.expanduser(path), valopt["right_name"])
        combine_imgs = os.path.join(os.path.expanduser(path), valopt["combine_name"])

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
        l_img = torch.tensor(l_img / 65535.0).float().permute(2, 0, 1)
        r_img = torch.tensor(r_img / 65535.0).float().permute(2, 0, 1)

        if l_img.size(2) < l_img.size(1):
            l_img = TF.rotate(l_img, 90)
            r_img = TF.rotate(r_img, 90)

        return l_img, r_img,  os.path.basename(c_img_name).split('.')


class TestDataset(Dataset):
    def __init__(self, testopt):
        super(TestDataset, self).__init__()
        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
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
        l_img = torch.tensor(l_img / 65535.0).float().permute(2, 0, 1)
        r_img = torch.tensor(r_img / 65535.0).float().permute(2, 0, 1)

        if l_img.size(1) > l_img.size(2):
            l_img = TF.rotate(l_img, 90)
            r_img = TF.rotate(r_img, 90)

        return l_img, r_img, os.path.basename(c_img_name).split('.')
