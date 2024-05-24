# -*- coding: utf-8 -*-
import numpy as np
import random
from monai.transforms import (
    Orientationd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandAffined,
    Rand2DElasticd,
    GaussianSmoothd,
)
# -*- coding: utf-8 -*-
import torch
from torch.utils import data
import numpy as np
import SimpleITK as sitk
import warnings

warnings.filterwarnings("ignore")




def transform_img_lab(image, label, args):
    data_dicts = {"image": image, "label": label}
    orientation = Orientationd(keys=["image", "label"], as_closest_canonical=True)
    rand_affine = RandAffined(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        spatial_size=(args.ROI_shape, args.ROI_shape),
        translate_range=(30, 30),
        rotate_range=(np.pi / 36, np.pi / 36),
        scale_range=(0.15, 0.15),
        padding_mode="zeros",
    )
    rand_elastic = Rand2DElasticd(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        spacing=(20, 20),
        magnitude_range=(1, 1),
        spatial_size=(args.ROI_shape, args.ROI_shape),
        translate_range=(10, 20),
        rotate_range=(np.pi / 36, np.pi / 36),
        scale_range=(0.15, 0.15),
        padding_mode="zeros",
    )
    scale_shift = ScaleIntensityRanged(
        keys=["image"], a_min=-10, a_max=10, b_min=-10, b_max=10, clip=True
    )
    gauss_noise = RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=1)
    gauss_smooth = GaussianSmoothd(keys=["image"], sigma=0.6, approx="erf")

    # Params: Hyper parameters for data augumentation, number (like 0.3) refers to the possibility
    if random.random() > 0.2:
        if random.random() > 0.3:
            data_dicts = orientation(data_dicts)
        if random.random() > 0.3:
            if random.random() > 0.5:
                data_dicts = rand_affine(data_dicts)
            else:
                data_dicts = rand_elastic(data_dicts)
        if random.random() > 0.5:
            data_dicts = scale_shift(data_dicts)
        if random.random() > 0.5:
            if random.random() > 0.5:
                data_dicts = gauss_noise(data_dicts)
            else:
                data_dicts = gauss_smooth(data_dicts)
    else:
        data_dicts = data_dicts
    return data_dicts

def read_file_from_txt(txt_path):
    files = []
    for line in open(txt_path, "r"):
        files.append(line.strip())
    return files


class Dataloader(data.Dataset):
    def __init__(self, args):
        super(Dataloader, self).__init__()
        self.image_file = read_file_from_txt(args.Image_Tr_txt)
        self.label_file = read_file_from_txt(args.Label_Tr_txt)
        self.shape = (args.ROI_shape, args.ROI_shape)
        self.args = args

    def __getitem__(self, index):
        image_path = self.image_file[index]
        label_path = self.label_file[index]

        # Read images and labels
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)

        y, x = image.shape
        image = image.astype(dtype=np.float32)
        label = label.astype(dtype=np.float32)

        # Normalization
        mean, std = np.load(self.args.Tr_Meanstd_name)
        image = (image - mean) / std
        label = label / (np.max(label))

        # Random crop, (center_y, center_x) refers the left-up coordinate of the Random_Crop_Block
        center_y = np.random.randint(0, y - self.shape[0] + 1, 1, dtype=np.int16)[0]
        center_x = np.random.randint(0, x - self.shape[1] + 1, 1, dtype=np.int16)[0]
        image = image[
                center_y: self.shape[0] + center_y, center_x: self.shape[1] + center_x
                ]
        label = label[
                center_y: self.shape[0] + center_y, center_x: self.shape[1] + center_x
                ]

        image = image[np.newaxis, :, :]
        label = label[np.newaxis, :, :]

        # Data Augmentation
        data_dict = transform_img_lab(image, label, self.args)
        image_trans = data_dict["image"]
        label_trans = data_dict["label"]
        if isinstance(image_trans, torch.Tensor):
            image_trans = image_trans.numpy()
        if isinstance(label_trans, torch.Tensor):
            label_trans = label_trans.numpy()

        return image_trans, label_trans

    def __len__(self):
        return len(self.image_file)


