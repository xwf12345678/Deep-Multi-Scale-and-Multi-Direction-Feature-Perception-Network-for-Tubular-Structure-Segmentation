# -*- coding: utf-8 -*-
from os import listdir
from os.path import join
import numpy as np
import SimpleITK as sitk
import os


"""
The purpose of this code is to calculate the "mean" and "std" of the image, 
which will be used in the subsequent normalization process

Take the image ending with "nii.gz" as an example (using SimpleITK)
"""


def Getmeanstd(args, image_path, meanstd_name):
    """
    :param args: Parameters
    :param image_path: Address of image
    :param meanstd_name: save name of "mean" and "std"  (using ".npy" format to save)
    :return: None
    """
    file_names = [x for x in listdir(join(image_path))]
    mean, std, length = 0.0, 0.0, 0.0

    for file_name in file_names:
        image = sitk.ReadImage(image_path + file_name)
        image = sitk.GetArrayFromImage(image).astype(np.float32)
        length += image.size
        mean += np.sum(image)
        # print(mean, length)
    mean = mean / length

    for file_name in file_names:
        image = sitk.ReadImage(image_path + file_name)
        image = sitk.GetArrayFromImage(image).astype(np.float32)
        std += np.sum(np.square((image - mean)))
        # print(std)
    std = np.sqrt(std / length)
    print("1 Finish Getmeanstd: ", meanstd_name)
    print("Mean and std are: ", mean, std)
    np.save(meanstd_name, [mean, std])
"""
The purpose of this code is to generate ".txt" files in */TXT/ for training and testing
"""


def Get_file_list(file_dir):
    files = os.listdir(file_dir)
    # Sort files named with numbers (Like 1.nii.gz, 2.nii.gz)
    files.sort(key=lambda x: int(x.split(".")[0]))
    files_num = len(files)
    return files, files_num


def Generate_Txt(image_path, txt_name):
    f = open(txt_name, "w")
    files, files_num = Get_file_list(image_path)
    index_count = 0
    count = 0
    for file in files:
        index_count = index_count + 1
        if count == files_num - 1:
            f.write(image_path + str(file))
            break
        if index_count >= 0:
            f.write(image_path + str(file) + "\n")
            count = count + 1
    f.close()
    print("2 Finish Generate_Txt: ", txt_name)
