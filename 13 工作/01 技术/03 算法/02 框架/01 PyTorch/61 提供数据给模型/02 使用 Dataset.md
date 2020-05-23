


# 使用 Dataset


自定义 Dataset:

- 从 `torch.utils.data.Dataset` 继承，并且重写：
  -  `__len__` 数据集的 len
  -  `__getitem__` ，支持索引，所以 `dataset[i]` 可以返回第 $i$ 个样本


在 `__init__` 加载 csv 文件，在 `__getitem__` 中读取并加载图片，这样图片会按需加载。

`__getitem__` 返回的的格式为： `{'image': image, 'landmarks': landmarks}`. 
数据集可选参数 `transform` 所以，任何对 sample 进行处理





软件包准备：

-  `scikit-image`: For image io and transforms <span style="color:red;">这个之前没有听说过。</span>
-  `pandas`: For easier csv parsing

数据准备：

- 要处理的数据：人脸数据，有对应的 68 点的 landmark 标记
- 数据下载：<https://download.pytorch.org/tutorial/faces.zip>
- 数据存放地址：'data/faces/'
- 数据格式：cvs







```python
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]  # 调整 landmark 坐标的时候，同时调整坐标轴，因为 图片在从 numpy 转为 pytorch 图片时也同样转坐标轴。
        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
                left: left + new_w]
        landmarks = landmarks - [left, top]
        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    # 将 ndarrays 转化为 Tensors
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # numpy 图片尺寸：height*width*channel
        # torch 图片尺寸：channel*height*width
        image = image.transpose((2, 0, 1))  # 交换坐标轴，将 numpy 图片转为 pytorch 格式
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}




class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

plt.ion()  # interactive mode


def show_landmarks(image, landmarks):
    # plt.cla() # which clears data but not axes
    # plt.clf() # which clears data and axes
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.01)  # pause a bit so that plots are updated

#
# fig = plt.figure()
# print(len(face_dataset))
#
# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#
#     print(sample['image'])
#     print(sample['landmarks'])
#     print(i, sample['image'].shape, sample['landmarks'].shape)
#
#     ax = plt.subplot(1, 4, i % 4 + 1)
#     plt.tight_layout()  # 这个是什么？
#     ax.clear()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_landmarks(**sample)
#
#     if i == 3:
#         plt.show()
#
#
#     def show_landmarks(image, landmarks):
#         # plt.cla() # which clears data but not axes
#         # plt.clf() # which clears data and axes
#         plt.imshow(image)
#         plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
#         plt.pause(0.01)  # pause a bit so that plots are updated


fig = plt.figure()
print(len(face_dataset))


scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# Apply each of the above transforms on sample.
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()
```

输出：

```
69
0 (324, 215, 3) (68, 2)
1 (500, 333, 3) (68, 2)
2 (250, 258, 3) (68, 2)
略..
66 (239, 209, 3) (68, 2)
67 (257, 169, 3) (68, 2)
68 (250, 175, 3) (68, 2)
```

显示的画面，是在动态变动的：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190628/SFytGXw4E01w.png?imageslim">
</p>
