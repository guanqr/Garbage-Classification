from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from config import DefaultConfig
from torchvision import transforms, datasets

class TrashSet():

    def __init__(self):
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.image_datasets = {x: datasets.ImageFolder(os.path.join(DefaultConfig.DataSetDir, x), self.data_transforms[x])
                               for x in ['train', 'val']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x],
                                                           batch_size=DefaultConfig.BatchSize,
                                                           shuffle=True,
                                                           num_workers=4)
                            for x in ['train', 'val']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes
        self.class_to_index = self.image_datasets['train'].class_to_idx
        self.class_num = len(self.class_to_index)
        self.index_to_class = [None]*self.class_num
        for key, v in self.class_to_index.items():
            self.index_to_class[v] = key

        self.class_train_image_count = np.zeros(self.class_num)
        self.class_val_image_count = np.zeros(self.class_num)
        for name in self.class_names:
            self.class_train_image_count[self.class_to_index[name]] = len(
                os.listdir(os.path.join(DefaultConfig.DataSetDir, 'train', name)))
            self.class_val_image_count[self.class_to_index[name]] = len(
                os.listdir(os.path.join(DefaultConfig.DataSetDir, 'val', name)))

        self.class_group = DefaultConfig.SetGroup

        self.group_index = dict()
        self.group_count = {'train': dict(), 'val': dict()}
        for k, v in self.class_group.items():
            index_list = []
            train_count = 0
            val_count = 0
            for name in v:
                index_list.append(self.class_to_index[name])
                train_count += len(os.listdir(os.path.join(DefaultConfig.DataSetDir, 'train', name)))
                val_count += len(os.listdir(os.path.join(DefaultConfig.DataSetDir, 'val', name)))
            self.group_index[k] = tuple(index_list)
            self.group_count['train'][k] = train_count
            self.group_count['val'][k] = val_count

        self.index_to_group = [None]*self.class_num
        for k, index_list in  self.group_index.items():
            for i in index_list:
                self.index_to_group[i] = k


    def save_params(self, path):
        params = {
            'class_num': self.class_num,
            'class_names': self.class_names,
            'class_to_index': self.class_to_index,
            'index_to_class': self.index_to_class,
            'class_group': self.class_group,
            'group_count': self.group_count,
            'index_to_group': self.index_to_group,
            'group_index': self.group_index
        }
        torch.save(params, path)




if __name__ == '__main__':
    from torchvision.datasets import ImageFolder
    from torchvision import transforms

    # TrashTrain = ImageFolder(root=(DefaultConfig.DataSetDir+'/Train'))
    TrashTrain = ImageFolder(root=DefaultConfig.DataSetDir)
    print(TrashTrain.classes)
    print(TrashTrain.class_to_idx)
    print(TrashTrain.imgs)
    pass
