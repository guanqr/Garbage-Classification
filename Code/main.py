import torchvision.models as models
from torchvision import transforms
from torch import nn
import torch
import cv2
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

from config import DefaultConfig
from models.mobilenetv3 import MobileNetV3_Small
from data.TrashSet import TrashSet


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def MyDevice(gpu=True):
    if gpu:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train_with_valid(trashSet):
    dataloaders = trashSet.dataloaders
    model = MobileNetV3_Small(trashSet.class_num)
    device = torch.device('cpu')
    #device = MyDevice(DefaultConfig.TrainWithGPU)
    ckpt = torch.load("models/mbv3_small.pth.tar",map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}, strict=False)
    model.to(device)

    for param in model.freeze_params():
        param.requires_grad = False
    for param in model.fine_tune_params():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fine_tune_params(), lr=DefaultConfig.LearningRate, weight_decay=2.5e-5)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DefaultConfig.LRScheduleStep, gamma=0.1)

    startEpoch = 0
    writer = SummaryWriter(DefaultConfig.LogDir)
    class_acc_dict = dict()
    group_acc_dict = dict()
    for epoch in range(startEpoch, DefaultConfig.EpochNum):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, DefaultConfig.EpochNum - 1))
        print('-' * 10)

        # Train phase
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        class_corrects = np.zeros(trashSet.class_num)
        group_corrects = dict()
        for k, v in trashSet.group_index.items():
            group_corrects[k] = 0

        for  i, data in enumerate(dataloaders['train']):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            for k, index_list in trashSet.group_index.items():
                group_labels = np.array([1 if x in index_list else 2 for x in labels])
                group_preds = np.array([1 if x in index_list else 3 for x in preds])
                group_corrects[k] += np.sum(group_preds==group_labels)

            for j in range(trashSet.class_num):
                class_labels = np.array([1 if x==j else 2 for x in labels])
                class_preds = np.array([1 if x==j else 3 for x in preds])
                class_corrects[j] += np.sum(class_preds==class_labels)

            print("epoch:", epoch, '/',DefaultConfig.EpochNum ,
                  'step loss', loss.item(),
                  'global_step=', epoch * len(dataloaders['train']) + i)

            if i % 100 == 99:
                writer.add_scalar('train/mini_batch_loss',
                                  loss.item(),
                                  global_step = epoch * len(dataloaders['train']) + i)


        exp_lr_scheduler.step()
        epoch_loss = running_loss / trashSet.dataset_sizes['train']
        epoch_acc = running_corrects.double() / trashSet.dataset_sizes['train']
        class_acc = class_corrects / trashSet.class_train_image_count
        for k, v in group_corrects.items():
            group_acc_dict[k] = group_corrects[k]/trashSet.group_count['train'][k]

        for j in range(trashSet.class_num):
            class_acc_dict[trashSet.index_to_class[j]] = class_acc[j]

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        writer.add_scalar('train/Epoch loss', epoch_loss, global_step=epoch)
        writer.add_scalar('train/Train Acc', epoch_acc, global_step=epoch)
        writer.add_scalars('train/Class Acc', class_acc_dict, global_step=epoch)
        writer.add_scalars('train/Group Acc', group_acc_dict, global_step=epoch)

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'epoch_loss': epoch_loss,
            'train_acc': epoch_acc,
            'optimizer_state_dict': optimizer.state_dict()
        }, DefaultConfig.LogDir+'/ckpt_'+str(epoch)+'.pth')
        print('Epoch',epoch, 'model saved.')

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_corrects = 0.0
        class_corrects = np.zeros(trashSet.class_num)
        for k, v in trashSet.group_index.items():
            group_corrects[k] = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                valid_corrects += torch.sum(preds == labels)

                for k, index_list in trashSet.group_index.items():
                    group_labels = np.array([1 if x in index_list else 2 for x in labels])
                    group_preds = np.array([1 if x in index_list else 3 for x in preds])
                    group_corrects[k] += np.sum(group_preds == group_labels)

                for j in range(trashSet.class_num):
                    class_labels = np.array([1 if x == j else 2 for x in labels])
                    class_preds = np.array([1 if x == j else 3 for x in preds])
                    class_corrects[j] += np.sum(class_preds == class_labels)

            epoch_valid_loss = valid_loss / trashSet.dataset_sizes['val']
            epoch_valid_acc = valid_corrects / trashSet.dataset_sizes['val']
            class_acc = class_corrects / trashSet.class_val_image_count
            for j in range(trashSet.class_num):
                class_acc_dict[trashSet.index_to_class[j]] = class_acc[j]

            for k, v in group_corrects.items():
                group_acc_dict[k] = group_corrects[k] / trashSet.group_count['val'][k]

            print('Valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_valid_loss, epoch_valid_acc))
            writer.add_scalar('valid/Epoch valid loss', epoch_valid_loss, global_step=epoch)
            writer.add_scalar('valid/Valid Acc', epoch_valid_acc, global_step=epoch)
            writer.add_scalars('valid/Class Acc', class_acc_dict, global_step=epoch)
            writer.add_scalars('valid/Group Acc', group_acc_dict, global_step=epoch)

        print('Epoch time cost:', time.time()-since)
        print()
    writer.close()


if __name__ == '__main__':
    trash_set = TrashSet()
    trash_set.save_params(DefaultConfig.DataSetInfoPath)
    train_with_valid(trash_set)