import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable as V
import cv2
import os
import math
import warnings
from tqdm import tqdm
import numpy as np
from time import time
from shutil import copyfile, move
from models.networks.GBETransNet import get_GBETransNet
from framework2 import MyFrame
from loss2.dice_bce_loss import Dice_bce_loss
from loss2.diceloss import DiceLoss
from metrics.iou import iou_pytorch
from evals import eval_func, eval_new
from data import ImageFolder
from inference2 import TTAFrame
from tensorboardX import SummaryWriter
from loss2.DetailLoss import DetailAggregateLoss


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

else:
    device = torch.device('cpu')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'


def train(Model=None):
    config_file = 'config.txt'
    dirs = []
    for line in open(config_file):
        dirs.append(line.split()[0])

    data_root = dirs[0]
    data_root = data_root.replace('\\', '/')
    pre_model = dirs[1]
    pre_model = pre_model.replace('\\', '/')
    bs_p_card = dirs[2]
    bs_p_card = bs_p_card.replace('\\', '/')
    lr = dirs[3]
    epoch_num = dirs[4]
    epoch_num = epoch_num.replace('\\', '/')
    model_name = dirs[5]
    model_name = model_name.replace('\\', '/')

    warnings.filterwarnings("ignore")

    BATCHSIZE_PER_CARD = int(bs_p_card)
    solver = MyFrame(Model, Dice_bce_loss,DetailAggregateLoss, float(lr))
    if pre_model.endswith('.th'):
        solver.load(pre_model)
    else:
        pass

    train_batchsize = BATCHSIZE_PER_CARD
    val_batchsize = BATCHSIZE_PER_CARD

    train_dataset = ImageFolder(data_root, mode='train')
    val_dataset = ImageFolder(data_root, mode='val')
    test_dataset = ImageFolder(data_root, mode='test')

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batchsize,
        shuffle=True,
        num_workers=0)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batchsize,
        shuffle=True,
        num_workers=0)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0)

    writer = SummaryWriter('./record')
    mylog = open('logs/' + model_name + '.log', 'w')
    tic = time()
    device = torch.device('cuda:0')
    no_optim = 0
    total_epoch = int(epoch_num)
    train_epoch_best_loss = 100.
    val_epoch_best_loss = 100.
    val_best_iou = 0.3
    # criteon = nn.CrossEntropyLoss().to(device)
    criteon = DiceLoss()
    # iou_criteon = SoftIoULoss(2)
    scheduler = solver.lr_strategy()

    for epoch in range(1, total_epoch + 1):
        print('---------- Epoch:' + str(epoch) + ' ----------')
        # data_loader_iter = iter(data_loader)
        data_loader_iter = data_loader
        train_epoch_loss = 0
        print('Train:')
        for img, mask in tqdm(data_loader_iter, ncols=20, total=len(data_loader_iter)):
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)

        val_data_loader_num = iter(val_data_loader)
        test_epoch_loss = 0
        test_mean_iou = 0
        val_pre_list = []
        val_mask_list = []
        print('Validation:')
        for val_img, val_mask in tqdm(val_data_loader_num, ncols=20, total=len(val_data_loader_num)):
            val_img, val_mask = val_img.to(device), val_mask.cpu()
            val_mask[np.where(val_mask > 0)] = 1
            val_mask = val_mask.squeeze(0)
            predict = solver.test_one_img(val_img)

            predict_temp = torch.from_numpy(predict).unsqueeze(0)
            predict_use = V(predict_temp.type(torch.FloatTensor), volatile=True)
            val_use = V(val_mask.type(torch.FloatTensor), volatile=True)

            test_epoch_loss += criteon.forward(predict_use, val_use)

            predict_use = predict_use.squeeze(0)
            predict_use = predict_use.unsqueeze(1)
            predict_use[predict_use >= 0.5] = 1
            predict_use[predict_use < 0.5] = 0
            predict_use = predict_use.type(torch.LongTensor)
            val_use = val_use.squeeze(1).type(torch.LongTensor)
            test_mean_iou += iou_pytorch(predict_use, val_use)

        batch_iou = test_mean_iou / len(val_data_loader_num)
        val_loss = test_epoch_loss / len(val_data_loader_num)

        writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        writer.add_scalar('train_loss', train_epoch_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('iou', batch_iou, epoch)
        mylog.write('********** ' + 'lr={:.10f}'.format(scheduler.get_lr()[0]) + ' **********' + '\n')
        mylog.write('--epoch:' + str(epoch) + '  --time:' + str(int(time() - tic)) + '  --train_loss:' + str(
            train_epoch_loss) + ' --val_loss:' + str(val_loss.item()) + ' --val_iou:' + str(batch_iou.item()) + '\n')
        print('--epoch:', epoch, '  --time:', int(time() - tic), '  --train_loss:', train_epoch_loss, ' --val_loss:',
              val_loss.item(), ' --val_iou:', batch_iou.item())
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('weights/' + model_name + '_train_loss_best.th')
        if batch_iou >= val_best_iou:
            val_best_iou = batch_iou
            solver.save('weights/' + model_name + '_iou_best.th')
        if val_loss <= val_epoch_best_loss:
            val_epoch_best_loss = val_loss
            solver.save('weights/' + model_name + '_val_loss_best.th')

        if no_optim > 10:
            if solver.old_lr < 5e-8:
                break
            solver.load('weights/' + model_name + '_train_loss_best.th')
            no_optim = 0

        scheduler.step()
        print('lr={:.10f}'.format(scheduler.get_lr()[0]))

        mylog.flush()

    # writer.add_graph(Model(), img)
    print('Train Finish !')
    mylog.close()

    # evaluation
    # model_path = './weights/'+model_name + '_iou_best.th'
    model_path = './weights/' + model_name + '_train_loss_best.th'
    solver = TTAFrame(Model)
    solver.load(model_path)

    label_list = []
    pre_list = []
    for img, mask in tqdm(test_data_loader, ncols=20, total=len(test_data_loader)):
        mask[mask > 0] = 1
        mask = torch.squeeze(mask)
        mask = mask.numpy()
        mask = mask.astype(np.int)
        label_list.append(mask)

        img = torch.squeeze(img)
        img = img.numpy()
        pre = solver.test_one_img_from_path_8(img)
        pre[pre >= 4.0] = 255
        pre[pre < 4.0] = 0

        pre = pre.astype(np.int)
        pre[pre > 0] = 1
        pre_list.append(pre)

    eval_new(label_list, pre_list)


if __name__ == '__main__':
    net = get_GBETransNet(1)
    # img = torch.randn((2, 3, 256, 256))
    # new = net(img)
    # print(new)
    train(net)
