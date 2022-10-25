'''Train CK+ with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from modules import transforms
import numpy as np
import os
from modules import utils
from modules import dataset
from torch.autograd import Variable
from modules import vgg
import h5py

if __name__ == '__main__':

  h5_data = 'vgg19-face-expression/data/CK_data.h5'
  output_path = "vgg19-face-expression/model"

  data = h5py.File(h5_data, 'r', driver='windows')
  use_cuda = torch.cuda.is_available()

  best_Test_acc = 0  # best PrivateTest accuracy
  best_Test_acc_epoch = 0
  start_epoch = 0  # start from epoch 0 or last checkpoint epoch

  learning_rate_decay_start = 20  # 50
  learning_rate_decay_every = 1 # 5
  learning_rate_decay_rate = 0.8 # 0.9

  cut_size = 44
  total_epoch = 60

  path = os.path.join(output_path)

  # Data
  print('==> Preparing data..')
  transform_train = transforms.Compose([
      transforms.RandomCrop(cut_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
  ])

  transform_test = transforms.Compose([
      transforms.TenCrop(cut_size),
      transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
  ])

  trainset = dataset.CK(split = 'Training', fold = 1, transform=transform_train, data=data)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
  testset = dataset.CK(split = 'Testing', fold = 1, transform=transform_test, data=data)
  testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=0)

  # Model

  net = vgg.VGG('VGG19')

  if use_cuda:
      net.cuda()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

  # Training
  def train(epoch):
      print('\nEpoch: %d' % epoch)
      global Train_acc
      net.train()
      train_loss = 0
      correct = 0
      total = 0

      if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
          frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
          decay_factor = learning_rate_decay_rate ** frac
          current_lr = 0.01 * decay_factor
          utils.set_lr(optimizer, current_lr)  # set the decayed rate
      else:
          current_lr = 0.01
      print('learning_rate: %s' % str(current_lr))


      for batch_idx, (inputs, targets) in enumerate(trainloader):
          if use_cuda:
              inputs, targets = inputs.cuda(), targets.cuda()
          optimizer.zero_grad()
          inputs, targets = Variable(inputs), Variable(targets)
          outputs = net(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          utils.clip_gradient(optimizer, 0.1)
          optimizer.step()

          train_loss += loss.data.item()
          _, predicted = torch.max(outputs.data, 1)
          total += targets.size(0)
          correct += predicted.eq(targets.data).cpu().sum()

          utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

      Train_acc = 100.*correct/total

  def test(epoch):
      global Test_acc
      global best_Test_acc
      global best_Test_acc_epoch
      net.eval()
      PrivateTest_loss = 0
      correct = 0
      total = 0
      for batch_idx, (inputs, targets) in enumerate(testloader):
          bs, ncrops, c, h, w = np.shape(inputs)
          inputs = inputs.view(-1, c, h, w)

          if use_cuda:
              inputs, targets = inputs.cuda(), targets.cuda()
          inputs, targets = Variable(inputs, volatile=True), Variable(targets)
          outputs = net(inputs)
          outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

          loss = criterion(outputs_avg, targets)
          PrivateTest_loss += loss.data.item()
          _, predicted = torch.max(outputs_avg.data, 1)
          total += targets.size(0)
          correct += predicted.eq(targets.data).cpu().sum()
          utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
      # Save checkpoint.
      Test_acc = 100.*correct/total

      if Test_acc > best_Test_acc:
          print('Saving..')
          print("best_Test_acc: %0.3f" % Test_acc)
          state = {'net': net.state_dict() if use_cuda else net,
              'best_Test_acc': Test_acc,
              'best_Test_acc_epoch': epoch,
          }
          if not os.path.isdir(output_path):
              os.mkdir(output_path)
          if not os.path.isdir(path):
              os.mkdir(path)
          torch.save(state, os.path.join(path, 'VGG_Model.t7'))
          best_Test_acc = Test_acc
          best_Test_acc_epoch = epoch

  for epoch in range(start_epoch, total_epoch):
      train(epoch)
      test(epoch)

  print("best_Test_acc: %0.3f" % best_Test_acc)
  print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)