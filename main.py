# coding=utf-8
import random
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.nn.functional as F
from torchvision import datasets, transforms
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.activation_based import functional
from models.cnn import CNN
from models.snn import CSNN
import copy
import logging
logger = logging.getLogger(__name__)

def set_random_seed(seed_value=43):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


def get_args():
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--seed', default=43, type=int, help='random seed')
    parser.add_argument('--batch_size', default=16, type=int, help='training and testing batchsize')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epoch', default=80, type=int, help='epoch num')
    parser.add_argument('--n_clients', default=1, type=int, help='num of clients of each model')
    parser.add_argument('--rounds', default=1, type=int, help='communicate every n epochs')
    parser.add_argument('--id', default=0, type=int, help='id of exp')
    args = parser.parse_args()
    return args


def train_cnn(device, epoch, trainloader, model, optimizer):
    model.train()

    num_correct = 0
    all_loss = 0
    all_num = 0

    for batch_index, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, dim=1)
        batch_correct = (predicted == labels).sum().item()
        batch_accuracy = batch_correct / images.shape[0]
        # logger.warning("epoch: {}, batch_id: {}, batch loss: {}, batch acc:{}".format(epoch, batch_index, loss.item(), batch_accuracy))

        # 每一个epoch预测对的总个数
        num_correct += (predicted == labels).sum().item()
        all_loss += loss.item()
        all_num += images.shape[0]

    # each epoch avg-acc and avg-loss
    epoch_accuracy = num_correct / all_num
    epoch_loss = all_loss / all_num
    logger.warning("Train Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, epoch_loss, epoch_accuracy))
    return model, epoch_accuracy, epoch_loss, optimizer


def test_cnn(device, epoch, testloader, model):
    model.eval()

    num_correct = 0
    all_loss = 0
    all_num = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)

            all_loss += F.cross_entropy(output, labels).item()  # item()用于取出tensor里边的值
            _, predicted = torch.max(output, dim=1)
            num_correct += (predicted == labels).sum().item()
            all_num += images.shape[0]

        epoch_loss = all_loss / all_num
        epoch_accuracy = num_correct / all_num
        logger.warning("Test epoch: {}, Avg loss: {}, Avg acc: {}".format(epoch, epoch_loss, epoch_accuracy))
    return model, epoch_accuracy, epoch_loss


def train_snn(device, epoch, trainloader, model, optimizer):
    model.train()

    num_correct = 0
    all_loss = 0
    all_num = 0

    for batch_index, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        images = images.permute(1, 0, 2, 3, 4)
        labels = labels.to(device)
        label_onehot = F.one_hot(labels, 10).float()

        # forward
        outputs = model(images)
        loss = F.mse_loss(outputs, label_onehot)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, dim=1)
        batch_correct = (predicted == labels).sum().item()
        batch_accuracy = batch_correct / images.shape[1]
        if batch_index % 100 == 0:
            logger.warning("epoch: {}, batch_id: {}, batch loss: {}, batch acc:{}".format(epoch, batch_index, loss.item(),
                                                                                 batch_accuracy))

        # 每一个epoch预测对的总个数
        num_correct += (predicted == labels).sum().item()
        all_loss += loss.item()
        all_num += images.shape[1]
        functional.reset_net(model)
    #
    # each epoch avg-acc and avg-loss
    epoch_accuracy = num_correct / all_num
    epoch_loss = all_loss / all_num
    logger.warning("Train Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, epoch_loss, epoch_accuracy))
    return model, epoch_accuracy, epoch_loss, optimizer


def test_snn(device, epoch, testloader, model):
    model.eval()

    num_correct = 0
    all_loss = 0
    all_num = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(testloader):
            images = images.to(device)
            images = images.permute(1, 0, 2, 3, 4)
            labels = labels.to(device)
            label_onehot = F.one_hot(labels, 10).float()

            outputs = model(images)
            loss = F.mse_loss(outputs, label_onehot)

            all_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            num_correct += (predicted == labels).sum().item()
            all_num += images.shape[1]

            functional.reset_net(model)

        epoch_loss = all_loss / all_num
        epoch_accuracy = num_correct / all_num
        logger.warning("Test epoch: {}, Avg loss: {}, Avg acc: {}".format(epoch, epoch_loss, epoch_accuracy))
    return model, epoch_accuracy, epoch_loss


def train_cnn_snn(device, epoch, cnn_trainloader, snn_trainloader, model_cnn, model_snn, optimizer_cnn, optimizer_snn, rounds):
    model_cnn.train()
    model_snn.train()

    num_correct_cnn = 0
    num_correct_snn = 0
    all_loss_cnn = 0
    all_loss_snn = 0
    all_num_cnn = 0
    all_num_snn = 0

    for batch_index, (data_cnn, data_snn) in enumerate(zip(cnn_trainloader, snn_trainloader)):

        images_cnn, labels_cnn = data_cnn
        images_cnn = images_cnn.to(device)
        labels_cnn = labels_cnn.to(device)

        images_snn, labels_snn = data_snn
        images_snn = images_snn.to(device)
        labels_snn = labels_snn.to(device)
        images_snn = images_snn.permute(1, 0, 2, 3, 4)
        label_onehot = F.one_hot(labels_snn, 10).float()

        # forward cnn
        outputs_cnn = model_cnn(images_cnn)
        loss_cnn = F.cross_entropy(outputs_cnn, labels_cnn)

        # forward snn
        outputs_snn = model_snn(images_snn)
        loss_snn = F.mse_loss(outputs_snn, label_onehot)

        # cal grad
        optimizer_cnn.zero_grad()
        optimizer_snn.zero_grad()
        loss_cnn.backward()
        loss_snn.backward()

        if epoch % rounds == 0:
            # avg grad
            average_grads = [(p1.grad + p2.grad) / 2 for p1, p2 in zip(model_cnn.parameters(), model_snn.parameters())]

            # update grad
            for param_cnn, param_snn, avg_grad in zip(model_cnn.parameters(), model_snn.parameters(), average_grads):
                param_cnn.grad = avg_grad
                param_snn.grad = avg_grad
            
        optimizer_cnn.step()
        optimizer_snn.step()

        # statistic for cnn
        _, predicted_cnn = torch.max(outputs_cnn, dim=1)
        batch_correct_cnn = (predicted_cnn == labels_cnn).sum().item()
        batch_accuracy_cnn = batch_correct_cnn / images_cnn.shape[0]

        # 每一个epoch预测对的总个数
        num_correct_cnn += (predicted_cnn == labels_cnn).sum().item()
        all_loss_cnn += loss_cnn.item()
        all_num_cnn += images_cnn.shape[0]

        # statistic for snn
        _, predicted_snn = torch.max(outputs_snn, dim=1)
        batch_correct_snn = (predicted_snn == labels_snn).sum().item()
        batch_accuracy_snn = batch_correct_snn / images_snn.shape[1]

        if batch_index % 100 == 0:
            logger.warning(
                "epoch: {}, batch_id: {}, batch loss snn: {}, batch acc snn:{}".format(epoch, batch_index, loss_snn.item(),
                                                                                       batch_accuracy_snn))
            logger.warning(
                "epoch: {}, batch_id: {}, batch loss cnn: {}, batch acc cnn:{}".format(epoch, batch_index, loss_cnn.item(),
                                                                                       batch_accuracy_cnn))

        # 每一个epoch预测对的总个数
        num_correct_snn += (predicted_snn == labels_snn).sum().item()
        all_loss_snn += loss_snn.item()
        all_num_snn += images_snn.shape[1]
        functional.reset_net(model_snn)
    #
    # each epoch avg-acc and avg-loss
    epoch_accuracy_cnn = num_correct_cnn / all_num_cnn
    epoch_loss_cnn = all_loss_cnn / all_num_cnn
    epoch_accuracy_snn = num_correct_snn / all_num_snn
    epoch_loss_snn = all_loss_snn / all_num_snn
    logger.warning("Train Epoch: {}, Loss cnn: {}, Accuracy cnn: {}".format(epoch, epoch_loss_cnn, epoch_accuracy_cnn))
    logger.warning("Train Epoch: {}, Loss snn: {}, Accuracy snn: {}".format(epoch, epoch_loss_snn, epoch_accuracy_snn))
    return model_cnn, model_snn, optimizer_cnn, optimizer_snn, epoch_accuracy_cnn, epoch_accuracy_snn, epoch_loss_cnn, epoch_loss_cnn


def train_cnn_snn_2(device, epoch, cnn_trainset, snn_trainset, model_cnn, model_snn, optimizer_cnn, optimizer_snn, rounds, n_clients):
    cnn_trainloaders = []
    snn_trainloaders = []
    
    len_dataset_cnn = len(cnn_trainset)
    len_dataset_snn = len(snn_trainset)
    torch.manual_seed(0)  # 为了可复现性设置随机种子

    # 计算每份的大小
    size_per_subset_cnn = len_dataset_cnn // n_clients
    size_per_subset_snn = len_dataset_snn // n_clients

    # 随机打乱
    indices = list(range(len_dataset_cnn))
    random.shuffle(indices)

    # 将数据集分成 n 份
    subsets_cnn = [Subset(cnn_trainset, indices[i * size_per_subset_cnn:(i + 1) * size_per_subset_cnn])
            for i in range(n_clients)]
    for i in range(n_clients):
        cnnloader = DataLoader(dataset=subsets_cnn[i], batch_size=args.batch_size, shuffle=True, drop_last=False,
                                        num_workers=0)
        cnn_trainloaders.append(cnnloader)
    
    
    # 随机打乱
    indices = list(range(len_dataset_snn))
    random.shuffle(indices)

    # 将数据集分成 n 份
    subsets_snn = [Subset(snn_trainset, indices[i * size_per_subset_snn:(i + 1) * size_per_subset_snn])
            for i in range(n_clients)]
    for i in range(n_clients):
        snnloader = DataLoader(dataset=subsets_snn[i], batch_size=args.batch_size, shuffle=True, drop_last=False,
                                        num_workers=0)
        snn_trainloaders.append(snnloader)

    # 将模型和优化器扩展为列表
    model_cnns = [copy.deepcopy(model_cnn) for _ in range(n_clients)]
    model_snns = [copy.deepcopy(model_snn) for _ in range(n_clients)]
    optimizer_cnns = [copy.deepcopy(optimizer_cnn) for _ in range(n_clients)]
    optimizer_snns = [copy.deepcopy(optimizer_snn) for _ in range(n_clients)]

    num_correct_cnns = [0] * n_clients
    num_correct_snns = [0] * n_clients
    all_loss_cnns = [0] * n_clients
    all_loss_snns = [0] * n_clients
    all_num_cnns = [0] * n_clients
    all_num_snns = [0] * n_clients

    for client_id in range(n_clients):
        cnn_trainloader = cnn_trainloaders[client_id]
        snn_trainloader = snn_trainloaders[client_id]
        model_cnn = model_cnns[client_id]
        model_snn = model_snns[client_id]
        optimizer_cnn = optimizer_cnns[client_id]
        optimizer_snn = optimizer_snns[client_id]

        model_cnn.train()
        model_snn.train()

        for batch_index, (data_cnn, data_snn) in enumerate(zip(cnn_trainloader, snn_trainloader)):
            images_cnn, labels_cnn = data_cnn
            images_cnn = images_cnn.to(device)
            labels_cnn = labels_cnn.to(device)

            images_snn, labels_snn = data_snn
            images_snn = images_snn.to(device)
            labels_snn = labels_snn.to(device)
            images_snn = images_snn.permute(1, 0, 2, 3, 4)
            label_onehot = F.one_hot(labels_snn, 10).float()

            # forward cnn
            outputs_cnn = model_cnn(images_cnn)
            loss_cnn = F.cross_entropy(outputs_cnn, labels_cnn)

            # forward snn
            outputs_snn = model_snn(images_snn)
            loss_snn = F.mse_loss(outputs_snn, label_onehot)

            # cal grad
            optimizer_cnn.zero_grad()
            optimizer_snn.zero_grad()
            loss_cnn.backward()
            loss_snn.backward()

            if epoch % rounds == 0:
                # avg grad
                average_grads_cnn = []
                average_grads_snn = []
                for param_cnns, param_snns in zip(model_cnns, model_snns):
                    grads_cnn = [p.grad for p in param_cnns.parameters() if p.grad is not None]
                    grads_snn = [p.grad for p in param_snns.parameters() if p.grad is not None]
                    average_grads_cnn.append(grads_cnn)
                    average_grads_snn.append(grads_snn)

                avg_grads_cnn = [torch.stack(grads).mean(0) for grads in zip(*average_grads_cnn)]
                avg_grads_snn = [torch.stack(grads).mean(0) for grads in zip(*average_grads_snn)]

                for param_cnn, param_snn, avg_grad_cnn, avg_grad_snn in zip(model_cnn.parameters(), model_snn.parameters(), avg_grads_cnn, avg_grads_snn):
                    param_cnn.grad = avg_grad_cnn
                    param_snn.grad = avg_grad_snn

            optimizer_cnn.step()
            optimizer_snn.step()

            # statistic for cnn
            _, predicted_cnn = torch.max(outputs_cnn, dim=1)
            batch_correct_cnn = (predicted_cnn == labels_cnn).sum().item()
            batch_accuracy_cnn = batch_correct_cnn / images_cnn.shape[0]
            num_correct_cnns[client_id] += (predicted_cnn == labels_cnn).sum().item()
            all_loss_cnns[client_id] += loss_cnn.item()
            all_num_cnns[client_id] += images_cnn.shape[0]

            # statistic for snn
            _, predicted_snn = torch.max(outputs_snn, dim=1)
            batch_correct_snn = (predicted_snn == labels_snn).sum().item()
            batch_accuracy_snn = batch_correct_snn / images_snn.shape[1]
            if batch_index % 100 == 0:
                logger.warning("epoch: {}, batch_id: {}, batch loss snn: {}, batch acc snn:{}".format(epoch, batch_index, loss_snn.item(), batch_accuracy_snn))
                logger.warning("epoch: {}, batch_id: {}, batch loss cnn: {}, batch acc cnn:{}".format(epoch, batch_index, loss_cnn.item(), batch_accuracy_cnn))
            num_correct_snns[client_id] += (predicted_snn == labels_snn).sum().item()
            all_loss_snns[client_id] += loss_snn.item()
            all_num_snns[client_id] += images_snn.shape[1]

            functional.reset_net(model_snn)

    # each epoch avg-acc and avg-loss
    epoch_accuracy_cnns = [num_correct_cnns[i] / all_num_cnns[i] for i in range(n_clients)]
    epoch_loss_cnns = [all_loss_cnns[i] / all_num_cnns[i] for i in range(n_clients)]
    epoch_accuracy_snns = [num_correct_snns[i] / all_num_snns[i] for i in range(n_clients)]
    epoch_loss_snns = [all_loss_snns[i] / all_num_snns[i] for i in range(n_clients)]
    
    for client_id in range(n_clients):
        logger.warning("Train Epoch: {}, Client: {}, Loss cnn: {}, Accuracy cnn: {}".format(epoch, client_id, epoch_loss_cnns[client_id], epoch_accuracy_cnns[client_id]))
        logger.warning("Train Epoch: {}, Client: {}, Loss snn: {}, Accuracy snn: {}".format(epoch, client_id, epoch_loss_snns[client_id], epoch_accuracy_snns[client_id]))

    return model_cnns, model_snns, optimizer_cnns, optimizer_snns, epoch_accuracy_cnns, epoch_accuracy_snns, epoch_loss_cnns, epoch_loss_snns


def train_fedprox(device, epoch, cnn_trainloader, snn_trainloader, model_cnn, model_snn, optimizer_cnn, optimizer_snn, mu=0.01):
    model_cnn.train()
    model_snn.train()
    num_correct_cnn = 0
    num_correct_snn = 0
    all_loss_cnn = 0
    all_loss_snn = 0
    all_num_cnn = 0
    all_num_snn = 0
    
    # 保存全局模型参数
    global_cnn_weights = copy.deepcopy(model_cnn.state_dict())
    global_snn_weights = copy.deepcopy(model_snn.state_dict())

    for batch_index, (data_cnn, data_snn) in enumerate(zip(cnn_trainloader, snn_trainloader)):
        images_cnn, labels_cnn = data_cnn
        images_cnn = images_cnn.to(device)
        labels_cnn = labels_cnn.to(device)
        images_snn, labels_snn = data_snn
        images_snn = images_snn.to(device)
        labels_snn = labels_snn.to(device)
        images_snn = images_snn.permute(1, 0, 2, 3, 4)
        label_onehot = F.one_hot(labels_snn, 10).float()

        # 计算FedProx正则化项
        # fed_prox_reg_cnn = 0
        # fed_prox_reg_snn = 0
        # for param_cnn, global_param_cnn in zip(model_cnn.parameters(), global_cnn_weights.values()):
        #     fed_prox_reg_cnn += ((param_cnn - global_param_cnn) ** 2).sum()
        # for param_snn, global_param_snn in zip(model_snn.parameters(), global_snn_weights.values()):
        #     fed_prox_reg_snn += ((param_snn - global_param_snn) ** 2).sum()
            
        fed_prox_reg_cnn = 0
        fed_prox_reg_snn = 0
        for param_cnn, global_param_cnn in zip(model_cnn.parameters(), global_cnn_weights.values()):
            if param_cnn.shape == global_param_cnn.shape:
                fed_prox_reg_cnn += ((param_cnn - global_param_cnn) ** 2).sum()
        for param_snn, global_param_snn in zip(model_snn.parameters(), global_snn_weights.values()):
            if param_snn.shape == global_param_snn.shape:
                fed_prox_reg_snn += ((param_snn - global_param_snn) ** 2).sum()

        # forward cnn
        outputs_cnn = model_cnn(images_cnn)
        loss_cnn = F.cross_entropy(outputs_cnn, labels_cnn) + mu / 2 * fed_prox_reg_cnn

        # forward snn
        outputs_snn = model_snn(images_snn)
        loss_snn = F.mse_loss(outputs_snn, label_onehot) + mu / 2 * fed_prox_reg_snn

        # cal grad
        optimizer_cnn.zero_grad()
        optimizer_snn.zero_grad()
        loss_cnn.backward()
        loss_snn.backward()

        # update grad
        optimizer_cnn.step()
        optimizer_snn.step()

        # statistic for cnn
        _, predicted_cnn = torch.max(outputs_cnn, dim=1)
        batch_correct_cnn = (predicted_cnn == labels_cnn).sum().item()
        batch_accuracy_cnn = batch_correct_cnn / images_cnn.shape[0]
        num_correct_cnn += (predicted_cnn == labels_cnn).sum().item()
        all_loss_cnn += loss_cnn.item()
        all_num_cnn += images_cnn.shape[0]

        # statistic for snn
        _, predicted_snn = torch.max(outputs_snn, dim=1)
        batch_correct_snn = (predicted_snn == labels_snn).sum().item()
        batch_accuracy_snn = batch_correct_snn / images_snn.shape[1]
        if batch_index % 100 == 0:
            logger.warning("epoch: {}, batch_id: {}, batch loss snn: {}, batch acc snn:{}".format(epoch, batch_index, loss_snn.item(), batch_accuracy_snn))
            logger.warning("epoch: {}, batch_id: {}, batch loss cnn: {}, batch acc cnn:{}".format(epoch, batch_index, loss_cnn.item(), batch_accuracy_cnn))
        num_correct_snn += (predicted_snn == labels_snn).sum().item()
        all_loss_snn += loss_snn.item()
        all_num_snn += images_snn.shape[1]

        functional.reset_net(model_snn)

    # each epoch avg-acc and avg-loss
    epoch_accuracy_cnn = num_correct_cnn / all_num_cnn
    epoch_loss_cnn = all_loss_cnn / all_num_cnn
    epoch_accuracy_snn = num_correct_snn / all_num_snn
    epoch_loss_snn = all_loss_snn / all_num_snn

    logger.warning("Train Epoch: {}, Loss cnn: {}, Accuracy cnn: {}".format(epoch, epoch_loss_cnn, epoch_accuracy_cnn))
    logger.warning("Train Epoch: {}, Loss snn: {}, Accuracy snn: {}".format(epoch, epoch_loss_snn, epoch_accuracy_snn))

    return model_cnn, model_snn, optimizer_cnn, optimizer_snn, epoch_accuracy_cnn, epoch_accuracy_snn, epoch_loss_cnn, epoch_loss_cnn


if __name__ == '__main__':
    # get args of the training and testing stages
    args = get_args()

    # set random seed
    set_random_seed(args.seed)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # get NMNIST dataset
    nmnist_train = NMNIST(root='./datasets/N-MNIST', train=True, data_type='frame', frames_number=20, split_by='number')
    nmnist_test = NMNIST(root='./datasets/N-MNIST', train=False, data_type='frame', frames_number=20, split_by='number')
    nmnist_train_dataloader = DataLoader(dataset=nmnist_train, batch_size=args.batch_size, shuffle=True,
                                         drop_last=False, num_workers=0)
    nmnist_test_dataloader = DataLoader(dataset=nmnist_test, batch_size=args.batch_size, shuffle=False,         drop_last=False, num_workers=0)

    # get MNIST dataset
    data_tf = transforms.Compose(
        [transforms.Resize((34, 34)),
         transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='./datasets', train=True, transform=data_tf, download=True)
    mnist_test = datasets.MNIST(root='./datasets', train=False, transform=data_tf)
    mnist_train_dataloader = DataLoader(dataset=mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                        num_workers=0)
    mnist_test_dataloader = DataLoader(dataset=mnist_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                       num_workers=0)

    # create models
    channel = 32
    # cnn
    cnn_model = CNN(channels=channel)
    cnn_model = cnn_model.to(device)
    # snn
    snn_model = CSNN(T=20, channels=channel)
    snn_model = snn_model.to(device)

    # set optim
    optimizer_c = torch.optim.Adam(cnn_model.parameters(), lr=args.lr)
    optimizer_s = torch.optim.Adam(snn_model.parameters(), lr=args.lr)

    # # train snn
    # for epoch in range(args.epoch):
    #     snn_model, train_acc, train_loss, optimizer_s = train_snn(device, epoch, nmnist_train_dataloader, snn_model, optimizer_s)
    #     snn_model, test_acc, test_loss = test_snn(device, epoch, nmnist_test_dataloader, snn_model)
    #
    # # train cnn
    # for epoch in range(args.epoch):
    #     cnn_model, train_acc, train_loss, optimizer_c = train_cnn(device, epoch, mnist_train_dataloader, cnn_model, optimizer_c)
    #     cnn_model, test_acc, test_loss = test_cnn(device, epoch, mnist_test_dataloader, cnn_model)

    # train cnn + snn
    best_acc_cnn = 0
    best_acc_snn = 0
    for epoch in range(args.epoch):
        if args.n_clients > 1:
            cnn_models, snn_models, \
                optimizer_cs, optimizer_ss, \
                epoch_accuracy_cnns, epoch_accuracy_snns, \
                epoch_loss_cnns, epoch_loss_cnns = \
                train_cnn_snn_2(device,
                            epoch,
                            mnist_train,
                            nmnist_train,
                            cnn_model,
                            snn_model,
                            optimizer_c,
                            optimizer_s, args.rounds, args.n_clients)
        elif args.n_clients == 1:
            cnn_model, snn_model, \
                optimizer_c, optimizer_s, \
                epoch_accuracy_cnn, epoch_accuracy_snn, \
                epoch_loss_cnn, epoch_loss_cnn = \
                train_cnn_snn(device,
                            epoch,
                            mnist_train_dataloader,
                            nmnist_train_dataloader,
                            cnn_model,
                            snn_model,
                            optimizer_c,
                            optimizer_s, args.rounds)
            # train_fedprox(device,
            #               epoch,
            #               mnist_train_dataloader,
            #               nmnist_train_dataloader,
            #               cnn_model,
            #               snn_model,
            #               optimizer_c,
            #               optimizer_s)
        else:
            logger.warning("Invalid n_clients")

        if args.n_clients == 1:
            cnn_model, test_acc_cnn, test_loss_cnn = test_cnn(device, epoch, mnist_test_dataloader, cnn_model)
            if test_acc_cnn > best_acc_cnn:
                best_acc_cnn = test_acc_cnn
            snn_model, test_acc_snn, test_loss_snn = test_snn(device, epoch, nmnist_test_dataloader, snn_model)
            if test_acc_snn > best_acc_snn:
                best_acc_snn = test_acc_snn
        else:
            for cnn_model in cnn_models:
                cnn_model, test_acc_cnn, test_loss_cnn = test_cnn(device, epoch, mnist_test_dataloader, cnn_model)
                if test_acc_cnn > best_acc_cnn:
                    best_acc_cnn = test_acc_cnn
            for cnn_model in snn_models:
                snn_model, test_acc_snn, test_loss_snn = test_snn(device, epoch, nmnist_test_dataloader, snn_model)
                if test_acc_snn > best_acc_snn:
                    best_acc_snn = test_acc_snn
    print("best cnn acc:{}".format(best_acc_cnn))
    print("best snn acc:{}".format(best_acc_snn))
