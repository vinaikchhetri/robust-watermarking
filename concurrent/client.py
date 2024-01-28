from torch.utils.data import Dataset
import torch
import models
import torch.optim as optim
from torchvision.models import resnet18
from utils import accuracy
import utils
import PIL
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, dataset, idxs, trx_list, try_list):
        self.dataset = dataset
        self.idxs = idxs
        self.trx_list, self.try_list = trx_list, try_list
        self.X = torch.cat([self.dataset[0][self.idxs], self.trx_list])
        self.Y = torch.cat([self.dataset[1][self.idxs], self.try_list])
        # print("Y",self.Y[0:len(self.dataset[0][self.idxs])])
        #print(self.Y)
        #print(self.Y.shape)

    def __len__(self):
        return len(self.idxs)+len(self.trx_list)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.Y[idx]

        return img, label

class Client():
    def __init__(self, data_client, trainset, trx_list, try_list, args, device, is_adversary):
        self.data_client = data_client
        self.args = args
        self.device = device
        self.trainset = trainset
        self.is_adversary = is_adversary
        self.loss = None
        self.trx_list, self.try_list =  trx_list, try_list
        train_y_list = self.try_list
        if self.is_adversary == 1:
            if self.args.target == "random":
                train_y_list = torch.randint(0,args.num_classes,(len(try_list),))
            else:
                train_y_list = torch.ones(len(try_list))
                train_y_list = train_y_list.type(torch. int64)
                train_y_list = train_y_list*3

        
        self.cd = CustomDataset(self.trainset, self.data_client, self.trx_list, train_y_list)
        # self.bcd = utils.CustomDataset(self.trx_list, self.try_list)
        if args.B == 8:
            self.bs = len(trainset)
        else:
            self.bs = args.B
        self.data_loader = torch.utils.data.DataLoader(self.cd, batch_size=self.bs,
                                                shuffle=True)
        # self.bcd_loader = torch.utils.data.DataLoader(self.bcd, batch_size=self.bs,
        #                                         shuffle=True)
        if args.gpu == "gpu":
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        if args.model == 'nn':
            self.model_local = models.MP(28*28,200,10)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
         
        if args.model == 'cnn':
            self.model_local = models.CNN_MNIST()
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
            self.optimizer = optim.SGD(self.model_local.parameters(), lr=0.1, momentum=0.5)
        
        if args.model == 'resnet':
            #self.model_local = resnet18(num_classes=10)
            self.model_local = models.ResNet(18, num_classes = self.args.num_classes)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
            self.optimizer = optim.SGD(self.model_local.parameters(), lr=0.01, momentum=0.5)
        
    def load_model(self, model_global):
        self.model_local.load_state_dict(model_global.state_dict())

    def client_update(self):
        # print(len(self.cd))
        # print(len(self.bcd))
        self.model_local.to(self.device)
        self.model_local.train()
        self.optimizer = optim.SGD(self.model_local.parameters(), lr=self.args.lr, momentum=0.9)

        for epoch in range(self.args.E):
            self.model_local.train()
            running_loss = 0.0
            running_acc = 0.0
            for index,data in enumerate(self.data_loader):
                
                inputs, labels = data
                # if self.is_adversary==1:
                #     print(labels)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                # for param in client.model_local.parameters():
                #    param.grad = None
                if self.args.model == 'nn':
                    inputs = inputs.flatten(1)
                outputs = self.model_local(inputs)
                pred = torch.argmax(outputs, dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss+=loss.item()
                acc = accuracy(pred,labels)
                running_acc+=acc

        return self.model_local
