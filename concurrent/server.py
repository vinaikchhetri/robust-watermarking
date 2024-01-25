from torch.utils.data import Dataset
import torch
import models
import torch.optim as optim
from torchvision.models import resnet18
import data_splitter
import client
import utils

import numpy as np
import time
from functools import reduce
import concurrent.futures
import os

class Server():
    def __init__(self, args):
        self.args = args
        self.K = self.args.K
        self.T = self.args.T
        self.C = self.args.C
        self.num_samples_dict = {} #number of samples per user.
        self.clients = []
        self.adversary_indices = []
        self.count_adv = []
        self.watermarkset = []
        self.watermark_data_loader = []
        self.trainset, self.testset, self.client_data_dict, self.trx_list, self.try_list, self.boundaries = data_splitter.splitter(self.args) # trainset is actually train batch. see data_splitter.py.
        self.backdoorset = utils.CustomDataset(self.trx_list, self.try_list) 
        self.trset = utils.CustomDataset(self.trainset[0], self.trainset[1]) 
        self.train_data_loader = torch.utils.data.DataLoader(self.trset, batch_size=64, shuffle=False)
        self.backdoor_data_loader = torch.utils.data.DataLoader(self.backdoorset, batch_size=64, shuffle=False)
        self.test_data_loader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=False)
        #self.poison_data_loader = torch.utils.data.DataLoader(self.test_poison, batch_size=64, shuffle=False)
        if self.args.gpu == "gpu":
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.model = args.model
        if self.model == 'nn':
            self.model_global = models.MP(28*28,200,10)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)
         
        if self.model == 'cnn':
            self.model_global = models.CNN_MNIST()
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)
        
        if self.model == 'resnet':
            # self.model_global = resnet18(num_classes=10)
            self.model_global = models.ResNet(18)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)
        
    def load_model(self, model_global):
        self.model_local.load_state_dict(model_global.state_dict())

    def create_clients(self):
        self.adversary_indices = np.random.choice(self.K, self.args.num_attackers, replace=False) # Randomly sample some adversaries from the clients.
        is_adversary = 0 # This is used determine if the client is adversary or not.
        counter = 0
        for i in range (self.K): #loop through clients
            if i in self.adversary_indices:
                is_adversary = 1
            else:
                is_adversary = 0
            self.num_samples_dict[i] = len(self.client_data_dict[i]) + self.boundaries[i]
            #print("asdf",self.boundaries[i])
            self.clients.append(client.Client(self.client_data_dict[i], self.trainset, self.trx_list[counter:counter+self.boundaries[i]], self.try_list[counter:counter+self.boundaries[i]], self.args, self.device, is_adversary))
            counter = counter + self.boundaries[i]

    def train(self):
        #best_test_acc = -1
        A_loss = []
        A_acc = []
        B_loss = []
        B_acc = []
        clients_sofar = set()
        initial = time.time()

        for rounds in range(self.T): #total number of rounds
            if self.C == 0:
                m = 1 
            else:
                m = int(max(self.C*self.K, 1)) 

            client_indices = np.random.choice(self.K, m, replace=False)
            client_indices.astype(int)
            overlap = np.intersect1d(client_indices, self.adversary_indices) # Find the overlap between the particiapting clients and the adversaries.
            self.count_adv.append(len(overlap)) # Save a history of the number of adversaries per round.
            clients_sofar = clients_sofar.union(set(client_indices))
            X_list = []
            Y_list = []
            for cli in clients_sofar:
                # print(self.clients[cli].trx_list.shape)
                # print(self.clients[cli].try_list.shape)
                X_list.append(self.clients[cli].trx_list)
                Y_list.append(self.clients[cli].try_list)
            # print(X_list[0].shape)
            # print(Y_list[0].shape)
            # print(len(X_list))
            # print(len(Y_list))
            X_list = X_list[0]
            Y_list = Y_list[0]
            self.watermarkset = utils.CustomDataset(X_list, Y_list)
            self.watermark_data_loader = torch.utils.data.DataLoader(self.watermarkset, batch_size=64, shuffle=False) 
            num_samples_list = [self.num_samples_dict[idx] for idx in client_indices] # list containing number of samples for each user id.
            total_num_samples = reduce(lambda x,y: x+y, num_samples_list, 0)

            store = {}
            avg_loss= 0 
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(client_indices), os.cpu_count() - 1)) as executor:
                results = []
                for index,client_idx in enumerate(client_indices): #loop through selected clients
                    self.clients[client_idx].load_model(self.model_global)
                    results.append(executor.submit(self.clients[client_idx].client_update).result())            
                # Retrieve results as they become available
                for ind,future in enumerate(results):
                    store[ind] = future.state_dict()
                    # store[ind] = future[0].state_dict()
                    #avg_loss+=future[1]
                #avg_loss/=len(results)

            w_global = {}
            for layer in store[0]:
                sum = 0
                for user_key in store:
                    sum += store[user_key][layer]*num_samples_list[user_key]/total_num_samples
                w_global[layer] = sum

            self.model_global.load_state_dict(w_global)

            # Performing evaluation on test data.
            test_loss, test_acc, watermark_loss, watermark_acc = self.test()
            A_loss.append(test_loss)
            B_loss.append(watermark_loss)
            A_acc.append(test_acc)
            B_acc.append(watermark_acc)

            print('Round '+ str(rounds))
            print(f'server stats: [test-loss: {test_loss:.3f}')
            print(f'server stats: [test-accuracy: {test_acc:.3f}')
            print()
            print(f'server stats: [watermark_loss: {watermark_loss:.3f}')
            print(f'server stats: [watermark_accuracy: {watermark_acc:.3f}')

        print("finished ", time.time() - initial)

        # torch.save(self.model_global.state_dict(), 'watermarked_model_'+self.args.name+'.pt')
        # torch.save(A_loss, 'test-loss-'+self.args.name+'.pt')
        # torch.save(A_acc, 'test-acc-'+self.args.name+'.pt')
        # torch.save(B_loss, 'poison-loss-'+self.args.name+'.pt')
        # torch.save(B_acc, 'poison-acc-'+self.args.name+'.pt')


    def test(self):
        with torch.no_grad():
            self.model_global.eval()
            test_running_loss = 0.0
            test_running_acc = 0.0
            for index1,data in enumerate(self.test_data_loader):  
                inputs, labels = data
                inputs = inputs.to(self.device)
                if self.model == 'nn':
                    inputs = inputs.flatten(1)
                labels = labels.to(self.device)
                output = self.model_global(inputs)
                loss = self.criterion(output, labels)
                pred = torch.argmax(output, dim=1)

                acc = utils.accuracy(pred, labels)
                test_running_acc += acc
                test_running_loss += loss

            watermark_running_loss = 0.0
            watermark_running_acc = 0.0
            for index2,data in enumerate(self.watermark_data_loader):  
                inputs, labels = data
                inputs = inputs.to(self.device)
                if self.model == 'nn':
                    inputs = inputs.flatten(1)
                labels = labels.to(self.device)
                output = self.model_global(inputs)
                loss = self.criterion(output, labels)
                pred = torch.argmax(output, dim=1)

                acc = utils.accuracy(pred, labels)
                watermark_running_acc += acc
                watermark_running_loss += loss

        return test_running_loss/(index1+1), test_running_acc/(index1+1), watermark_running_loss/(index2+1), watermark_running_acc/(index2+1) 

    

            
