import os
import json
from  PIL import  Image
from  collections import  defaultdict
import torch
from torch.utils.data import  Dataset
labels = ['black', 'blue', 'gray', 'green', 'orange', 'red', 'white', 'yellow']
def decode_label(index):
    return  labels[index]

def encode_label_from_path(path):
    for index,value in enumerate(labels):
        if value in path:
            return  index
        
class VehicleColorDataset(Dataset):
    def __init__(self, image_list, class_list, transforms = None):
        self.transform = transforms
        self.image_list = image_list
        self.class_list = class_list
        self.data_len = len(self.image_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(self.class_list[index])
    

class Logger(object):
    def __init__(self, log_dir, name, chkpt_interval):
        super(Logger,self).__init__()
        self.chkpt_interval = chkpt_interval
        self.log_dir = log_dir
        self.name = name
        os.makedirs(os.path.join(log_dir, name), exist_ok= True)
        self.log_path = os.path.join(log_dir, name, 'logs.json')
        self.model_path = os.path.join(log_dir, name, 'model_best.pt')
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0
        self.best_acc = 0.0 # Tracking the best accuracy so only save the best model, 102 don't have enough space to save all models

    def log(self, key, value ):
        if isinstance(value, dict):
            for k,v in value.items():
                self.log(f'{key}.{k}',v)
        else:
            self.logs[key].append(value)

    def checkpoint(self, model, current_acc):
        if current_acc > self.best_acc:
            print(f"New best accuracy: {current_acc:.4f} (prev: {self.best_acc:.4f}), saving model...")
            self.best_acc = current_acc
            self.save(model)
        self.logs['epoch'] +=1
        # if (self.logs['epoch'] + 1 ) % self.chkpt_interval == 0:
        #     self.save(model)
        

    def save(self, model):
        print("Saving Model...")
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, sort_keys=True, indent=4)
        epoch = self.logs['epoch'] + 1
        torch.save(model.state_dict(), self.model_path)