# Import statements
import torch
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import time
import argparse
from utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    steps = 0
    print_every = 10
    for e in range(epochs):
        running_loss = 0
        
        # dataloaders[0] is training dataloader
        for ii, (inputs, labels) in enumerate(dataloaders[0]):
            steps += 1 
            #Check if torch.cuda.is_available(): 
            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda') # use cuda
            else:
                model.cpu() # use a CPU if user says anything other than "gpu"
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valloss = 0
                accuracy=0
                
                # dataloaders[1] is validation dataloader
                for ii, (inputs2,labels2) in enumerate(dataloaders[1]): 
                        optimizer.zero_grad()
                        
                        if gpu == 'gpu':
                            inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') # use cuda
                            model.to('cuda:0') # using cuda
                        else:
                            pass # just use inputs as is 

                        with torch.no_grad():    
                            outputs = model.forward(inputs2)
                            valloss = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                valloss = valloss / len(dataloaders[1])
                accuracy = accuracy /len(dataloaders[1])

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(valloss),
                      "Accuracy: {:.4f}".format(accuracy),
                     )

                running_loss = 0
            
def main():
    args = parse_args()   
    print('Execution started....')
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
    
    validataion_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                                      [0.229, 0.224, 0.225])]) 

    testing_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])]) 

    image_datasets = [ImageFolder(train_dir, transform=training_transforms),
                      ImageFolder(val_dir, transform=validataion_transforms),
                      ImageFolder(test_dir, transform=testing_transforms)]
    
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('drop', nn.Dropout(p=0.6)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = image_datasets[0].class_to_idx
    gpu = args.gpu # get the gpu settings
    start = time.time()
    print('Training of Model Started at:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)))
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    end = time.time()
    total_time = end - start
    print('Training of Model Completed at:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end)))
    print("\n Total Time spent on training the model: {:.0f}m {:.0f}s".format(total_time//60, total_time % 60))
    model.class_to_idx = class_index
    path = args.save_dir
    save_checkpoint(path, model, optimizer, args, classifier)


if __name__ == "__main__":
    main()
    