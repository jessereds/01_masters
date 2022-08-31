# IMPORTANT: If you're doing subsequent runs,

# 1. Use specified seed values for each job submission
# 2. Comment out the initial torch.save on line 104(ish)

from __future__ import print_function
from __future__ import division
from random import sample
import torch
import torch.utils.data.distributed
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import pathlib
from PIL import Image
import tarfile
import shutil
import time
import copy
import sys

print("Python Version: ", sys.version)
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

imgdir_path = pathlib.Path(str(os.getenv('location')))

file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])

print("The number of images in the dataset is:", len(file_list))

print("Is a GPU available?", torch.cuda.is_available())
print("How many GPUs are available?", torch.cuda.device_count())
print("What's the current GPU number?", torch.cuda.current_device())
print("Where's the first GPU?", torch.cuda.device(0))
print("What are the names of the GPUs?")
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('This should say cuda if the GPUs are set up properly:', device)

namorph = pd.read_table(str(os.getenv('location2')))

labels = torch.zeros(len(namorph.TType), dtype=torch.int64)

for i in range(len(namorph.TType)):
    if namorph.TType[i] == -5.0:
        labels[i] = 0 # Elliptical
    elif -3 <= namorph.TType[i] and namorph.TType[i] <= 0:
        labels[i] = 1 # Lenticular
    elif 1 <= namorph.TType[i] and namorph.TType[i] <= 9:
        labels[i] = 2 # Spiral
    elif 10 == namorph.TType[i] or 99 == namorph.TType[i]:
        labels[i] = 3 # Irr+Misc

print("The labels tensor is:", labels)

img_height, img_width = 256, 256
size = [224, 224]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height, img_width)),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
])

class ImageDataset():
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label
    def __len__(self):
        return len(self.labels)

image_dataset = ImageDataset(file_list, labels, transform)

file_number = 9
torch.manual_seed(len(file_list) * file_number) # Use other seed values for subsequent runs.
np.random.seed(len(file_list) * file_number) # I think this is the relevant seed, not the Torch one above.
data_dir = imgdir_path
model_name = "alexnet"
# model_name = "resnet"
num_classes = 4
batch_size = 384
# num_runs = 100
num_epochs_1 = 200
num_epochs_8 = 400
feature_extract = False
# samples_list = []
# samples_counts = []
# samples_percentages = []
# os.mkdir(f'/scratch/jsa378/misc_output')
os.makedirs(f'/scratch/jsa378/misc_output/individual_plots/model_1', exist_ok=True)
os.makedirs(f'/scratch/jsa378/misc_output/individual_plots/model_8', exist_ok=True)
torch.save({ # To start from scratch. Comment out if carrying on from prior run(s).
    'samples_list': [], # Indices for the training and test data sets for each random split.
    'samples_counts': [], # How many E, L, S, Irr+Misc are in training and test for each random split.
    'samples_percentages': [], # Percentages of E, L, S, Irr+Misc in training and test for each random split.
    'pretrained_best_train_acc': [], # Peak training accuracies for each random split.
    'pretrained_best_test_acc': [], # Peak test accuracies for each random split.
    'pretrained_best_train_acc_epoch': [], # Epoch when peak training accuracy occurred.
    'pretrained_best_test_acc_epoch': [], # Epoch when peak testing accuracy occurred.
    'pretrained_best_train_loss': [], # 
    'pretrained_best_test_loss': [],
    'pretrained_best_train_loss_epoch': [],
    'pretrained_best_test_loss_epoch': [],
    'non_pretrained_best_train_acc': [],
    'non_pretrained_best_test_acc': [],
    'non_pretrained_best_train_acc_epoch': [],
    'non_pretrained_best_test_acc_epoch': [],
    'non_pretrained_best_train_loss': [],
    'non_pretrained_best_test_loss': [],
    'non_pretrained_best_train_loss_epoch': [],
    'non_pretrained_best_test_loss_epoch': [],
    'pretrained_cumulative_train_acc': [],
    'pretrained_cumulative_test_acc': [],
    'pretrained_cumulative_train_loss': [],
    'pretrained_cumulative_test_loss': [],
    'non_pretrained_cumulative_train_acc': [],
    'non_pretrained_cumulative_test_acc': [],
    'non_pretrained_cumulative_train_loss': [],
    'non_pretrained_cumulative_test_loss': [],
    'pretrained_elliptical_acc': [],
    'pretrained_lenticular_acc': [],
    'pretrained_spiral_acc': [],
    'pretrained_irrmisc_acc': [],
    'non_pretrained_elliptical_acc': [],
    'non_pretrained_lenticular_acc': [],
    'non_pretrained_spiral_acc': [],
    'non_pretrained_irrmisc_acc': [],
    },
	# '/project/rrg-lelliott/jsa378/misc_output/sample_perf_history.tar')
    f'/scratch/jsa378/misc_output/sample_perf_history_{file_number}.tar')
# sample_perf_history = torch.load(f'/project/rrg-lelliott/jsa378/misc_output/sample_perf_history.tar')
sample_perf_history = torch.load(f'/scratch/jsa378/misc_output/sample_perf_history_{file_number}.tar')
train_size = 12000
test_size = len(image_dataset) - train_size
indices = torch.arange(len(labels))

pretrained_train_accs = []
pretrained_train_losses = []
pretrained_test_accs = []
pretrained_test_losses = []
non_pretrained_train_accs = []
non_pretrained_train_losses = []
non_pretrained_test_accs = []
non_pretrained_test_losses = []

def make_tarfile(output_filename, source_dir):
    # since = time.time()
    with tarfile.open(output_filename, "x") as tar:
        tar.add(source_dir, arcname='.')
    # time_elapsed = time.time() - since
    # print('Archive created in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def train_model(model, dataloaders, criterion, optimizer, num_epochs, model_number, is_inception=False):
    since = time.time()
    
    val_acc_history = []
    train_acc_history = []
    
    val_loss_history = []
    train_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    os.mkdir(f'/scratch/jsa378/model_{model_number}_run_{run}')
    # os.mkdir(f'/project/rrg-lelliott/jsa378/model_{model_number}_ddp_output/run_{run}/')
        
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode
        
            running_loss = 0.0
            running_corrects = 0
        
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
            
                # zero the parameter gradients
                optimizer.zero_grad()
            
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an
                    # auxiliary output. In train mode we calculate the loss by
                    # summing the final output and the auxiliary output but in
                    # testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
        
        print()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'best_model_accuracy': best_acc,
            # 'best_model_weights': best_model_wts,
            'loss': loss,
            # 'train_hist': train_acc_history,
            # 'val_hist': val_acc_history,
        # }, f'/project/rrg-lelliott/jsa378/model_{model_number}_ddp_output/run_{run}/checkpoint_run_{run}_epoch_{epoch}.tar')
        # }, str(os.getenv('location3'))+f'/checkpoint_run_{run}_epoch_{epoch}.tar')
        }, f'/scratch/jsa378/model_{model_number}_run_{run}/checkpoint_run_{run}_epoch_{epoch}.tar')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_model_accuracy': best_acc,
            'best_model_weights': best_model_wts,
            'loss': loss,
            'train_acc_hist': train_acc_history,
            'val_acc_hist': val_acc_history,
            'train_loss_hist': train_loss_history,
            'val_loss_hist': val_loss_history,
        # }, f'/project/rrg-lelliott/jsa378/model_{model_number}_ddp_output/run_{run}/checkpoint_run_{run}.tar')
        # }, str(os.getenv('location3'))+f'/checkpoint_run_{run}.tar')
        }, f'/scratch/jsa378/model_{model_number}_run_{run}/checkpoint_run_{run}.tar')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, val_loss_history, train_loss_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained):
    # Initialize these variables which will be set in this if statement.
    # Each of these variables is model specific.
    
    model_ft = None
    input_size = 0
    
    if model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def main():

    rank = os.environ.get("SLURM_LOCALID")

    current_device = 0
    torch.cuda.set_device(current_device)
    
    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))

    dist.init_process_group(backend="mpi", init_method='tcp://127.0.0.1:3456') # Use backend="mpi" or "gloo". NCCL does not work on a single GPU due to a hard-coded multi-GPU topology check.
    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))

    training_set_indices = torch.from_numpy(np.random.choice(len(labels), train_size, replace=False))
    combined = torch.cat((indices, training_set_indices))
    uniques, counts = combined.unique(return_counts=True)
    test_set_indices = uniques[counts == 1]
    intersection = uniques[counts > 1]
    # samples_list.append([training_set_indices, test_set_indices])
    sample_perf_history['samples_list'].append({
        f'training_set_indices': training_set_indices,
        f'test_set_indices': test_set_indices,
        })

    train_dataset = Subset(image_dataset, training_set_indices)
    test_dataset = Subset(image_dataset, test_set_indices)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=6, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=(test_sampler is None), num_workers=6, sampler=test_sampler)
    
    dataloaders_dict = {
    "train": train_dataloader,
    "val": test_dataloader
    }

    train_elliptical_count = 0
    train_lenticular_count = 0
    train_spiral_count = 0
    train_irrmisc_count = 0

    for data in train_dataloader:
        images, labels2 = data
        for i in range(len(labels2)):
            if labels2[i] == 0:
                train_elliptical_count += 1
            if labels2[i] == 1:
                train_lenticular_count += 1
            if labels2[i] == 2:
                train_spiral_count += 1
            if labels2[i] == 3:
                train_irrmisc_count += 1

    print('Training set breakdown:')
    print('Elliptical', train_elliptical_count, 'Lenticular', train_lenticular_count, 'Spiral', train_spiral_count, 'Irr+Misc', train_irrmisc_count)
    print('Sum of the above numbers:')
    print(train_elliptical_count + train_lenticular_count + train_spiral_count + train_irrmisc_count)
    print('Length of the train_dataset:')
    print(len(train_dataset))
    print('Percentages:')
    print('Elliptical', train_elliptical_count/len(train_dataset), 'Lenticular', train_lenticular_count/len(train_dataset), 'Spiral', train_spiral_count/len(train_dataset), 'Irr+Misc', train_irrmisc_count/len(train_dataset))
    print('')

    test_elliptical_count = 0
    test_lenticular_count = 0
    test_spiral_count = 0
    test_irrmisc_count = 0

    for data in test_dataloader:
        images, labels2 = data
        for i in range(len(labels2)):
            if labels2[i] == 0:
                test_elliptical_count += 1
            if labels2[i] == 1:
                test_lenticular_count += 1
            if labels2[i] == 2:
                test_spiral_count += 1
            if labels2[i] == 3:
                test_irrmisc_count += 1

    print('Test set breakdown:')
    print('Elliptical', test_elliptical_count, 'Lenticular', test_lenticular_count, 'Spiral', test_spiral_count, 'Irr+Misc', test_irrmisc_count)
    print('Sum of the above numbers:')
    print(test_elliptical_count + test_lenticular_count + test_spiral_count + test_irrmisc_count)
    print('Length of the test_dataset:')
    print(len(test_dataset))
    print('Percentages:')
    print('Elliptical', test_elliptical_count/len(test_dataset), 'Lenticular', test_lenticular_count/len(test_dataset), 'Spiral', test_spiral_count/len(test_dataset), 'Irr+Misc', test_irrmisc_count/len(test_dataset))
    print('')

#     samples_counts.append([train_elliptical_count, train_lenticular_count, train_spiral_count, train_irrmisc_count])
#     samples_counts.append([test_elliptical_count, test_lenticular_count, test_spiral_count, test_irrmisc_count])
#     samples_percentages.append([train_elliptical_count/len(train_dataset), train_lenticular_count/len(train_dataset), train_spiral_count/len(train_dataset), train_irrmisc_count/len(train_dataset)])
#     samples_percentages.append([test_elliptical_count/len(test_dataset), test_lenticular_count/len(test_dataset), test_spiral_count/len(test_dataset), test_irrmisc_count/len(test_dataset)])

    sample_perf_history['samples_counts'].append({
	f'training_set': [train_elliptical_count, train_lenticular_count, train_spiral_count, train_irrmisc_count],
	f'test_set': [test_elliptical_count, test_lenticular_count, test_spiral_count, test_irrmisc_count],
    })

    sample_perf_history['samples_percentages'].append({
	f'training_set': [train_elliptical_count/len(train_dataset), train_lenticular_count/len(train_dataset), train_spiral_count/len(train_dataset), train_irrmisc_count/len(train_dataset)],
	f'test_set': [test_elliptical_count/len(test_dataset), test_lenticular_count/len(test_dataset), test_spiral_count/len(test_dataset), test_irrmisc_count/len(test_dataset)],
    })
    
    # torch.save(sample_perf_history, '/project/rrg-lelliott/jsa378/misc_output/sample_perf_history.tar')
    # }, str(os.getenv('location3'))+f'/misc_output/sample_perf_history.tar')
    # 

    criterion = nn.CrossEntropyLoss()
    
    print(f'Beginning run {run} of pretrained AlexNet.')
    
    model_number = 1
    model_1, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    print(model_1)
    model_1 = model_1.to(device)
    model_1 = torch.nn.parallel.DistributedDataParallel(model_1, device_ids=[current_device]) # Wrap the model with DistributedDataParallel
    
    params_to_update = model_1.parameters()
    print("Params to learn:")

    if feature_extract:
        params_to_update = []
        for name,param in model_1.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_1.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    model_1, val_acc_hist, train_acc_hist, val_loss_hist, train_loss_hist = train_model(model_1, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs_1, model_number=model_number, is_inception=(model_name=="inception"))

    # checkpoint = torch.load(f'/project/rrg-lelliott/jsa378/model_1_ddp_output/run_{run}/checkpoint_run_{run}.tar')
    # checkpoint = torch.load(str(os.getenv('location3'))+f'/checkpoint_run_{run}.tar')
    checkpoint = torch.load(f'/scratch/jsa378/model_{model_number}_run_{run}/checkpoint_run_{run}.tar')

    train_acc_list = []
    for i in range(len(checkpoint['train_acc_hist'])):
        train_acc_list.append(checkpoint['train_acc_hist'][i].item())

    val_acc_list = []
    for i in range(len(checkpoint['val_acc_hist'])):
        val_acc_list.append(checkpoint['val_acc_hist'][i].item())

    train_loss_list = []
    for i in range(len(checkpoint['train_loss_hist'])):
        # train_loss_list.append(checkpoint['train_loss_hist'][i].item())
        train_loss_list.append(checkpoint['train_loss_hist'][i])

    val_loss_list = []
    for i in range(len(checkpoint['val_loss_hist'])):
        # val_loss_list.append(checkpoint['val_loss_hist'][i].item())
        val_loss_list.append(checkpoint['val_loss_hist'][i])

    print("Training accuracy history:")
    print(train_acc_list)
    print("Test accuracy history:")
    print(val_acc_list)

    print("The best training accuracy was achieved in epoch", train_acc_list.index(max(train_acc_list)), "and that best accuracy was", max(train_acc_list))
    print("The best testing accuracy was achieved in epoch", val_acc_list.index(max(val_acc_list)), "and that best accuracy was", max(val_acc_list))

    sample_perf_history['pretrained_best_train_acc'].append(max(train_acc_list))
    sample_perf_history['pretrained_best_test_acc'].append(max(val_acc_list))
    sample_perf_history['pretrained_best_train_acc_epoch'].append(train_acc_list.index(max(train_acc_list)))
    sample_perf_history['pretrained_best_test_acc_epoch'].append(val_acc_list.index(max(val_acc_list)))
    sample_perf_history['pretrained_best_train_loss'].append(min(train_loss_list))
    sample_perf_history['pretrained_best_test_loss'].append(min(val_loss_list))
    sample_perf_history['pretrained_best_train_loss_epoch'].append(train_loss_list.index(min(train_loss_list)))
    sample_perf_history['pretrained_best_test_loss_epoch'].append(val_loss_list.index(min(val_loss_list)))

    sample_perf_history['pretrained_cumulative_train_acc'].append(train_acc_list)
    sample_perf_history['pretrained_cumulative_test_acc'].append(val_acc_list)
    sample_perf_history['pretrained_cumulative_train_loss'].append(train_loss_list)
    sample_perf_history['pretrained_cumulative_test_loss'].append(val_loss_list)

    data = torch.tensor(val_acc_hist)
    data2 = torch.tensor(train_acc_hist)
    fig = plt.figure()
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f'Pretrained AlexNet, run {run}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    plt.plot(data, label='Test')
    plt.plot(data2, label='Train')
    ax.legend()
    # fig.savefig(f'/project/rrg-lelliott/jsa378/model_1_ddp_output/run_{run}/acc_history_run_{run}.png')
    # fig.savefig(str(os.getenv('location3'))+f'/acc_history_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/model_{model_number}_run_{run}/acc_history_model_{model_number}_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/misc_output/individual_plots/model_{model_number}/acc_history_model_{model_number}_run_{run}.png')

    data = torch.tensor(val_loss_hist)
    data2 = torch.tensor(train_loss_hist)
    fig = plt.figure()
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f'Pretrained AlexNet, run {run}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.plot(data, label='Test')
    plt.plot(data2, label='Train')
    ax.legend()
    # fig.savefig(f'/project/rrg-lelliott/jsa378/model_1_ddp_output/run_{run}/loss_history_run_{run}.png')
    # fig.savefig(str(os.getenv('location3'))+f'/loss_history_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/model_{model_number}_run_{run}/loss_history_model_{model_number}_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/misc_output/individual_plots/model_{model_number}/loss_history_model_{model_number}_run_{run}.png')

    nb_classes = 4

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders_dict['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_1(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    torch.set_printoptions(sci_mode=False)
    print(confusion_matrix)
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    
    checkpoint['confusion_matrix'] = confusion_matrix
    checkpoint['per_class_accuracies'] = confusion_matrix.diag()/confusion_matrix.sum(1)
    checkpoint['test_accuracy'] = torch.trace(confusion_matrix)/len(test_dataset)

    sample_perf_history['pretrained_elliptical_acc'].append(checkpoint['per_class_accuracies'][0].item())
    sample_perf_history['pretrained_lenticular_acc'].append(checkpoint['per_class_accuracies'][1].item())
    sample_perf_history['pretrained_spiral_acc'].append(checkpoint['per_class_accuracies'][2].item())
    sample_perf_history['pretrained_irrmisc_acc'].append(checkpoint['per_class_accuracies'][3].item())

    # torch.save(checkpoint, f'/project/rrg-lelliott/jsa378/model_1_ddp_output/run_{run}/checkpoint_run_{run}.tar')
    # torch.save(checkpoint, str(os.getenv('location3'))+f'/checkpoint_run_{run}.tar')
    torch.save(checkpoint, f'/scratch/jsa378/model_{model_number}_run_{run}/checkpoint_run_{run}.tar')

    array = confusion_matrix
    ls = ['Elliptical', 'Lenticular', 'Spiral', 'Irr+Misc']
    df_cm = pd.DataFrame(array, index = ls, columns = ls)
    fig = plt.figure(figsize=(10, 7))
    plt.title(f'Pretrained AlexNet, run {run}')
    sn.heatmap(df_cm, annot=True, fmt='g')
    # fig.savefig(f'/project/rrg-lelliott/jsa378/model_1_ddp_output/run_{run}/confusion_matrix_run_{run}.png')
    # fig.savefig(str(os.getenv('location3'))+f'/confusion_matrix_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/model_{model_number}_run_{run}/confusion_matrix_model_{model_number}_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/misc_output/individual_plots/model_{model_number}/confusion_matrix_model_{model_number}_run_{run}.png')

    since = time.time()
    os.mkdir(f'/project/rrg-lelliott/jsa378/model_{model_number}_ddp_output/run_{run}/')
    make_tarfile(f'/project/rrg-lelliott/jsa378/model_{model_number}_ddp_output/run_{run}/run_{run}.tar', f'/scratch/jsa378/model_{model_number}_run_{run}')
    shutil.rmtree(f'/scratch/jsa378/model_{model_number}_run_{run}')
    time_elapsed = time.time() - since
    print(f'Run {run}', 'archive of pretrained AlexNet created in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    ###
    
    print(f'Beginning run {run} of non-pretrained AlexNet.')
    
    model_number = 8
    model_8, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    print(model_8)
    model_8 = model_8.to(device)
    model_8 = torch.nn.parallel.DistributedDataParallel(model_8, device_ids=[current_device]) # Wrap the model with DistributedDataParallel
    
    params_to_update = model_8.parameters()
    print("Params to learn:")

    if feature_extract:
        params_to_update = []
        for name,param in model_8.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_8.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    model_8, val_acc_hist, train_acc_hist, val_loss_hist, train_loss_hist = train_model(model_8, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs_8, model_number=model_number, is_inception=(model_name=="inception"))

    # checkpoint = torch.load(f'/project/rrg-lelliott/jsa378/model_8_ddp_output/run_{run}/checkpoint_run_{run}.tar')
    # checkpoint = torch.load(str(os.getenv('location3'))+f'/checkpoint_run_{run}.tar')
    checkpoint = torch.load(f'/scratch/jsa378/model_{model_number}_run_{run}/checkpoint_run_{run}.tar')

    train_acc_list = []
    for i in range(len(checkpoint['train_acc_hist'])):
        train_acc_list.append(checkpoint['train_acc_hist'][i].item())

    val_acc_list = []
    for i in range(len(checkpoint['val_acc_hist'])):
        val_acc_list.append(checkpoint['val_acc_hist'][i].item())

    train_loss_list = []
    for i in range(len(checkpoint['train_loss_hist'])):
        # train_loss_list.append(checkpoint['train_loss_hist'][i].item())
        train_loss_list.append(checkpoint['train_loss_hist'][i])

    val_loss_list = []
    for i in range(len(checkpoint['val_loss_hist'])):
        # val_loss_list.append(checkpoint['val_loss_hist'][i].item())
        val_loss_list.append(checkpoint['val_loss_hist'][i])

    print("Training accuracy history:")
    print(train_acc_list)
    print("Test accuracy history:")
    print(val_acc_list)

    print("The best training accuracy was achieved in epoch", train_acc_list.index(max(train_acc_list)), "and that best accuracy was", max(train_acc_list))
    print("The best testing accuracy was achieved in epoch", val_acc_list.index(max(val_acc_list)), "and that best accuracy was", max(val_acc_list))

    sample_perf_history['non_pretrained_best_train_acc'].append(max(train_acc_list))
    sample_perf_history['non_pretrained_best_test_acc'].append(max(val_acc_list))
    sample_perf_history['non_pretrained_best_train_acc_epoch'].append(train_acc_list.index(max(train_acc_list)))
    sample_perf_history['non_pretrained_best_test_acc_epoch'].append(val_acc_list.index(max(val_acc_list)))
    sample_perf_history['non_pretrained_best_train_loss'].append(min(train_loss_list))
    sample_perf_history['non_pretrained_best_test_loss'].append(min(val_loss_list))
    sample_perf_history['non_pretrained_best_train_loss_epoch'].append(train_loss_list.index(min(train_loss_list)))
    sample_perf_history['non_pretrained_best_test_loss_epoch'].append(val_loss_list.index(min(val_loss_list)))

    sample_perf_history['non_pretrained_cumulative_train_acc'].append(train_acc_list)
    sample_perf_history['non_pretrained_cumulative_test_acc'].append(val_acc_list)
    sample_perf_history['non_pretrained_cumulative_train_loss'].append(train_loss_list)
    sample_perf_history['non_pretrained_cumulative_test_loss'].append(val_loss_list)

    data = torch.tensor(val_acc_hist)
    data2 = torch.tensor(train_acc_hist)
    fig = plt.figure()
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f'Non-Pretrained AlexNet, run {run}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    plt.plot(data, label='Test')
    plt.plot(data2, label='Train')
    ax.legend()
    # fig.savefig(f'/project/rrg-lelliott/jsa378/model_8_ddp_output/run_{run}/acc_history_run_{run}.png')
    # fig.savefig(str(os.getenv('location3'))+f'/acc_history_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/model_{model_number}_run_{run}/acc_history_model_{model_number}_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/misc_output/individual_plots/model_{model_number}/acc_history_model_{model_number}_run_{run}.png')

    data = torch.tensor(val_loss_hist)
    data2 = torch.tensor(train_loss_hist)
    fig = plt.figure()
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f'Non-Pretrained AlexNet, run {run}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.plot(data, label='Test')
    plt.plot(data2, label='Train')
    ax.legend()
    # fig.savefig(f'/project/rrg-lelliott/jsa378/model_8_ddp_output/run_{run}/loss_history_run_{run}.png')
    # fig.savefig(str(os.getenv('location3'))+f'/loss_history_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/model_{model_number}_run_{run}/loss_history_model_{model_number}_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/misc_output/individual_plots/model_{model_number}/loss_history_model_{model_number}_run_{run}.png')

    nb_classes = 4

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders_dict['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_8(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    torch.set_printoptions(sci_mode=False)
    print(confusion_matrix)
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    
    checkpoint['confusion_matrix'] = confusion_matrix
    checkpoint['per_class_accuracies'] = confusion_matrix.diag()/confusion_matrix.sum(1)
    checkpoint['test_accuracy'] = torch.trace(confusion_matrix)/len(test_dataset)

    sample_perf_history['non_pretrained_elliptical_acc'].append(checkpoint['per_class_accuracies'][0].item())
    sample_perf_history['non_pretrained_lenticular_acc'].append(checkpoint['per_class_accuracies'][1].item())
    sample_perf_history['non_pretrained_spiral_acc'].append(checkpoint['per_class_accuracies'][2].item())
    sample_perf_history['non_pretrained_irrmisc_acc'].append(checkpoint['per_class_accuracies'][3].item())

    # torch.save(checkpoint, f'/project/rrg-lelliott/jsa378/model_8_ddp_output/run_{run}/checkpoint_run_{run}.tar')
    # torch.save(checkpoint, str(os.getenv('location3'))+f'/checkpoint_run_{run}.tar')
    torch.save(checkpoint, f'/scratch/jsa378/model_{model_number}_run_{run}/checkpoint_run_{run}.tar')

    array = confusion_matrix
    ls = ['Elliptical', 'Lenticular', 'Spiral', 'Irr+Misc']
    df_cm = pd.DataFrame(array, index = ls, columns = ls)
    fig = plt.figure(figsize=(10, 7))
    plt.title(f'Non-Pretrained AlexNet, run {run}')
    sn.heatmap(df_cm, annot=True, fmt='g')
    # fig.savefig(f'/project/rrg-lelliott/jsa378/model_8_ddp_output/run_{run}/confusion_matrix_run_{run}.png')
    # fig.savefig(str(os.getenv('location3'))+f'/confusion_matrix_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/model_{model_number}_run_{run}/confusion_matrix_model_{model_number}_run_{run}.png')
    fig.savefig(f'/scratch/jsa378/misc_output/individual_plots/model_{model_number}/confusion_matrix_model_{model_number}_run_{run}.png')

    since = time.time()
    os.mkdir(f'/project/rrg-lelliott/jsa378/model_{model_number}_ddp_output/run_{run}/')
    make_tarfile(f'/project/rrg-lelliott/jsa378/model_{model_number}_ddp_output/run_{run}/run_{run}.tar', f'/scratch/jsa378/model_{model_number}_run_{run}')
    shutil.rmtree(f'/scratch/jsa378/model_{model_number}_run_{run}')
    time_elapsed = time.time() - since
    print(f'Run {run}', 'archive of pretrained AlexNet created in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # torch.save(sample_perf_history, '/project/rrg-lelliott/jsa378/misc_output/sample_perf_history.tar')
    # }, str(os.getenv('location3'))+f'/misc_output/sample_perf_history.tar')

    torch.save(sample_perf_history, f'/scratch/jsa378/misc_output/sample_perf_history_{file_number}.tar')

    dist.destroy_process_group()
     

# for run in range(num_runs):
for run in range(10 * file_number, 10 * (file_number + 1)):

    # print("Beginning run", run, "of", num_runs, ".")
    print("Beginning run", run, ".")

    main()

# # Make summary figures.

# # sample_perf_history = torch.load(f'/project/rrg-lelliott/jsa378/misc_output/sample_perf_history.tar')
# sample_perf_history = torch.load(f'/scratch/jsa378/misc_output/sample_perf_history.tar')

# # Make general histograms.

# color1 = []
# color2 = []

# for i in range(len(sample_perf_history['pretrained_best_test_acc'])):
#     color1.append('tab:green')
#     color2.append('tab:red')

# data = torch.tensor(sample_perf_history['pretrained_best_test_acc'])
# data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_acc'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(data, bins=10, color=color1)
# ax2.hist(data2, bins=10, color=color2)
# ax1.set_title('Peak Test Accuracy: \n Pretrained AlexNet')
# ax1.set_xlabel('Accuracy')
# ax1.set_ylabel('Count')
# ax2.set_title('Peak Test Accuracy: \n Non-Pretrained AlexNet')
# ax2.set_xlabel('Accuracy')
# # ax2.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/accuracy_histogram.png')
# fig.savefig(f'/scratch/jsa378/misc_output/accuracy_histogram.png')

# data = torch.tensor(sample_perf_history['pretrained_best_test_acc_epoch'])
# data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_acc_epoch'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
# ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(data, bins=10, color=color1)
# ax2.hist(data2, bins=10, color=color2)
# ax1.set_title('Epoch of Peak \n Test Accuracy: \n Pretrained AlexNet')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Count')
# ax2.set_title('Epoch of Peak \n Test Accuracy: \n Non-Pretrained AlexNet')
# ax2.set_xlabel('Epoch')
# # ax2.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/accuracy_epoch_histogram.png')
# fig.savefig(f'/scratch/jsa378/misc_output/accuracy_epoch_histogram.png')

# data = torch.tensor(sample_perf_history['pretrained_best_test_loss'])
# data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_loss'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(data, bins=10, color=color1)
# ax2.hist(data2, bins=10, color=color2)
# ax1.set_title('Minimum Test Loss: \n Pretrained AlexNet')
# ax1.set_xlabel('Loss')
# ax1.set_ylabel('Count')
# ax2.set_title('Minimum Test Loss: \n Non-Pretrained AlexNet')
# ax2.set_xlabel('Loss')
# # ax2.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_histogram.png')
# fig.savefig(f'/scratch/jsa378/misc_output/loss_histogram.png')

# data = torch.tensor(sample_perf_history['pretrained_best_test_loss_epoch'])
# data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_loss_epoch'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
# # plt.subplots_adjust(left=0.1, right=1)
# ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(data, bins=10, color=color1)
# ax2.hist(data2, bins=10, color=color2)
# ax1.set_title('Epoch of Minimum \n Test Loss: \n Pretrained AlexNet')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Count')
# ax2.set_title('Epoch of Minimum \n Test Loss: \n Non-Pretrained AlexNet')
# ax2.set_xlabel('Epoch')
# # ax2.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_epoch_histogram.png')
# fig.savefig(f'/scratch/jsa378/misc_output/loss_epoch_histogram.png')

# # Make general accuracy plots (pretrained AlexNet).

# fig = plt.figure()
# ax = fig.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_title(f'Pretrained AlexNet Training Accuracy, All Runs')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Accuracy')
# for i in range(len(sample_perf_history['pretrained_cumulative_train_acc'])):
#     plt.plot(torch.tensor(sample_perf_history['pretrained_cumulative_train_acc'][i]), alpha=0.25)
# # fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/pretrained_train_acc.png')
# fig.savefig(f'/scratch/jsa378/misc_output/pretrained_train_acc.png')

# fig = plt.figure()
# ax = fig.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_title(f'Pretrained AlexNet Test Accuracy, All Runs')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Accuracy')
# for i in range(len(sample_perf_history['pretrained_cumulative_test_acc'])):
#     plt.plot(torch.tensor(sample_perf_history['pretrained_cumulative_test_acc'][i]), alpha=0.25)
# # fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/pretrained_test_acc.png')
# fig.savefig(f'/scratch/jsa378/misc_output/pretrained_test_acc.png')

# fig = plt.figure()
# ax = fig.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_title(f'Pretrained AlexNet Training Loss, All Runs')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Loss')
# for i in range(len(sample_perf_history['pretrained_cumulative_train_loss'])):
#     plt.plot(torch.tensor(sample_perf_history['pretrained_cumulative_train_loss'][i]), alpha=0.25)
# # fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/pretrained_train_loss.png')
# fig.savefig(f'/scratch/jsa378/misc_output/pretrained_train_loss.png')

# fig = plt.figure()
# ax = fig.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_title(f'Pretrained AlexNet Test Loss, All Runs')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Loss')
# for i in range(len(sample_perf_history['pretrained_cumulative_test_loss'])):
#     plt.plot(torch.tensor(sample_perf_history['pretrained_cumulative_test_loss'][i]), alpha=0.25)
# # fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/pretrained_test_loss.png')
# fig.savefig(f'/scratch/jsa378/misc_output/pretrained_test_loss.png')

# # Make general accuracy plots (non-pretrained AlexNet).

# fig = plt.figure()
# ax = fig.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_title(f'Non-Pretrained AlexNet Training Accuracy, All Runs')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Accuracy')
# for i in range(len(sample_perf_history['non_pretrained_cumulative_train_acc'])):
#     plt.plot(torch.tensor(sample_perf_history['non_pretrained_cumulative_train_acc'][i]), alpha=0.25)
# # fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/non_pretrained_train_acc.png')
# fig.savefig(f'/scratch/jsa378/misc_output/non_pretrained_train_acc.png')

# fig = plt.figure()
# ax = fig.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_title(f'Non-Pretrained AlexNet Test Accuracy, All Runs')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Accuracy')
# for i in range(len(sample_perf_history['non_pretrained_cumulative_test_acc'])):
#     plt.plot(torch.tensor(sample_perf_history['non_pretrained_cumulative_test_acc'][i]), alpha=0.25)
# # fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/non_pretrained_test_acc.png')
# fig.savefig(f'/scratch/jsa378/misc_output/non_pretrained_test_acc.png')

# fig = plt.figure()
# ax = fig.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_title(f'Non-Pretrained AlexNet Training Loss, All Runs')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Loss')
# for i in range(len(sample_perf_history['non_pretrained_cumulative_train_loss'])):
#     plt.plot(torch.tensor(sample_perf_history['non_pretrained_cumulative_train_loss'][i]), alpha=0.25)
# # fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/non_pretrained_train_loss.png')
# fig.savefig(f'/scratch/jsa378/misc_output/non_pretrained_train_loss.png')

# fig = plt.figure()
# ax = fig.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_title(f'Non-Pretrained AlexNet Test Loss, All Runs')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Loss')
# for i in range(len(sample_perf_history['non_pretrained_cumulative_test_loss'])):
#     plt.plot(torch.tensor(sample_perf_history['non_pretrained_cumulative_test_loss'][i]), alpha=0.25)
# # fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/non_pretrained_test_loss.png')
# fig.savefig(f'/scratch/jsa378/misc_output/non_pretrained_test_loss.png')

# # Make histograms of per-class accuracies.

# data = torch.tensor(sample_perf_history['pretrained_elliptical_acc'])
# data2 = torch.tensor(sample_perf_history['pretrained_lenticular_acc'])
# data3 = torch.tensor(sample_perf_history['pretrained_spiral_acc'])
# data4 = torch.tensor(sample_perf_history['pretrained_irrmisc_acc'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6), sharey=True, tight_layout=True)
# axs[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[0, 0].hist(data, bins=10)
# axs[0, 1].hist(data2, bins=10)
# axs[1, 0].hist(data3, bins=10)
# axs[1, 1].hist(data4, bins=10)
# axs[0, 0].set_title('Elliptical Classification Accuracy: \n Pretrained AlexNet')
# axs[0, 0].set_xlabel('Accuracy')
# axs[0, 0].set_ylabel('Count')
# axs[0, 1].set_title('Lenticular Classification Accuracy: \n Pretrained AlexNet')
# axs[0, 1].set_xlabel('Accuracy')
# axs[0, 1].set_ylabel('Count')
# axs[1, 0].set_title('Spiral Classification Accuracy: \n Pretrained AlexNet')
# axs[1, 0].set_xlabel('Accuracy')
# axs[1, 0].set_ylabel('Count')
# axs[1, 1].set_title('Irr+Misc Classification Accuracy: \n Pretrained AlexNet')
# axs[1, 1].set_xlabel('Accuracy')
# axs[1, 1].set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/pretrained_class_accuracy_histograms.png')
# fig.savefig(f'/scratch/jsa378/misc_output/pretrained_class_accuracy_histograms.png')

# data = torch.tensor(sample_perf_history['non_pretrained_elliptical_acc'])
# data2 = torch.tensor(sample_perf_history['non_pretrained_lenticular_acc'])
# data3 = torch.tensor(sample_perf_history['non_pretrained_spiral_acc'])
# data4 = torch.tensor(sample_perf_history['non_pretrained_irrmisc_acc'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6), sharey=True, tight_layout=True)
# axs[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[0, 0].hist(data, bins=10)
# axs[0, 1].hist(data2, bins=10)
# axs[1, 0].hist(data3, bins=10)
# axs[1, 1].hist(data4, bins=10)
# axs[0, 0].set_title('Elliptical Classification Accuracy: \n Non-Pretrained AlexNet')
# axs[0, 0].set_xlabel('Accuracy')
# axs[0, 0].set_ylabel('Count')
# axs[0, 1].set_title('Lenticular Classification Accuracy: \n Non-Pretrained AlexNet')
# axs[0, 1].set_xlabel('Accuracy')
# axs[0, 1].set_ylabel('Count')
# axs[1, 0].set_title('Spiral Classification Accuracy: \n Non-Pretrained AlexNet')
# axs[1, 0].set_xlabel('Accuracy')
# axs[1, 0].set_ylabel('Count')
# axs[1, 1].set_title('Irr+Misc Classification Accuracy: \n Non-Pretrained AlexNet')
# axs[1, 1].set_xlabel('Accuracy')
# axs[1, 1].set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/non_pretrained_class_accuracy_histograms.png')
# fig.savefig(f'/scratch/jsa378/misc_output/non_pretrained_class_accuracy_histograms.png')

# # Make side-by-side plots of pretrained and non-pretrained AlexNet over the same run.

# for i in range(len(sample_perf_history['pretrained_cumulative_train_acc'])):
#     # fig = plt.figure()
#     # ax = fig.gca()
#     fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6),sharey=False, tight_layout=True)
#     axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
#     axs[0, 0].set_title(f'Pretrained AlexNet, run {i}')
#     axs[0, 0].set_xlabel('Epoch')
#     axs[0, 0].set_ylabel('Accuracy')
#     data = torch.tensor(sample_perf_history['pretrained_cumulative_test_acc'][i])
#     data2 = torch.tensor(sample_perf_history['pretrained_cumulative_train_acc'][i])
#     axs[0, 0].plot(data, label='Test')
#     axs[0, 0].plot(data2, label='Train')
#     axs[0, 0].legend()
#     axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
#     axs[0, 1].set_title(f'Non-Pretrained AlexNet, run {i}')
#     axs[0, 1].set_xlabel('Epoch')
#     axs[0, 1].set_ylabel('Accuracy')
#     data3 = torch.tensor(sample_perf_history['non_pretrained_cumulative_test_acc'][i])
#     data4 = torch.tensor(sample_perf_history['non_pretrained_cumulative_train_acc'][i])
#     axs[0, 1].plot(data3, label='Test')
#     axs[0, 1].plot(data4, label='Train')
#     axs[0, 1].legend()
#     axs[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
#     axs[1, 0].set_title(f'Pretrained AlexNet, run {i}')
#     axs[1, 0].set_xlabel('Epoch')
#     axs[1, 0].set_ylabel('Loss')
#     data5 = torch.tensor(sample_perf_history['pretrained_cumulative_test_loss'][i])
#     data6 = torch.tensor(sample_perf_history['pretrained_cumulative_train_loss'][i])
#     axs[1, 0].plot(data5, label='Test')
#     axs[1, 0].plot(data6, label='Train')
#     axs[1, 0].legend()
#     axs[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
#     axs[1, 1].set_title(f'Non-Pretrained AlexNet, run {i}')
#     axs[1, 1].set_xlabel('Epoch')
#     axs[1, 1].set_ylabel('Loss')
#     data7 = torch.tensor(sample_perf_history['non_pretrained_cumulative_test_loss'][i])
#     data8 = torch.tensor(sample_perf_history['non_pretrained_cumulative_train_loss'][i])
#     axs[1, 1].plot(data7, label='Test')
#     axs[1, 1].plot(data8, label='Train')
#     axs[1, 1].legend()
#     # fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/comparison_run_{i}.png')
#     fig.savefig(f'/scratch/jsa378/misc_output/comparison_run_{i}.png')

# make_tarfile('/project/rrg-lelliott/jsa378/misc_output.tar', '/scratch/jsa378/misc_output')
# shutil.rmtree(f'/scratch/jsa378/misc_output')
