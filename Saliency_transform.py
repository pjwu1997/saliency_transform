# %%
import torch
from model.resnet import ResNet18
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import random
import copy
from captum.attr import Saliency
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import random_split
from datetime import datetime
import json
from dataset import MyCIFAR10, MyCIFAR100
import os


config_file = open('config.json')
config = json.load(config_file)
file_path = config["file_path"]
job_type = config["job_type"]
augment_prob = config["augment_prob"] # Proportion of augment samples
num_epoch = config["num_epoch"] # Total epochs
pretrain_epoch = config["pretrain_epoch"] # Perform saliency calculation after pretrain epoch
interval = config["interval"] # Perform saliency calculation every interval steps
threshold = config["threshold"] # Proportion of saliency
img_size = config["img_size"] # Proportion of image_size(one sided) 
repeat_times = config["repeat_times"]
early_stop = config["early_stop"]

model = ResNet18(num_classes=10).cuda()
# optimizer = Adam(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# %%
# Define transformation, this is for cifar10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])

# %%
# Get Dataset
# trainset = datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform_train)
# testset = datasets.CIFAR10(root='./data', train=False,
#                                         download=True, transform=transform)
# trainset = datasets.CIFAR100(root='./data', train=True,
#                                         download=True, transform=transform_train)

# testset = datasets.CIFAR100(root='./data', train=False,
#                                         download=True, transform=transform)
trainset = MyCIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = MyCIFAR10(root='./data', train=False, download=True, transform=transform)

train_ds, val_ds = random_split(trainset, [45000, 5000])

trainLoader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
valLoader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
testLoader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

saliency_dict = {} # id: (left, right, bottom, floor)
# %%
# Functions
def get_saliency(interpreter, inputs, target):
    """ Perform integrated gradient for saliency.
    """
    # define saliency interpreter
    # interpreter = captum.attr.Saliency(net)
    attribution = interpreter.attribute(inputs, target)
    attribution = torch.squeeze(attribution)
    return torch.sum(attribution, 0)

def update_all_saliency(interpreter, dataloader):
    global saliency_dict
    saliency_dict = {}
    total_count = 0
    for ((input_images, targets), id_list) in dataloader:
        input_images = input_images.cuda()
        targets = targets.cuda()
        total_count += len(id_list)
        for i, input_image in enumerate(input_images):
            saliency_map = get_saliency(interpreter, input_image.unsqueeze(0), targets[i])
            saliency_indices = get_salient_region(saliency_map, img_size, threshold)
            id = id_list[i].item()
            if saliency_indices:
                saliency_dict[id] = saliency_indices
    print(f'Update results: {len(saliency_dict)/len(dataloader)}% are in')
    return len(saliency_dict) / len(dataloader)
                

def get_salient_region(input_saliency, length=None, threshold=0.5):
    """
    input saliency -> 2-dim gray-scale image
    """
    shape = input_saliency.shape
    if length == None:
        length = int(((threshold + 0.1) * shape[0]) // 2)
    # max_saliency = torch.max(input_saliency)
    # min_saliency = torch.min(input_saliency)
    # normalized_saliency = (input_saliency - min_saliency) / (max_saliency - min_saliency)
    
    total_saliency = torch.sum(input_saliency)
    # print(shape)
    max_positions = np.unravel_index(torch.argsort(torch.flatten(input_saliency), descending=True).cpu(), shape)
    # print(max_positions)
    for cnt, (x_position, y_position) in enumerate(zip(max_positions[0], max_positions[1])):
        # print(x_position)
        if cnt >= 15:
            return None
        if x_position - length < 0:
            left = 0
            right = left + length * 2
        elif (x_position + length) > shape[0] - 1:
            right = shape[0]
            left = right - length * 2
        else:
            left = x_position - length
            right = left + length * 2
        if y_position - length < 0:
            bottom = 0
            floor = bottom + length * 2
        elif (y_position + length) > shape[1] - 1:
            floor = shape[1]
            bottom = floor - length * 2
        else:
            bottom = y_position - length
            floor = bottom + length * 2
        region = input_saliency[left:right, bottom:floor]
        if torch.sum(region) / total_saliency > threshold:
            return (left, right, bottom, floor)
    return None

def augment_cover(input_images, targets, id_list, prob=0.3):
    """Cover the salienct region by some extent

    Args:
        input_image (_type_): _description_
        saliency_indices (_type_): _description_
    """
    # print('Hi')
    global saliency_dict
    matrix = None
    for index in range(len(id_list)):
        saliency_indices = None
        id = id_list[index].item()
        if id in saliency_dict:
            saliency_indices = saliency_dict[id]
        if saliency_indices:
            left, right, bottom, floor = saliency_indices
            if matrix is None:
                matrix = prob * torch.ones((right - left, floor - bottom))
                matrix = torch.bernoulli(matrix).cuda() ## matrix contain only 0,1
            else:
                # left, right, bottom, floor = saliency_indices
                input_images[index, :, left:right, bottom:floor] *= matrix
    return input_images
                  
def baseline(input_images, targets, id_list):
    return input_images

def augment_noise(input_images, targets, id_list):
    #print('Hi')
    global saliency_dict
    image_shape = input_images[0].shape
    for index in range(len(id_list)):
        saliency_indices = None
        id = id_list[index].item()
        if id in saliency_dict:
            saliency_indices = saliency_dict[id]
        if saliency_indices:
                left, right, bottom, floor = saliency_indices
                matrix = torch.randn(image_shape) * 0.1
                matrix[:, left:right, bottom:floor] = 0
                # print(matrix.shape)
                input_images[index, :] += matrix.cuda()
    return input_images

def augment_scale(input_images, targets, id_list, scale_range=[0.5, 1.5]):
    scale_proportion = scale_range[0] + np.random.rand() * (scale_range[1] - scale_range[0])
    shape = input_images[0].shape[1:]
    p = None
    length = None
    for index in range(len(id_list)):
        saliency_indices = None
        id = id_list[index].item()
        if id in saliency_dict:
            saliency_indices = saliency_dict[id]
        if saliency_indices:
            left, right, bottom, floor = saliency_indices
            cropped_image = input_images[index, :, left:right, bottom:floor]
            if length is None:
                length = int(scale_proportion * (right - left))
                if length % 2 == 1:
                    length -= 1
            if p is None:
                p = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((length, length)),
                    transforms.ToTensor()])
            cropped_image = p(cropped_image.cpu()).cuda()
            center = (int((left + right)//2), int((bottom + floor) // 2))
            new_left = center[0] - length // 2
            new_right = center[0] + length // 2
            new_bottom = center[1] - length // 2
            new_floor = center[1] + length // 2
            if new_left < 0:
                new_left = 0
                new_right = new_left + length
            elif new_right > shape[0]:
                new_right = shape[0]
                new_left = new_right - length
            if new_bottom < 0:
                new_bottom = 0
                new_floor = new_bottom + length
            elif new_floor > shape[1]:
                new_floor = shape[1]
                new_bottom = new_floor - length
            print(cropped_image.shape)
            input_images[index, :, new_left:new_right, new_bottom:new_floor] = cropped_image
    return input_images

def train(dataloader, net, criterion, optimizer,scheduler, augment_fn_list, augment_prob = 0.2):
    n_correct = 0
    n_total = 0
    loss_list = []
    net.train()
    for i, ((input_images, targets), id_list) in enumerate(dataloader):
        input_images = input_images.cuda()
        targets = targets.cuda()           
        if np.random.rand() >= augment_prob:
            # print(123)
            pass
        else:
            augment_fn = random.choice(augment_fn_list)
            input_images = augment_fn(input_images, targets, id_list).cuda()
        n_total += len(targets)
        pred = net(input_images)
        n_correct += (pred.argmax(dim=-1) == targets).sum().item()
        loss = criterion(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().item())
    print(f'Training accuracy: {n_correct/n_total}, loss: {np.mean(loss_list)}')
    scheduler.step()
    return n_correct/n_total

def validate(dataloader, net, criterion, type='Validation'):
    net.eval()
    n_correct = 0
    n_total = 0
    loss_list = []
    for (input_images, targets), id_list in dataloader:
        input_images = input_images.cuda()
        targets = targets.cuda()
        n_total += len(targets)
        pred = net(input_images)
        n_correct += (pred.argmax(dim=-1) == targets).sum().item()
        loss = criterion(pred, targets)
        loss_list.append(loss.detach().item())
    print(f'{type} accuracy: {n_correct/n_total}, loss: {np.mean(loss_list)}')
    return n_correct/n_total
    
augment_fn_list = []
for method in job_type:
    augment_fn_list.append(globals()[method])

def main():
    now = datetime.now()
    current_time = now.strftime("%d-%m-%Y-%H-%M-%S")
    for times in range(repeat_times):
        file_name = f'./{file_path}/{current_time}/{times}/result.csv'
        file_name_2 = f'./{file_path}/{current_time}/{times}/result_saliency.csv'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        current_best_val = 0
        current_test_acc = 0
        early_stop_cnt = 0
        with open(file_name, 'w') as f, open(file_name_2, 'w') as f2:
            f.write(f'{current_time}\n')
            json.dump(config, f, indent=4)
            json.dump(config, f2, indent=4)
            f.write('\n')
            f2.write('\n')
            f.write('Epoch,Training,Validation,Test\n')
            f.write('Epoch,Prob\n')
            for epoch in range(num_epoch):
                print(epoch)
                if epoch % interval == 0 and epoch >= pretrain_epoch:
                    interpreter = Saliency(model)
                    result = update_all_saliency(interpreter, trainLoader)
                    f2.write(f'{epoch},{result}\n')
                    f2.flush()
                if epoch < pretrain_epoch:
                    train_acc = train(trainLoader, model, criterion, optimizer, scheduler, augment_fn_list, augment_prob=0)
                else:
                    train_acc = train(trainLoader, model, criterion, optimizer, scheduler, augment_fn_list, augmet_prob=augment_prob)
                val_acc = validate(valLoader, model, criterion, 'Validation')
                test_acc = validate(testLoader, model, criterion, 'Test')
                f.write(f'{epoch},{train_acc},{val_acc},{test_acc}\n')
                f.flush()
                # Early stop mechanism
                if val_acc > current_best_val:
                    early_stop_cnt = 0
                    current_best_val = val_acc
                    current_test_acc = test_acc
                    torch.save(model.state_dict(), f'./{file_path}/{current_time}/{times}/best.pt')
                else:
                    early_stop_cnt += 1
                if early_stop_cnt > early_stop:
                    f.write(f'{current_best_val}, {current_test_acc}\n')
                    break
                f.write(f'{current_best_val}, {current_test_acc}\n')
                

if __name__ == '__main__':
    main()
                    
            
            
            