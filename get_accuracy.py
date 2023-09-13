import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug
import copy
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    args.dsa = True if args.dsa == 'True' else False
    args.dsa_param = ParamDiffAug()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    D_images, D_labes = None, None
    print("DD = ", args.DD_files)
    if args.DD_files:
        D_images, D_labes = return_images_and_labels()
    (channel, im_size, num_classes, class_names, mean, std, dst_train,
     dst_test, testloader, loader_train_dict, class_map,
     class_map_inv) = get_dataset(args.dataset, args.data_path,
                                args.batch_real, args.subset, args=args, D_images=D_images, D_labes=D_labes)

    print('Hyper-parameters: \n', args.__dict__)

    trainloader = DataLoader(dataset=dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0, drop_last=False)


    ''' Train the model '''
    model = get_network(args.model, channel, num_classes, im_size).to(args.device) 
    model.train()
    lr = args.lr_teacher
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
    optimizer.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    lr_schedule = [args.train_epochs // 2 + 1]

    

    for e in range(args.train_epochs):
        train_loss, train_acc = epoch("train", dataloader=trainloader, net=model, optimizer=optimizer,
                                    criterion=criterion, args=args, aug=True)

        test_loss, test_acc = epoch("test", dataloader=testloader, net=model, optimizer=None,
                                    criterion=criterion, args=args, aug=False)

        if e in lr_schedule and args.decay:
            lr *= 0.1
            teacher_optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
            teacher_optim.zero_grad()
        print("Epoch: {}\tTrain Acc: {} \tTest Acc: {}".format(e, train_acc, test_acc))



    ''' Evaluate per class accuracy '''
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    #print("class names: ", class_names)
    #print("class correct: ", class_correct)
    #print("class total:", class_total)
    for i in range(num_classes):
        print('Accuracy of %5s : %2d %%' % (class_names[i], 100 * class_correct[i] / class_total[i]))


###

def return_images_and_labels():
    # Go to path
    base_path = './logged_files/CIFAR10/'
    
    # List all subdirectories in the base path
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Get the newest directory based on creation time
    newest_directory = max(dirs, key=lambda d: os.path.getctime(os.path.join(base_path, d)))

    # New path = path + newest folder
    new_path = os.path.join(base_path, newest_directory)
    
    # Return paths
    return os.path.join(new_path, 'images_best.pt'), os.path.join(new_path, 'labels_best.pt')


###


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=1)

    #parser.add_argument('--data_file', type=str, default=None, help="Path to the data file")
    #parser.add_argument('--label_file', type=str, default=None, help="Path to the labels files")
    # combined into a better way:
    parser.add_argument('--DD_files', type=bool, default=False, help="used DD files")

    args = parser.parse_args()
    main(args)


#python get_accuracy.py --DD_files=True