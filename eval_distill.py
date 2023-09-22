import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import argparse
from tqdm import tqdm
from torchvision import transforms
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import csv
import copy

def evaluate_synthetic_data(args, image_syn, label_syn):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, _, _, _, dst_test, testloader, _, _, _ = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    args.lr_net = syn_lr.item() 
    class_acc_all = []

    for it in eval_it_pool:
        for model_eval in model_eval_pool:
            print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))
            accs_test = []
            for it_eval in range(args.num_eval):
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                _, _, acc_test, class_acc = evaluate_synset(it_eval, net_eval, image_syn, label_syn, testloader, args, texture=args.texture, num_classes=num_classes, per_class_acc=True)
                accs_test.append(acc_test)
                class_acc_all.append(class_acc)
            accs_test = np.array(accs_test)
            acc_test_mean = np.mean(accs_test)
            acc_test_std = np.std(accs_test)
            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs_test), model_eval, acc_test_mean, acc_test_std))
            print(class_acc_all)

            folder_name = f"./eval_distill_results/ipc{num_of_images}"
            print(f"Saving images at: {folder_name}")

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            with open(f'{folder_name}/class_accuracies.csv', 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                # Write header
                csv_writer.writerow(['Model'] + class_names)
                
                # Write data
                for i, row in enumerate(class_acc_all):
                    csv_writer.writerow([i+1] + row)


def return_images_and_labels(n):
    # Go to path
    base_path = f'./logged_files/{args.logged_images_path}/ipc{n}/CIFAR10/'
    
    print(f"getting the data from: {base_path}")
    # List all subdirectories in the base path
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Get the newest directory based on creation time
    newest_directory = max(dirs, key=lambda d: os.path.getctime(os.path.join(base_path, d)))

    # New path = path + newest folder
    new_path = os.path.join(base_path, newest_directory)
    
    # Return paths
    images = os.path.join(new_path, 'images_best.pt')
    labels = os.path.join(new_path, 'labels_best.pt')
    print("loading images and labels from:")
    print(images)
    print(labels)
    return torch.load(images) , torch.load(labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument('--save_path', type=str, default="", help='the path were the logged files will be saved')
    parser.add_argument('--logged_images_path', type=str, default="", help="which expermint to load data from")

    args = parser.parse_args()

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None
    for num_of_images in [1, 10, 50]:
        D_images, D_labes = return_images_and_labels(num_of_images)
        evaluate_synthetic_data(args, D_images, D_labes)
