import argparse
import numpy as np
import pickle
import torch
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME

from examples.minkunet import MinkUNet34C
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter

read_path = '/media/dora/Samsung_T5/intobjseg/datasets/scannet_official/'
write_path = '/media/dora/Samsung_T5/intobjseg/datasets/scannet_official/results/'

def load_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    return coords, colors, pcd

class RandomLineDataset(Dataset):

    def __init__(
        self,
        quantization_size=0.005, dataset='examples/dataset_train.npy', restrict_training_classes=False):

        dataset_train = np.load(dataset)
        # Remove objects of certain classes from training
        #if restrict_training_classes:
        #    training_classes = np.loadtxt('examples/scannet_benchmark_labelids.txt', dtype=str, delimiter=' ')
        #    dataset_classes = np.loadtxt('examples/dataset_train_classes.txt', dtype=str)
        #    dataset_train=dataset_train[np.isin(dataset_classes,training_classes[:,1])]
        self.dataset_train = dataset_train
        self.dataset_size = len(dataset_train)
        self.quantization_size = quantization_size


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):

        #always the same scene!!!!!!i For debugging
        # coords, colors, pcd = load_file(config.file_name)
        # labelscoords, labelscolors, labelspcd = load_file(config.mask_file_name)
        # pccoords, pccolors, pcpcd = load_file(config.pc_file_name)
        # nccoords, nccolors, ncpcd = load_file(config.nc_file_name)


        scene_name = self.dataset_train[i,0]
        object_id = self.dataset_train[i,1]
        coords, colors, pcd = load_file(read_path+'crops5x5/' + scene_name + '/' + scene_name + '_crop_' + object_id + '.ply')
        labelscoords, labelscolors, labelspcd = load_file(read_path +'masks5x5/' + scene_name + '/' + scene_name + '_mask_' + object_id + '.ply')
        pccoords, pccolors, pcpcd = load_file(read_path +'masks5x5/' + scene_name + '/' + scene_name + '_mask_' + object_id + '_pc.ply')
        nccoords, nccolors, ncpcd = load_file(read_path +'masks5x5/' + scene_name + '/' + scene_name + '_mask_' + object_id + '_nc.ply')

        feats = np.column_stack((colors, pccolors[:,0], nccolors[:,0])) #RGB + positive clicks + negative clicks
        labels = labelscolors[:,0]
        labels = labels.astype(np.int32)
        # labels = torch.from_numpy(labels)

        voxel_size = 0.05
        # Quantize the input
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=coords/voxel_size,
            features=feats,
            labels=labels,
            #quantization_size=self.quantization_size,
            ignore_label=-100)

        return discrete_coords, unique_feats, unique_labels


def collation_fn(data_labels):
    coords, feats, labels = list(zip(*data_labels))
    coords_batch, feats_batch, labels_batch = [], [], []

    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0))

    return coords_batch, feats_batch, labels_batch

# def evaluate(device, net, criterion):
#
#     print('Start evaluate on val')
#     prev_epoch_count = 0
#     check = True
#     accum_loss, accum_iter, tot_validation_iter = 0, 0, 0
#     truepositive, mintersection, union = 0, 0, 0
#
#     # Dataset, data loader - validation dataset
#     eval_dataset = RandomLineDataset(dataset='examples/dataset_val.npy')
#     eval_dataloader = DataLoader(
#         eval_dataset,
#         batch_size=config.batch_size,
#         shuffle=False,
#         collate_fn=ME.utils.batch_sparse_collate,
#         num_workers=1)
#
#     writer = SummaryWriter(config.save_writer)
#
#
#     eval_iter = iter(eval_dataloader)
#
#     net.eval()
#     for i, data in enumerate(eval_iter):
#         coords, feats, labels = data
#         input = ME.SparseTensor(
#             feats.float(),
#             coords,
#             quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE, device=device)
#
#         out = net(input)#.slice(input)
#         #_, pred = logits.max(1)
#
#         _, pred = torch.max(out.F.squeeze(), 1)
#
#         pred = pred.cpu().numpy()
#
#         intersection = labels*pred
#         truepositive += sum(intersection==1)
#         union += sum(pred==1) + sum(labels==1) - sum(intersection==1)
#
#         out = net(input)
#         loss = criterion(out.F.squeeze(), labels.long().to(device))
#         accum_loss += loss.item()
#         accum_iter += 1
#         tot_validation_iter += 1
#
#
#
#     # Writer: loss and mIoU
#     print(
#         print(f'Total Validation Iter: {tot_validation_iter}, Validation Loss: {accum_loss / accum_iter}, Validation mIoU: {(100 * mintersection / union).item()}')
#     )
#
#     writer.add_scalar('Validation Loss',
#                       accum_loss / accum_iter,
#                       tot_validation_iter) # for axis as epochs: tot_iter=100 (100: dataset_size / batch_size)
#     writer.add_scalar('Validation mIoU',
#                       (100 * truepositive / union).item(),
#                       tot_validation_iter)
#
#     accum_loss, accum_iter, truepositive, union = 0, 0, 0, 0

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using {device}")
    print(device)
    #print(torch.cuda.current_device())
    #print(torch.cuda.get_device_name(0))

    # Binary mask generation
    net = MinkUNet34C(in_channels=5, out_channels=2, D=3).to(device)
    net = net.to(device)
    # If use pre-training weights
    if os.path.exists(config.weights):
        model_dict = torch.load(config.weights)
        model_dict['final.bias'] = torch.randn(1, 2, dtype=torch.float32)
        model_dict['final.kernel'] = torch.randn(96, 2, dtype=torch.float32)
        model_dict['conv0p1s1.kernel'] = torch.randn(125, 5, 32, dtype=torch.float32)
        net.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Dataset, data loader
    train_dataset = RandomLineDataset(restrict_training_classes=config.restrict_training_classes)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        # 1) collate_fn=collation_fn,
        # 2) collate_fn=ME.utils.batch_sparse_collate,
        # 3) collate_fn=ME.utils.SparseCollation(),
        collate_fn=ME.utils.batch_sparse_collate,
        num_workers=1)

    accum_loss, accum_iter, tot_iter = 0, 0, 0
    intersection, mintersection, union = 0, 0, 0

    writer = SummaryWriter(config.save_writer)
    net.train()
    grad_dict_total = {k: 0 for k, v in net.named_parameters()}
    N_samples = 0

    for epoch in range(config.max_epochs):
        train_iter = iter(train_dataloader)

        # Training
        net.train()
        for i, data in enumerate(train_iter):
            coords, feats, labels = data
            
            input = ME.SparseTensor(feats.float(), coords, device=device)#.to(device)
            out = net(input)
            try:
                optimizer.zero_grad()
                loss = criterion(out.F.squeeze(), labels.long().to(device))
                loss.backward(retain_graph=True)
                optimizer.step()
                print(out.shape)
            except (RuntimeError, MemoryError) as error:  # Handle OOM
                print(f'+++ Recovering from exception: {error} +++')
                torch.cuda.empty_cache()
                continue

            # Save MAS weights https://arxiv.org/pdf/1711.09601.pdf

            # Zero the parameter gradients
            optimizer.zero_grad()

            # get the function outputs
            outputs = net(input)

            # compute the sqaured l2 norm of the function outputs
            l2_norm = torch.norm(outputs.F.squeeze(), 2, dim=1)

            squared_l2_norm = l2_norm ** 2

            sum_norm = torch.sum(squared_l2_norm)

            # compute gradients for these parameters
            sum_norm.backward()

            grad_dict_total = {k:( grad_dict_total[k] + torch.abs(v.grad))  for k, v in net.named_parameters()}
            N_samples+=1

            #grad_dict_e = {k:( grad_dict_total[k]/N_samples)  for k, v in net.named_parameters()}
            #with open(write_path + 'weights/exp_14_limited_classes/mas_weights'  + '_' + str(epoch) +'.pkl', 'wb') as f:
            #    pickle.dump(grad_dict_e, f)
            #pass



            accum_loss += loss.item()
            accum_iter += 1
            tot_iter += 1

            _, pred = torch.max(out.F.squeeze(), 1)
            truepositive = pred*labels.to(device)
            intersection = torch.sum(truepositive==1)

            mintersection += intersection
            union += torch.sum(pred==1) + torch.sum(labels==1) - intersection
            lr = optimizer.param_groups[0]['lr']

            if tot_iter % 1 == 0 or tot_iter == 1:
                print(
                    f'Epoch: {epoch} iter: {tot_iter}, Loss: {accum_loss / accum_iter}, mIoU: {(100 * mintersection / union).item()}, lr: {lr} '
                )
                #evaluate(device,net,criterion)
                if i % 200 == 0:
                    writer.add_scalar('Training Loss',
                                      accum_loss / accum_iter,
                                      tot_iter)
                    writer.add_scalar('Training mIoU',
                                      100 * mintersection / union,
                                      tot_iter)
                    # writer.add_scalar('Learning rate',
                    #                   lr,
                    #                   tot_iter)
                    accum_loss, accum_iter, mintersection, union = 0, 0, 0, 0

        if (epoch % 5 == 0)or(epoch==0):
            #evaluate(device, net, criterion)
            grad_dict_e = {k:( grad_dict_total[k]/N_samples)  for k, v in net.named_parameters()}
            #with open(write_path + 'weights/exp_14_limited_classes/mas_weights'  + '_' + str(epoch) +'.pkl', 'wb') as f:
            #    pickle.dump(grad_dict_e, f)
            #pass
            torch.save(net.state_dict(), write_path + 'weights/exp_14_limited_classes/' + config.save_weights + '_' + str(epoch+6) + '.pth')

    torch.save(net.state_dict(), write_path + 'weights/exp_14_limited_classes/' + config.save_weights + '_' + str(epoch+6) + '.pth')
    #grad_dict_e = {k: (grad_dict_total[k] / N_samples) for k, v in net.named_parameters()}
    #with open(write_path + 'weights/exp_14_limited_classes/mas_weights' + '_' + str(epoch) + '.pkl', 'wb') as f:
    #    pickle.dump(grad_dict_e, f)
    print('Training Complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--max_epochs', default=15, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--weights', type=str, default='/home/celikkan/Scan2BIM/Minkowski/weights.pth')

    parser.add_argument('--restrict_training_classes', type=bool, default=True)

    parser.add_argument('--save_weights', type=str, default='weights_exp14')
    parser.add_argument('--save_writer', type=str, default='runs/experiment14')

    parser.add_argument('--file_name', type=str, default='/data/scannet_official/crops5x5/scene0191_01/scene0191_01_crop_16.ply')
    parser.add_argument('--mask_file_name', type=str, default='/data/scannet_official/masks5x5/scene0191_01/scene0191_01_mask_16.ply')
    parser.add_argument('--pc_file_name', type=str, default='/data/scannet_official/masks5x5/scene0191_01/scene0191_01_mask_16_pc.ply')
    parser.add_argument('--nc_file_name', type=str, default='/data/scannet_official/masks5x5/scene0191_01/scene0191_01_mask_16_nc.ply')

    config = parser.parse_args()
    main(config)
