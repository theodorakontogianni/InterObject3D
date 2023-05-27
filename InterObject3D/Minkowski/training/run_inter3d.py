import argparse
import numpy as np
import torch
import trimesh
from interactive_adaptation.interactive_adaptation import RandomLineDataset, RandomLineDatasetS3DIS, \
    RandomLineDatasetSemKITTI, RandomLineDatasetApple

from interactive_adaptation.interactive_adaptation import InteractiveSegmentationModel
import pyviz3d.visualizer as viz

try:
    import open3d as o3d
except ImportError:
    raise ImportError('Please install open3d with `pip install open3d`.')

MAX_NUM_CLICKS = 20
MAX_IOU = 80

def dataloader(config):
    if config.dataset == 'scannet':
        train_dataset = RandomLineDataset(config)
    elif config.dataset == 's3dis':
        train_dataset = RandomLineDatasetS3DIS(config)
    elif config.dataset == 'semKITTI':
        train_dataset = RandomLineDatasetSemKITTI(config)
    elif config.dataset == 'apple':
        train_dataset = RandomLineDatasetApple(config)

    instances_id = range(config.instance_counter_id, config.instance_counter_id + config.number_of_instances)
    subset = torch.utils.data.Subset(train_dataset, instances_id)
    train_dataloader = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=1, shuffle=False)
    return train_dataloader






def get_model(device, trainable=False):
    # Model
    inseg_global = InteractiveSegmentationModel(pretraining_weights=config.pretraining_weights)
    global_model = inseg_global.create_model(device, inseg_global.pretraining_weights_file)
    if not trainable:
        global_model.eval()
    else:
        global_model.train()
    return inseg_global, global_model


def main(_):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    print(f"Using {device}")

    dataloader_test = dataloader(config)
    inseg_model_class, inseg_global_model = get_model(device)

    if config.save_results_file:
        f = open(config.results_path +
                 config.results_file_name + '_'
                 + str(config.instance_counter_id) + '_14.csv', 'w')

    for i, data in enumerate(iter(dataloader_test)):

        if config.verbal:
            print('Scene: ', data[0][0], ' object_id:', data[1][0])
        if i % 200 == 0:
            if config.save_results_file:
                f.flush()
            print(i, 'out of ', config.number_of_instances)

        scene_name = data[0][0]
        object_id = data[1][0]
        coords = data[2][0]
        gtcolors = data[4][0]
        labels = gtcolors[:, 0].float().long().to(device)
        feats = data[7][0]
        coords_qv = data[8]  # [1, num_points, 3]
        labels_qv = data[10]
        inverse_map = data[11]
        num_points = feats.size()[0]
        feats_qv = feats[inverse_map]

        pred, logits = inseg_model_class.prediction(feats.float(), coords.cpu().numpy(), inseg_global_model, device)
        iou = inseg_model_class.mean_iou(pred, labels)

        # if sum(labels)<2500:
        #    continue
        if config.verbal: print('IOU: ', iou, sum(labels))
        print(i)


        if config.real_user:
            center_coo, center_gt = inseg_model_class.get_next_click_coo_torch_real_user(coords, pred, labels, feats, 0)
        else:
            center_coo, center_gt, candidates, candidates_heat, fn, fp = inseg_model_class.get_next_simulated_click(
                pred, labels, coords_qv, labels_qv, inverse_map, inseg_model_class)

        if center_coo == None:
            num_clicks = 0
            for n in range(num_clicks + 1, 21):
                line = str(config.instance_counter_id + i) + ' ' + scene_name + ' ' + object_id + ' ' + str(
                    n) + ' ' + str(iou.cpu().numpy()) + '\n'
                num_clicks += 1
                if config.save_results_file: f.write(line)
                if config.verbal:  print('num clicks: ', num_clicks, 'IOU:  ', iou.item(), 'click :', center_coo,
                                         center_gt)

            num_clicks = 21
            continue
        # expand click region around a cube - increases info
        new_click_mask = inseg_model_class.generate_clickmask_torch(
            torch.hstack((coords[:, 0:3],
                          feats[:, 3].unsqueeze_(1) * 255)), center_coo, config.cubeedge)


        if config.save_results_file:
            line = str(config.instance_counter_id + i) + ' ' + scene_name + ' ' + object_id + ' ' + str(0) + ' ' + str(
                iou.cpu().numpy()) + '\n'
            f.write(line)

        num_clicks = 1

        while (num_clicks <= MAX_NUM_CLICKS):
            if center_gt == 1:
                # if FN add the click to the positive mask
                feats[:, 3] += new_click_mask[:, 0]
            else:
                # if FP add the click to the negative mask
                feats[:, 4] += new_click_mask[:, 0]

            # prediction with the new click

            pred, logits = inseg_model_class.prediction(feats.float(), coords.cpu().numpy(), inseg_global_model, device)

            # update prediction with sparse gt
            pos_indices = (feats[:, 3] >= 1)  # positive locations
            neg_indices = (feats[:, 4] >= 1)  # negative locations
            pred[pos_indices] = 1
            pred[neg_indices] = 0


            iou = inseg_model_class.mean_iou(pred, labels)

            if config.save_results_file:
                line = str(config.instance_counter_id + i) + ' ' + scene_name + ' ' + object_id + ' ' + str(
                    num_clicks) + ' ' + str(iou.cpu().numpy()) + '\n'
                f.write(line)
            if config.verbal:  print('num clicks: ', num_clicks, 'IOU: ', iou.item(), 'click :', center_coo, center_gt)

            if config.real_user:
                center_coo, center_gt = inseg_model_class.get_next_click_coo_torch_real_user(coords, pred, labels,
                                                                                             feats, 1)
            else:
                center_coo, center_gt, candidates, candidates_heat, fn, fp = inseg_model_class.get_next_simulated_click(
                    pred, labels, coords_qv, labels_qv, inverse_map, inseg_model_class)


            if center_coo == None:
                for n in range(num_clicks + 1, 21):
                    line = str(config.instance_counter_id + i) + ' ' + scene_name + ' ' + object_id + ' ' + str(
                        n) + ' ' + str(iou.cpu().numpy()) + '\n'
                    if config.save_results_file: f.write(line)
                    if config.verbal: print('num clicks: ', n, 'IOU: ', iou.item(), 'click :', center_coo,
                                             center_gt)

                num_clicks = 21
                continue
            new_click_mask = inseg_model_class.generate_clickmask_torch(
                torch.hstack((coords[:, 0:3], torch.zeros((num_points, 1)))), center_coo, config.cubeedge)

            num_clicks += 1

    if config.save_results_file:f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cubeedge', type=float, default=0.1)

    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument('--verbal', type=bool, default=False)
    parser.add_argument('--visual', type=bool, default=False)
    parser.add_argument('--real_user', type=bool, default=False)
    parser.add_argument('--uncertainty_based', type=bool, default=False)
    parser.add_argument('--results_path', type=str, default='./dataset_mini/results/')

    parser.add_argument('--save_results_file', type=bool, default=False)
    parser.add_argument('--results_file_name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='scannet')

    parser.add_argument('--instance_counter_id', type=int, default=0)
    parser.add_argument('--number_of_instances', type=int, default=1)

    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--pretraining_weights', type=str,
                        default='/globalwork/celikkan/scannet_official/weights/exp_14/weights_exp14_13.pth')

    parser.add_argument('--dataset_scenes', type=str,
                        default='./dataset_mini/dataset_scannet_val_mini.npy')
    parser.add_argument('--dataset_classes', type=str,
                        default='./dataset_mini/dataset_scannet_val_classes_mini.txt')
    parser.add_argument('--dataset_folder_scene', type=str, default='./dataset_mini/crops5x5/')
    parser.add_argument('--dataset_folder_masks', type=str, default='./dataset_mini/masks5x5/')

    config = parser.parse_args()

    main(config)
