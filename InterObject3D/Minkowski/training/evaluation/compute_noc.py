
import numpy as np


def load_results(results_file, label=None, MAX_IOU=90, dataset=None, dataset_classes=None,
                 exclude_classes=['wall', 'ceiling', 'floor', 'unlabelled', 'unlabeled']):
    objects = {}

    def process_objects(data):
        nonlocal objects
        for entry in data:
            objects[entry[0].replace('scene', '') + '_' + entry[1]] = 1

    def filter_objects_by_classes(dataset_, dataset_classes, exclude_classes):
        mask = np.isin(dataset_classes, exclude_classes, invert=True)
        print('total number of objects: ', np.shape(dataset_classes))
        print('number of objects kept: ', sum(mask))
        return dataset_[mask], dataset_classes[mask]

    if exclude_classes:
        dataset, dataset_classes = filter_objects_by_classes(dataset, dataset_classes, exclude_classes)
        process_objects(dataset)
    if label is not None:
        dataset = dataset[dataset_classes == label]
        objects = {}
        process_objects(dataset)
        print('number of objects kept from class ', label, ": ", len(objects))
    else:
        process_objects(dataset)
        print('number of objects kept: ', len(objects))

    results_dict_KatIOU = {}
    num_objects = 0
    ordered_clicks = []
    all_object = {}
    results_dict_per_click = {}
    results_dict_per_click_iou = {}
    all_data = {}

    with open(results_file, 'r') as f:
        for line in f:
            splits = line.rstrip().split(' ')
            scene_name = splits[1].replace('scene', '')
            object_id = splits[2]
            num_clicks = splits[3]
            iou = splits[4]

            obj_key = scene_name + '_' + object_id

            if obj_key in objects:
                all_object.setdefault(obj_key, 1)
                all_data.setdefault(obj_key, []).append((num_clicks, iou))

                if float(iou) >= MAX_IOU:
                    if obj_key not in results_dict_KatIOU:
                        results_dict_KatIOU[obj_key] = float(num_clicks)
                        num_objects += 1
                        ordered_clicks.append(float(num_clicks))
                elif int(num_clicks) >= 20 and float(iou) >= 0:
                    if obj_key not in results_dict_KatIOU:
                        results_dict_KatIOU[obj_key] = float(num_clicks)
                        num_objects += 1
                        ordered_clicks.append(float(num_clicks))


                results_dict_per_click.setdefault(num_clicks, 0)
                results_dict_per_click_iou.setdefault(num_clicks, 0)

                results_dict_per_click[num_clicks] += 1
                results_dict_per_click_iou[num_clicks] += float(iou)

    if not results_dict_KatIOU:
        print('no objects to evaluate')
        return 0

    click_at_80 = sum(results_dict_KatIOU.values()) / len(results_dict_KatIOU.values())
    print('click@', MAX_IOU, click_at_80, num_objects, len(all_object), len(results_dict_KatIOU))
    clicks = [str(i) for i in range(21)]
    iou = [results_dict_per_click_iou[i] / results_dict_per_click[i] for i in clicks]

    for obj_key, data in all_data.items():
        if len(data) != 21:
            pass  # Handle incomplete data if needed

    return ordered_clicks, clicks, iou


if __name__ == '__main__':


    datasets = ['scannet'] #'s3dis','semKITTI'


    if 'scannet' in datasets:

        label_scannet_benchmark = ['wall', 'floor', 'cabinet',
                                   'bed',
                                   'chair',
                                   'sofa',
                                   'table',
                                   'door',
                                   'window',
                                   'bookshelf',
                                   'picture',
                                   'counter',
                                   'desk',
                                   'curtain',
                                   'refridgerator',
                                   'shower',
                                   'curtain',
                                   'toilet',
                                   'sink',
                                   'bathtub',
                                   'otherfurniture']
        label_scannet_all = {
            'unlabelled', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
            'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow',
            'mirror', 'floormat', 'clothes', 'ceiling', 'books', 'refrigerator', 'television', 'paper',
            'towel', 'showercurtain', 'box', 'whiteboard', 'person', 'nightstand', 'toilet', 'sink', 'lamp',
            'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop'
        }

        path = "./results/scannet_val/"
        base_005_005 = 'baseline_all2_005_005_0_14.csv'
        dataset_scannet = np.load(
            './results/scannet_val/dataset_scannet_val.npy')
        dataset_classes_scannet = np.loadtxt(
            './results/scannet_val/dataset_scannet_val_classes2.txt',
            dtype=str)

        print('scannet')
        for iou_max in [80, 85, 90]:
            print('\t All classes')

            _, clicks_scannet, iou_scannet = load_results(path + base_005_005, None, iou_max, dataset_scannet,
                                                          dataset_classes_scannet, None)
            print('\t Seen classes')

            load_results(path + base_005_005, None, iou_max, dataset_scannet, dataset_classes_scannet, exclude_classes=list(set(label_scannet_all)-set(label_scannet_benchmark)))

            print('\t UNSeen classes')

            load_results(path + base_005_005, None, iou_max, dataset_scannet, dataset_classes_scannet, exclude_classes=list(['unlabelled']+label_scannet_benchmark))


            # print('Class Bed')
            # _, clicks_scannet, iou_scannet = load_results(path + base_005_005, 'bed', iou_max, dataset_scannet,
            #                                                dataset_classes_scannet, None)
