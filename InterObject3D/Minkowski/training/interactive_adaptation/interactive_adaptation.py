import torch
import numpy as np

import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
import pickle
import io
from examples.minkunet import MinkUNet34C

try:
    import open3d as o3d
except ImportError:
    raise ImportError('Please install open3d with `pip install open3d`.')

USE_TRAINING_CLICKS = False


def load_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    return coords, colors, pcd

class RandomLineDataset(Dataset):

    def __init__(self, config, quantization_size=0.05):

        dataset_train = np.load(config.dataset_scenes) # np.load('/globalwork/kontogianni/intobjseg/datasets/scannet_official/dataset_scannet_val.npy')
        if config.label:
            dataset_classes = np.loadtxt(config.dataset_classes, dtype=str) # np.loadtxt('/globalwork/kontogianni/intobjseg/datasets/scannet_official/dataset_scannet_val_classes2.txt', dtype=str)
            dataset_train =  dataset_train[dataset_classes==config.label]
        print('Dataset val size ', len(dataset_train))

        self.dataset_folder_scene = config.dataset_folder_scene # '/globalwork/celikkan/scannet_official/crops5x5/'
        self.dataset_folder_masks = config.dataset_folder_masks # '/globalwork/celikkan/scannet_official/masks5x5/'


        self.dataset_train = dataset_train
        self.dataset_size = len(dataset_train)
        self.quantization_size = quantization_size
        self.config = config

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):

        if self.config.debug:
            scene_name = self.config.scene
            object_id= self.config.obj
        else:
            scene_name = self.dataset_train[i,0].replace('scene','')
            object_id = self.dataset_train[i,1]

        file_name = self.dataset_folder_scene + 'scene' + scene_name + '/scene' + scene_name + '_crop_' + object_id + '.ply'

        pc_file_name = self.dataset_folder_masks + 'scene' + scene_name + '/scene' + scene_name + '_mask_' + object_id + '_pc.ply'
        nc_file_name = self.dataset_folder_masks + 'scene' + scene_name + '/scene' + scene_name + '_mask_' + object_id + '_nc.ply'
        gt_file_name = self.dataset_folder_masks + 'scene' + scene_name + '/scene' + scene_name + '_mask_' + object_id + '.ply'

        coords, scenecolors, pcd = load_file(file_name)
        pccoords, pccolors, pcpcd = load_file(pc_file_name)
        nccoords, nccolors, ncpcd = load_file(nc_file_name)
        gtcoords, gtcolors, gtpcd = load_file(gt_file_name)




        feats = self.create_model_input(scenecolors, pccolors, nccolors, USE_TRAINING_CLICKS)

        unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coords,
            quantization_size=self.quantization_size,
            return_index=True,
            ignore_label=-100)

        coords_qv = coords[inverse_map]
        feats_qv = feats[inverse_map]
        labels_qv = gtcolors[:,0][inverse_map]

        return scene_name, object_id, coords, scenecolors, gtcolors, pccolors, nccolors, feats, coords_qv,feats_qv,labels_qv, inverse_map

    def create_model_input(self, scenecolors, pccolors, nccolors, use_training_clicks=False):
        feats = np.column_stack((scenecolors, pccolors[:, 0], nccolors[:, 0]))
        if not use_training_clicks:
            feats[:, 3:5] = np.zeros((feats.shape[0], 2))
        else:
            print('*Using training clicks for debug!*')
        return feats

class RandomLineDatasetS3DIS(RandomLineDataset):

    def __init__(self, config, quantization_size=0.05):

        RandomLineDataset.__init__(self, config, quantization_size=quantization_size)
        print('Evaluate S3DIS')
        dataset_train = np.load(config.dataset_scenes)#'/globalwork/kontogianni/intobjseg/datasets/stanford_s3dis/Area_5/dataset_area5.npy')
        if config.label:
            dataset_classes = np.loadtxt(config.dataset_classes, dtype=str)#'/globalwork/kontogianni/intobjseg/datasets/stanford_s3dis/Area_5/dataset_area5_classes.txt', dtype=str)
            dataset_train = dataset_train[dataset_classes==config.label]
        print('Dataset val size ', len(dataset_train))

        self.dataset_folder_scene = config.dataset_folder_scene #'/globalwork/kontogianni/intobjseg/datasets/stanford_s3dis/Area_5/crops5x5/'
        self.dataset_folder_masks = config.dataset_folder_masks #'/globalwork/kontogianni/intobjseg/datasets/stanford_s3dis/Area_5/cubeedge_005/masks5x5/'


        self.dataset_train = dataset_train
        self.dataset_size = len(dataset_train)
        self.quantization_size = quantization_size
        self.config = config
    def __getitem__(self, i):

        if self.config.debug:
            scene_name = self.config.scene
            object_id= self.config.obj
        else:
            scene_name = self.dataset_train[i,0].replace('scene','')
            object_id = self.dataset_train[i,1]
        file_name = self.dataset_folder_scene + '/' + scene_name + '/' + scene_name + '_crop_' + object_id + '.ply'

        pc_file_name = self.dataset_folder_masks + '/' + scene_name + '/' + scene_name + '_mask_' + object_id + '_pc.ply'
        nc_file_name = self.dataset_folder_masks + '/' + scene_name + '/' + scene_name + '_mask_' + object_id + '_nc.ply'
        gt_file_name = self.dataset_folder_masks + '/' + scene_name + '/' + scene_name + '_mask_' + object_id + '.ply'

        coords, scenecolors, pcd = load_file(file_name)
        pccoords, pccolors, pcpcd = load_file(pc_file_name)
        nccoords, nccolors, ncpcd = load_file(nc_file_name)
        gtcoords, gtcolors, gtpcd = load_file(gt_file_name)


        feats = self.create_model_input(scenecolors, pccolors, nccolors, USE_TRAINING_CLICKS)

        unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coords,
            quantization_size=self.quantization_size,
            return_index=True,
            ignore_label=-100)

        coords_qv = coords[inverse_map]
        feats_qv = feats[inverse_map]
        labels_qv = gtcolors[:,0][inverse_map]

        return scene_name, object_id, coords, scenecolors, gtcolors, pccolors, nccolors, feats, coords_qv,feats_qv,labels_qv, inverse_map

class RandomLineDatasetSemKITTI(RandomLineDataset):
    def __init__(self, config, quantization_size=0.05):

        RandomLineDataset.__init__(self, config, quantization_size=quantization_size)
        print('Evaluate SemanticKITTI')
        dataset_train = np.load(config.dataset_scenes)#'/globalwork/kontogianni/intobjseg/datasets/SemanticKITTI/01/dataset_01_f.npy')
        if config.label:
            dataset_classes = np.loadtxt(config.dataset_classes, dtype=str)
               #'/globalwork/kontogianni/intobjseg/datasets/SemanticKITTI/01/dataset_01_classes_f.txt', dtype=str)
            dataset_train = dataset_train[dataset_classes == config.label]
        print('Dataset val size ', len(dataset_train))

        self.dataset_folder_scene = config.dataset_folder_scene# '/globalwork/kontogianni/intobjseg/datasets/SemanticKITTI/01/crops5x5/'
        self.dataset_folder_masks = config.dataset_folder_masks#'/globalwork/kontogianni/intobjseg/datasets/SemanticKITTI/01/masks5x5/'

        self.dataset_train = dataset_train
        self.dataset_size = len(dataset_train)
        self.quantization_size = quantization_size
        self.config = config

    def __getitem__(self, i):

        if self.config.debug:
            scene_name = self.config.scene
            object_id= self.config.obj
        else:
            scene_name = self.dataset_train[i,0].replace('scene','')
            object_id = self.dataset_train[i,1]
        file_name = self.dataset_folder_scene + '/' + scene_name + '/' + scene_name + '_crop_' + object_id + '.ply'


        #pc_file_name = self.dataset_folder_masks + '/' + scene_name + '/' + scene_name + '_mask_' + object_id + '_pc.ply'
        #nc_file_name = self.dataset_folder_masks + '/' + scene_name + '/' + scene_name + '_mask_' + object_id + '_nc.ply'
        gt_file_name = self.dataset_folder_masks + '/' + scene_name + '/' + scene_name + '_mask_' + object_id + '.ply'

        coords, scenecolors, pcd = load_file(file_name)
        #pccoords, pccolors, pcpcd = load_file(pc_file_name)
        #nccoords, nccolors, ncpcd = load_file(nc_file_name)
        gtcoords, gtcolors, gtpcd = load_file(gt_file_name)


        feats = self.create_model_input(scenecolors, scenecolors, scenecolors, USE_TRAINING_CLICKS)

        unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coords,
            quantization_size=self.quantization_size,
            return_index=True,
            ignore_label=-100)

        coords_qv = coords[inverse_map]
        feats_qv = feats[inverse_map]
        labels_qv = gtcolors[:,0][inverse_map]

        return scene_name, object_id, coords, scenecolors, gtcolors, scenecolors, scenecolors, feats, coords_qv,feats_qv,labels_qv, inverse_map


class RandomLineDatasetApple(RandomLineDataset):
    def __init__(self, config, quantization_size=0.05):

        RandomLineDataset.__init__(self, config, quantization_size=quantization_size)
        print('Evaluate Apple')
        dataset_train = np.load(config.dataset_scenes)#'/globalwork/kontogianni/intobjseg/datasets/SemanticKITTI/01/dataset_01_f.npy')
        if config.label:
            dataset_classes = np.loadtxt(config.dataset_classes, dtype=str)
               #'/globalwork/kontogianni/intobjseg/datasets/SemanticKITTI/01/dataset_01_classes_f.txt', dtype=str)
            dataset_train = dataset_train[dataset_classes == config.label]
        print('Dataset val size ', len(dataset_train))

        self.dataset_folder_scene = config.dataset_folder_scene# '/globalwork/kontogianni/intobjseg/datasets/SemanticKITTI/01/crops5x5/'
        self.dataset_folder_masks = config.dataset_folder_masks#'/globalwork/kontogianni/intobjseg/datasets/SemanticKITTI/01/masks5x5/'

        self.dataset_train = dataset_train
        self.dataset_size = len(dataset_train)
        self.quantization_size = quantization_size
        self.config = config

    def __getitem__(self, i):

        if self.config.debug:
            scene_name = self.config.scene
            object_id= self.config.obj
        else:
            scene_name = self.dataset_train[i,0].replace('scene','')
            object_id = self.dataset_train[i,1]
        file_name = self.dataset_folder_scene + '/' + scene_name + '/' + scene_name + '_crop_' + object_id + '.ply'
        file_name = '/home/dora/remote/rwth/data/intobjseg/datasets/apple/47331653/47331653_3dod_mesh.ply'
        #pc_file_name = self.dataset_folder_masks + '/' + scene_name + '/' + scene_name + '_mask_' + object_id + '_pc.ply'
        #nc_file_name = self.dataset_folder_masks + '/' + scene_name + '/' + scene_name + '_mask_' + object_id + '_nc.ply'
        gt_file_name = self.dataset_folder_masks + '/' + scene_name + '/' + scene_name + '_mask_' + object_id + '.ply'

        coords, scenecolors, pcd = load_file(file_name)
        #pccoords, pccolors, pcpcd = load_file(pc_file_name)
        #nccoords, nccolors, ncpcd = load_file(nc_file_name)
        #gtcoords, gtcolors, gtpcd = load_file(gt_file_name)
        gtcolors=coords

        feats = self.create_model_input(scenecolors, scenecolors, scenecolors, USE_TRAINING_CLICKS)

        unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coords,
            quantization_size=self.quantization_size,
            return_index=True,
            ignore_label=-100)

        coords_qv = coords[inverse_map]
        feats_qv = feats[inverse_map]
        labels_qv = gtcolors[:,0][inverse_map]

        return scene_name, object_id, coords, scenecolors, gtcolors, scenecolors, scenecolors, feats, coords_qv,feats_qv,labels_qv, inverse_map


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class InteractiveSegmentationModel(object):

    def __init__(self, pretraining_weights='/globalwork/celikkan/scannet_official/weights/exp_14/weights_exp14_13.pth'):

        self.pretraining_weights_file = pretraining_weights


    def create_model(self, device, pretrained_weights_file=None):
        model = MinkUNet34C(in_channels=5, out_channels=2, D=3).to(device)
        if pretrained_weights_file:
            #  Get weights
            weights = pretrained_weights_file
            print('weights', weights)
            if pretrained_weights_file:
                #  Get weights
                if not torch.cuda.is_available():
                    #model_dict = CPU_Unpickler(weights).load()
                    map_location = 'cpu'
                    print('Cuda not found, using CPU')
                    model_dict = torch.load(weights, map_location)  # which one is correct??The 14

                else:
                    map_location = None
                    model_dict = torch.load(weights, map_location)  # which one is correct??The 14
            model.load_state_dict(model_dict)
           # print('Pretrained weights loaded.')
        return model


    def add_optimiser(self, model, config):
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False)
        return criterion, optimizer


    def create_model_input(self, scenecolors, pccolors, nccolors, use_training_clicks=False):
        feats = np.column_stack((scenecolors, pccolors[:, 0], nccolors[:, 0]))
        if not use_training_clicks:
            feats[:, 3:5] = np.zeros((feats.shape[0], 2))
        else:
            print('*Using training clicks for debug!*')
        return feats

    def prediction(self, feats, coords, model, device):
        #with torch.no_grad():
        voxel_size = 0.05
        # Feed-forward pass and get the prediction
        sinput = ME.SparseTensor(
            features=feats,
            coordinates=ME.utils.batched_coordinates([coords / voxel_size]),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=device

        )  # .to(device)
        model.eval()
        logits = model(sinput).slice(sinput)
        # get the prediction on the input tensor field
        # out_field = soutput.slice(in_field)
        logits = logits.F
        _, pred = logits.max(1)

        # pred = pred.cpu().numpy()
        return pred, logits

    def mean_iou(self, pred, labels):
        intersection = labels * pred
        truepositive = intersection.sum()
        union = torch.logical_or(labels, pred)
        union = torch.sum(union)
        iou = 100 * (truepositive / union)
        # print('IOU', iou)
        return iou

    def visualize_open3d(self,coords, feats, click, labels):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector((feats[:, 0:3]).cpu().numpy())
        #pcd.paint_uniform_color([0.7, 0.7, 0.7])



        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 1.0, 1.0])

        l = torch.min(coords[labels==1],axis=0).values.cpu().numpy()
        m = torch.max(coords[labels==1],axis=0).values.cpu().numpy()

        bbox1 = o3d.geometry.AxisAlignedBoundingBox(min_bound=l-0.5, max_bound=m+0.5)

        bbox2 = o3d.geometry.AxisAlignedBoundingBox(min_bound=click-0.05, max_bound=click+0.05)
        bbox2.color = (1, 0, 0)
        bbox3 = o3d.geometry.AxisAlignedBoundingBox(min_bound=click-0.025, max_bound=click+0.025)
        bbox3.color = (0, 1, 0)
        bbox4 = o3d.geometry.AxisAlignedBoundingBox(min_bound=click-0.005, max_bound=click+0.005)
        bbox4.color = (0, 0, 1)
        #bbox1 = o3d.geometry.AxisAlignedBoundingBox(min_bound=(4.0, 2.6, 0.2), max_bound=(4.7, 3.3, 0.85))
        #bbox1.color = (0, 1, 0)
        #bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(4.3, 3.1, 0.52), max_bound=(4.4, 3.2, 0.62))
        #bbox.color = (1, 0, 0)

        pcd_crop=pcd.crop(bbox1)

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd)
        #
        # ctr = vis.get_view_control()
        # parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2022-02-18-17-54-42.json")
        # ctr.convert_from_pinhole_camera_parameters(parameters)
        # print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
        # ctr.change_field_of_view(step=50)
        # print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
        # vis.run()


        # vis=o3d.visualization.draw_geometries([pcd,bbox2,bbox3,bbox4],
        #                                   zoom=0.1,
        #                                   front=[5.9,3.8,0.77],
        #                                   lookat=[4.4, 3.9, 0.47],
        #                                   up=[5.4, 3.5, 0.77],
        #                                   point_show_normal=False)

        #vis = o3d.visualization.VisualizerWithEditing()

        #vis.create_window()
        #vis.add_geometry(pcd)
        #vis.run()  # user picks points
        return pcd_crop, pcd.normals



    def pick_points(self, pcd, prediction):
        print("")
        print(
            "1) Please pick one correspondence using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        #o3d.visualization.draw_geometries([pcd, prediction])

        #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.03)
        #mesh.compute_vertex_normals()
        import open3d.visualization.gui as gui


        vis = o3d.visualization.VisualizerWithEditing()

        vis.create_window()
        #vis.add_geometry(pcd)
        vis.get_render_option().point_size = 10.5

        vis.add_geometry(prediction)
        #vis.add_geometry(mesh)

        vis.run()  # user picks points
        label=0

        #vis=o3d.visualization.draw_geometries_with_key_callbacks()
        #key=o3d.visualization.gui.KeyEvent()
        #vis.get_render_option()key_to_callback[ord("K")] = change_background_to_black

        #vis.register_key_callback(ord("P") ,label=1)
        #print(key)
        vis.destroy_window()
        print("")

        if len(vis.get_picked_points())==2:
            return  vis.get_picked_points()[1],0
        else:return vis.get_picked_points()[0],1

    def get_next_click_coo_torch_real_user(self, coords, pred, gt, feats, num_clicks):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector((feats[:, 0:3]).cpu().numpy())

        prediction = o3d.geometry.PointCloud()
        pred_rgb = np.zeros(np.shape(coords[:, 0:3].cpu().numpy()))
        if num_clicks==0:
            color_gt = np.array((254, 190, 29))
            #pred_rgb[:, 0] = 254 * (gt.cpu().numpy())
            #pred_rgb[:, 1] = 190 * (gt.cpu().numpy())
            #pred_rgb[:, 0] = 29 * (gt.cpu().numpy())
            #feats[:, 0:3][gt>0]=0
        else:
            pred_rgb[:, 1] = 255 * (pred.cpu().numpy())
        prediction.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
        prediction.colors = o3d.utility.Vector3dVector(pred_rgb+(feats[:, 0:3]).cpu().numpy())


        picked_point, gt_point = self.pick_points(pcd, prediction)
        if not picked_point:
            return None,None
        center_id=picked_point
        #return coords[center_id], gt[center_id]

        return coords[center_id], gt_point

    def get_next_click_coo_torch(self, discrete_coords, unique_labels, gt, pred):

        zero_indices = (unique_labels == 0)  # background
        one_indices = (unique_labels == 1)  # foreground
        if zero_indices.sum() == 0 or one_indices.sum() == 0:
            return None, None, -1, None, None

        # All distances from foreground points to background points
        pairwise_distances = torch.cdist(discrete_coords[zero_indices, :], discrete_coords[one_indices, :])
        # Bg points on the border
        pairwise_distances, _ = torch.min(pairwise_distances, dim=0)
        # point furthest from border
        center_id = torch.where(pairwise_distances == torch.max(pairwise_distances, dim=0)[0])
        center_coo = discrete_coords[one_indices, :][center_id[0][0]]
        center_label = gt[one_indices][center_id[0][0]]
        center_pred = pred[one_indices][center_id[0][0]]
        #print('center_pred', center_pred, center_label)

        candidates = discrete_coords[one_indices, :]
        candidates_heat = []
        max_dist = torch.max(pairwise_distances)

        for i in pairwise_distances:
            candidates_heat.append([255 * i / max_dist, 0, 0])
        return center_coo, center_label, max_dist, candidates, candidates_heat

    def generate_clickmask_torch(self, vertices_pc, center_coo, cubeedge=0.05):

        refx, refy, refz = center_coo

        vertices_pc[torch.logical_and(torch.logical_and(
            (torch.abs(vertices_pc[:, 0] - refx) < cubeedge),
            (torch.abs(vertices_pc[:, 1] - refy) < cubeedge))
            , (torch.abs(vertices_pc[:, 2] - refz) < cubeedge)), 3] = 1

        # print('click mask', np.shape(vertices_pc[:,3]), vertices_pc[:,3].sum())
        return vertices_pc[:, 3].unsqueeze_(1)

    def sample_user_input(self, user_input, coords, feats):
        n_points = len(user_input)
        sampled_input = torch.zeros((feats.shape[0], 1))
        if n_points>1:
            nsamples = np.random.randint(1, n_points)
            sample_indices = np.random.choice(range(n_points), nsamples, replace=False)
            sampled_points = np.array(user_input)[sample_indices]
            #print('number of user input points: ',n_points)
            #print('number of sampled use input points',nsamples)
            for i in range(nsamples):
                #print(sampled_points[i])
                sampled_input += self.generate_clickmask_torch(
                    torch.hstack((coords[:, 0:3],
                                  torch.zeros((feats.shape[0], 1))* 255,
                                  )),
                    torch.tensor(sampled_points[i])
                )
                #print('-----',i)
                #print(torch.nonzero(sampled_input).shape)
                #print(torch.nonzero(feats[:, 3]).shape)
        elif n_points==1:
            #print(user_input[0])
            sampled_input += self.generate_clickmask_torch(
                torch.hstack((coords[:, 0:3],
                              torch.zeros((feats.shape[0], 1)) * 255,
                        )),
                torch.tensor(user_input[0])
            )
        else:
            return sampled_input
        #print('-----')
        #print(torch.nonzero(sampled_input).shape)
        #print(torch.nonzero(feats[:,3]).shape)
        return sampled_input

    def get_next_simulated_click_dense(self, pred, labels, coords, inseg_model_class):
        fn = torch.logical_and(torch.logical_xor(pred, labels), labels)  # FN
        fp = torch.logical_and(torch.logical_xor(pred, labels), pred)  # FP
        # get next positive click candidate
        pcenter_coo, pcenter_gt, pmax_dist, candidates_p, candidates_p_heat = inseg_model_class.get_next_click_coo_torch(
            coords, fn,
            labels,pred)
        # get next negative click candidate
        ncenter_coo, ncenter_gt, nmax_dist, candidates_n, candidates_n_heat = inseg_model_class.get_next_click_coo_torch(
            coords, fp,
            labels,pred)
        if pmax_dist >= nmax_dist:
            center_coo = pcenter_coo
            center_gt = pcenter_gt
            candidates = candidates_p
            candidates_heat = candidates_p_heat
        else:
            center_coo = ncenter_coo
            center_gt = ncenter_gt
            candidates = candidates_n
            candidates_heat = candidates_n_heat
        return center_coo, center_gt, candidates, candidates_heat, fn, fp

    def get_next_simulated_click(self, pred, labels, coords_qv, labels_qv, inverse_map, inseg_model_class):
        fn = torch.logical_and(torch.logical_xor(pred, labels), labels)  # FN
        fp = torch.logical_and(torch.logical_xor(pred, labels), pred)  # FP

        # get next positive click candidate
        pcenter_coo, pcenter_gt, pmax_dist, candidates_p, candidates_p_heat = inseg_model_class.get_next_click_coo_torch(
            coords_qv, fn[inverse_map],
            labels_qv, pred[inverse_map])
        # get next negative click candidate
        ncenter_coo, ncenter_gt, nmax_dist, candidates_n, candidates_n_heat = inseg_model_class.get_next_click_coo_torch(
            coords_qv, fp[inverse_map],
            labels_qv, pred[inverse_map])
        if pmax_dist >= nmax_dist:
            center_coo = pcenter_coo
            center_gt = pcenter_gt
            candidates = candidates_p
            candidates_heat = candidates_p_heat
        else:
            center_coo = ncenter_coo
            center_gt = ncenter_gt
            candidates = candidates_n
            candidates_heat = candidates_n_heat

        return center_coo, center_gt, candidates, candidates_heat, fn, fp

    def get_next_uncertain_click(self, pred, labels, coords, logits, used_pixels):
        fn = torch.logical_and(torch.logical_xor(pred, labels), labels)  # FN
        fp = torch.logical_and(torch.logical_xor(pred, labels), pred)  # FP

        correct = torch.logical_and(labels, pred)
        unca = torch. logits[:, 0] - logits[:, 1]
        unca[correct]=1000

        for used_pixel in used_pixels:

            unca[used_pixel]=1000
            refx, refy, refz = coords[used_pixel][0]
            cubeedge=0.05

            unca[torch.logical_and(torch.logical_and(
                (torch.abs(coords[:, 0] - refx) < cubeedge),
                (torch.abs(coords[:, 1] - refy) < cubeedge))
                , (torch.abs(coords[:, 2] - refz) < cubeedge))] = 1000


        k1 = 1000
        candidate_vals, candidate_ids = torch.topk(unca, k=k1, largest=False)
        best_candidate = candidate_ids[torch.randint(1, k1, (1, 1))][0]
        used_pixels.append(best_candidate)
        center_coo = coords[best_candidate[0]]
        center_gt = labels[best_candidate[0]]

        print(unca[best_candidate])
        print(logits[best_candidate, 0],logits[best_candidate, 1])


        #center_coo=torch.tensor([-35.6070,   1.3940,   0.7920])
        #center_gt = torch.tensor(0)
        #used_pixels.append(best_candidate.item())


        candidates = coords[candidate_ids, :]
        candidates_heat = []
        max_dist = torch.max(candidate_vals)

        for i in candidate_vals:
            candidates_heat.append([255 * i.item() / max_dist.item(), 0, 0])
        return center_coo, center_gt, candidates, candidates_heat, fn, fp


class InteractiveSegmentationModelAdaptive(InteractiveSegmentationModel):

    def __init__(self, pretraining_weights='/globalwork/celikkan/scannet_official/weights/exp_14/weights_exp14_13.pth'):

        self.pretraining_weights_file = pretraining_weights
        InteractiveSegmentationModel.__init__(self, pretraining_weights)

    def create_model(self, device, config, pretrained_weights_file=None):
        model = MinkUNet34C(in_channels=5, out_channels=2, D=3).to(device)
        if pretrained_weights_file:
            #  Get weights
            weights = pretrained_weights_file
            if not torch.cuda.is_available():
                # model_dict = CPU_Unpickler(weights).load()
                map_location = 'cpu'

            else:
                map_location = None
            model_dict = torch.load(weights, map_location)  # which one is correct??The 14
            model.load_state_dict(model_dict)
            #print('Pretrained weights loaded.')

        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False)
        #optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

        return model, criterion, optimizer

    def prediction(self, feats, coords, model, device):
        voxel_size = 0.05

        #with torch.no_grad():
        # Feed-forward pass and get the prediction
        sinput = ME.SparseTensor(
            features=feats,
            coordinates=ME.utils.batched_coordinates([coords / voxel_size]),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=device

        )  # .to(device)
        #model.eval()
        #model.zero_grad()
        logits = model(sinput).slice(sinput)
        #model.eval()
        #model.zero_grad()

        # get the prediction on the input tensor field
        # out_field = soutput.slice(in_field)
        logits = logits.F
        _, pred = logits.max(1)

    #    pred = pred.cpu().numpy()
        #coords=torch.Tensor(coords)
        #print(feats.size(),coords.size())

        # coords = ME.utils.sparse_quantize(
        #     coordinates=coords / voxel_size,
        #     # feats=feats,
        #     ignore_label=-100)
        # input = ME.SparseTensor(features=feats.float(), coordinates=coords.float(), device=device)
        # logits = model(input)
        # logits = logits.F
        # _, pred = logits.max(1)
        return pred, logits