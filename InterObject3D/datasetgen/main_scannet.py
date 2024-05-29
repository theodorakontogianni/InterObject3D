import numpy as np
import pyviz3d.visualizer as viz
import os
import os.path as osp
from plyfile import PlyData, PlyElement
import json
import open3d as o3d
from pathlib import Path
from absl import app
from absl import flags
import random
import math

FLAGS = flags.FLAGS
flags.DEFINE_string('path', '/media/dora/Samsung_T5/intobjseg/datasets/scannet_official//scans/', 'Path to 3D scenes')
flags.DEFINE_string('name', 'scene0000_00', 'Name of the scene.')
flags.DEFINE_string('output_dir', '/media/dora/Samsung_T5/intobjseg/datasets/scannet_official/results/', 'Where to write generated scenes.')

def main(_):

    #scene_name = 'scene0140_01'
    scene_name = FLAGS.name

    if scene_name[:5] == 'scene':

        vertices = read_mesh_vertices_rgb(FLAGS.path + '/' + scene_name + '/' + scene_name + '_vh_clean_2.ply')
        vertices_labels = read_mesh_vertices_rgb(FLAGS.path + '/' + scene_name + '/' + scene_name + '_vh_clean_2.labels.ply')
        object_id_to_segs, label_to_segs = read_aggregation(FLAGS.path + '/' + scene_name + '/' + scene_name +  '.aggregation.json')
        seg_to_verts, num_verts = read_segmentation(FLAGS.path + '/' + scene_name + '/' + scene_name +  '_vh_clean_2.0.010000.segs.json')

        #Path('/globalwork/celikkan/scannet_official/masks/' + scene_name).mkdir(parents=True, exist_ok=True)
        Path(FLAGS.output_dir +'masks5x5/'+ scene_name).mkdir(parents=True, exist_ok=True)
        Path(FLAGS.output_dir +'crops5x5/'+ scene_name).mkdir(parents=True, exist_ok=True)


        for i in range(len(object_id_to_segs)):

            # Binary mask
            vertices2 = np.copy(vertices)

            seg_temp = object_id_to_segs[i+1]
            v_temp = []
            for x in seg_temp:
                v_temp.append(seg_to_verts[x])

            vertices2[:, 3:6] = 0
            for j in v_temp:
                vertices2[j, 3:6] = 255


            # 5mx5mxeverything
            # mask5x5 = aabb5x5(vertices2)
            mask5x5 = aabb5x5(vertices2,vertices)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mask5x5[:,0:3])
            pcd.colors = o3d.utility.Vector3dVector(mask5x5[:,3:6]/255)
            output_name = FLAGS.output_dir + 'crops5x5/' + scene_name + '/' + scene_name + '_crop_' + str(i) + '.ply'
            o3d.io.write_point_cloud(output_name, pcd)

            mask5x5 = aabb5x5(vertices2, vertices2)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mask5x5[:,0:3])
            pcd.colors = o3d.utility.Vector3dVector(mask5x5[:,3:6]/255)
            output_name = FLAGS.output_dir + "masks5x5/" +scene_name + '/' + scene_name + '_mask_' + str(i) + '.ply'
            o3d.io.write_point_cloud(output_name, pcd)



            # Clicks
            edge = 0.05#0.1
            factor = 1.5
            pc_num = random.randint(1,5)
            nc_num = random.randint(1,15)

            vertices_pc = pc_gen(mask5x5, pc_num, edge)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices_pc[:, 0:3])
            pcd.colors = o3d.utility.Vector3dVector(vertices_pc[:, 3:6] / 255)
            output_name = FLAGS.output_dir + 'masks5x5/' + scene_name + '/' + scene_name + '_mask_' + str(i) + '_pc.ply'
            o3d.io.write_point_cloud(output_name, pcd)

            vertices_nc = nc_gen(mask5x5, nc_num, edge, factor)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices_nc[:, 0:3])
            pcd.colors = o3d.utility.Vector3dVector(vertices_nc[:, 3:6] / 255)
            output_name = FLAGS.output_dir + 'masks5x5/' + scene_name + '/' + scene_name + '_mask_' + str(i) + '_nc.ply'
            o3d.io.write_point_cloud(output_name, pcd)

        print('Done ' + scene_name)

    else:
        print('Invalid scene name:' + scene_name)

def pyviz3d(scene):
    #Visualization
    # First, we set up a visualizer
    v = viz.Visualizer()
    # Random point clouds.
    for j in range(5):
        i = j + 1
        name = 'Points_'+str(i)
        num_points = 3
        point_positions = np.random.random(size=[num_points, 3])
        point_colors = (np.random.random(size=[num_points, 3]) * 255).astype(np.uint8)
        point_size = 25 * i
        # Here we add point clouds to the visualizer
        v.add_points(name, point_positions, point_colors, point_size=point_size, visible=False)
    for scene_name in ['room']:
        #scene = vertices
        point_positions = scene[:, 0:3]
        point_colors = scene[:, 3:6]
        point_size = 25.0
        # Add more point clouds
        v.add_points(scene_name, point_positions, point_colors, point_size=point_size)
        # v.add_points(scene_name, point_positions, point_size=point_size)
    # When we added everything we need to the visualizer, we save it.
    v.save('test')

def read_mesh_vertices_rgb(filename):
    """read XYZ and RGB for each vertex."""
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["red"]
        vertices[:, 4] = plydata["vertex"].data["green"]
        vertices[:, 5] = plydata["vertex"].data["blue"]
    return vertices

def read_mesh_vertices(filename):
    """read XYZ for each vertex."""
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
    return vertices

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data["segGroups"])
        for i in range(num_objects):
            object_id = data["segGroups"][i]["objectId"] + 1  # instance ids should be 1-indexed
            label = data["segGroups"][i]["label"]
            segs = data["segGroups"][i]["segments"]
            object_id_to_segs[object_id] = segs
            # if label in label_to_segs:
            #     label_to_segs[label].extend(segs)
            # else:
            #     label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs

def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data["segIndices"])
        for i in range(num_verts):
            seg_id = data["segIndices"][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def aabb5x5(vertices2, vertices):
    alpha = 5 #meters
    vertices_aabb = np.copy(vertices2)
    vertices_aabb2 = np.copy(vertices)

    object_only = np.delete(vertices_aabb, np.where(vertices_aabb[:, 3] == 0)[0], axis=0)
    xmax = np.amax(object_only[:,0])
    xmin = np.amin(object_only[:,0])
    xmid = (xmax + xmin) / 2
    xmax = xmid + (alpha / 2)
    xmin = xmid - (alpha / 2)
    ymax = np.amax(object_only[:,1])
    ymin = np.amin(object_only[:,1])
    ymid = (ymax + ymin) / 2
    ymax =ymid + (alpha / 2)
    ymin = ymid - (alpha / 2)

    mask5x5 = np.delete(vertices_aabb2, np.where((vertices_aabb2[:, 0] > xmax) | (vertices_aabb2[:, 0] < xmin) | (vertices_aabb2[:, 1] > ymax) | (vertices_aabb2[:, 1] < ymin))[0], axis=0)
    return mask5x5

# def aabb5x5(vertices):
#     alpha = 5 #meters
#     vertices_aabb = np.copy(vertices)
#     vertices_aabb2 = np.copy(vertices)
#
#     object_only = np.delete(vertices_aabb, np.where(vertices_aabb[:, 3] == 0)[0], axis=0)
#     xmax = np.amax(object_only[:,0])
#     xmin = np.amin(object_only[:,0])
#     xmid = (xmax + xmin) / 2
#     xmax = xmid + (alpha / 2)
#     xmin = xmid - (alpha / 2)
#     ymax = np.amax(object_only[:,1])
#     ymin = np.amin(object_only[:,1])
#     ymid = (ymax + ymin) / 2
#     ymax =ymid + (alpha / 2)
#     ymin = ymid - (alpha / 2)
#
#     mask5x5 = np.delete(vertices_aabb2, np.where((vertices_aabb2[:, 0] > xmax) | (vertices_aabb2[:, 0] < xmin) | (vertices_aabb2[:, 1] > ymax) | (vertices_aabb2[:, 1] < ymin))[0], axis=0)
#     return mask5x5

def pc_gen(vertices, pc_num, cubeedge):
    vertices_pc = np.copy(vertices)
    samplingset = np.where(vertices_pc[:,5] == 255)[0]
    #Positive Clicks
    if samplingset.size > 0:
        i = 0
        while (i < pc_num):
            m = np.random.choice(samplingset, 1)
            refx = vertices_pc[m, 0]
            refy = vertices_pc[m, 1]
            refz = vertices_pc[m, 2]
            vertices_pc[(abs(vertices_pc[:, 0] - refx) < cubeedge) & (abs(vertices_pc[:, 1] - refy) < cubeedge) & (abs(vertices_pc[:, 2] - refz) < cubeedge), 4] = 0
            i += 1
        vertices_pc[(vertices_pc[:, 4] == 255), 3:6] = 0
        vertices_pc[(vertices_pc[:, 3] == 255), 3:6] = 255
        return vertices_pc
    else:
        return vertices # !!!


def nc_gen(vertices, nc_num, cubeedge, aabbfactor):
    # alpha = ( math.sqrt(aabbfactor) - 1) / 2
    alpha = 0.2
    vertices_aabb = np.copy(vertices)

    object_only = np.delete(vertices_aabb, np.where(vertices_aabb[:, 3] == 0)[0], axis=0)
    if object_only.size > 0:
        # generate negative clicks
        xmax = np.amax(object_only[:, 0])
        xmin = np.amin(object_only[:, 0])
        dx = xmax - xmin
        xmax = xmax + (alpha * dx)
        xmin = xmin - (alpha * dx)
        ymax = np.amax(object_only[:, 1])
        ymin = np.amin(object_only[:, 1])
        dy = ymax - ymin
        ymax = ymax + (alpha * dy)
        ymin = ymin - (alpha * dy)
        vertices_aabb[((vertices_aabb[:, 0] <= xmax) & (vertices_aabb[:, 0] >= xmin) &
                       (vertices_aabb[:, 1] <= ymax) & (vertices_aabb[:, 1] >= ymin) ) , 3:6] = 255
        vertices_nc = np.copy(vertices)
        vertices_nc[:, 3:6] = vertices_aabb[:, 3:6] - vertices[:, 3:6]

        #Negative clicks
        samplingset = np.where(vertices_nc[:, 5] == 255)[0]
        if samplingset.size > 0:
            i = 0
            while (i < nc_num):
                m = np.random.choice(samplingset, 1)
                refx = vertices_nc[m, 0]
                refy = vertices_nc[m, 1]
                refz = vertices_nc[m, 2]
                vertices_nc[(abs(vertices_nc[:, 0] - refx) < cubeedge) & (abs(vertices_nc[:, 1] - refy) < cubeedge) & (abs(vertices_nc[:, 2] - refz) < cubeedge), 4] = 0
                i += 1
            vertices_nc[(vertices_nc[:, 4] == 255), 3:6] = 0
            vertices_nc[(vertices_nc[:, 3] == 255), 3:6] = 255
            return vertices_nc
        else:
            return vertices

    else:
        return vertices  # !!!

if __name__ == '__main__':
    app.run(main)

