import os
import os.path as osp
os.environ['LANG']='en_US'
#os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import trimesh
import mcubes
import pyrender
from pyglet import gl
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from glob import glob

def ellipse(i, num, x_max, y_max):
    theta = i / (num-1) * math.pi * 2
    x = math.cos(theta) * x_max
    y = math.sin(theta) * y_max
    return x, y

def main(folder, name, size=256):
    sigma_threshold = 0.5 
    num = 60
    voxel_grid = np.load(f'{folder}/{name}_voxel.npy')
    voxel_grid = np.maximum(voxel_grid, 0)
    vertices, triangles = mcubes.marching_cubes(voxel_grid, sigma_threshold)
    mesh = trimesh.Trimesh(vertices/size, triangles)
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    for i in range(num):
        yaw, pitch = ellipse(i, num, -0.5, 0.25)
        ambient = -30
        scene = pyrender.Scene(ambient_light=[ambient, ambient, ambient], bg_color=[255, 255, 255])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 180.0 * 12)
        light = pyrender.DirectionalLight(color=[255,255,255], intensity=10)

        scene.add(mesh, pose=  np.eye(4))
        scene.add(light, pose=  np.eye(4))

        distance = 4.5
        x = math.sin(yaw) * math.cos(pitch) * distance
        y = math.sin(pitch) * distance
        z = math.cos(yaw) * math.cos(pitch) * distance

        rotate_y = np.array([[ math.cos(yaw),  0,  math.sin(yaw)],
                             [ 0,              1,              0],
                             [-math.sin(yaw),  0,  math.cos(yaw)]])
        rotate_x = np.array([[ 1,  0,  0],
                             [ 0,  math.cos(pitch), -math.sin(pitch)],
                             [ 0,  math.sin(pitch),  math.cos(pitch)]])
        rotate_xy = np.matmul(rotate_y, rotate_x)
        RT = np.array([[ 1,  0,  0,  0.5+x],
                       [ 0,  1,  0,  0.5-y],
                       [ 0,  0,  1,  0.5+z],
                       [ 0,  0,  0,  1]])
        RT[:3,:3] = rotate_xy
        scene.add(camera, pose=RT)

        # render scene
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(scene)

        im = Image.fromarray(color)
        margin = 60
        im = im.crop((margin, margin, 512-margin, 512-margin))
        im.save(f"{folder}/{name}_{i:03d}_crop.png")

# exe: PYOPENGL_PLATFORM=osmesa python xxx.py
if __name__ == '__main__':
    folder = 'ours/bfm_demo_512'
    names = sorted(glob(osp.join(folder, '*.npy')))
    print("#files:", len(names))
    for name in names:
        name = name.split('/')[-1][:-10]
        main(folder, name, size=512)
