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

def main(folder, name, size=256):
    sigma_threshold = 0.5
    voxel_grid = np.load(f'{folder}/{name}_voxel.npy')
    voxel_grid = np.maximum(voxel_grid, 0)
    vertices, triangles = mcubes.marching_cubes(voxel_grid, sigma_threshold)
    mesh = trimesh.Trimesh(vertices/size, triangles)
    mesh.export(f'{folder}/{name}.obj')
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    view_phi = [-0.5, -0.25, 0, 0.25, 0.5]

    for i, phi in enumerate(view_phi):
        ambient = -30
        scene = pyrender.Scene(ambient_light=[ambient, ambient, ambient], bg_color=[255, 255, 255])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 180.0 * 12)
        light = pyrender.DirectionalLight(color=[255,255,255], intensity=10)

        scene.add(mesh, pose=  np.eye(4))
        scene.add(light, pose=  np.eye(4))

        distance = 4.5
        x = math.sin(phi) * distance
        z = math.cos(phi) * distance
        scene.add(camera, pose=[[ math.cos(phi),  0,  math.sin(phi), 0.5+x],
                                [ 0,              1,              0, 0.5],
                                [-math.sin(phi),  0,  math.cos(phi), 0.5+z],
                                [ 0,  0,  0,  1]])

        # render scene
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(scene)

        im = Image.fromarray(color)
        margin = 60
        im = im.crop((margin, margin, 512-margin, 512-margin))
        im.save(f"{folder}/{name}_{sigma_threshold}_{i}_crop.png")

# exe: PYOPENGL_PLATFORM=osmesa python xxx.py
if __name__ == '__main__':
    folder = 'ours/BFM_occ'
    names = sorted(glob(osp.join(folder, '*.npy')))
    print("#files:", len(names))
    for name in names:
        name = name.split('/')[-1][:-10]
        main(folder, name, size=256)
