import os
import sys
import cv2
import numpy as np
import os.path as osp
from skimage import io
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision import transforms


def generate_grid(h, w):
    x = np.linspace(-1.0, 1.0, w)
    y = np.linspace(-1.0, 1.0, h)
    xy = np.meshgrid(x, y)

    grid = torch.tensor([xy]).float()
    grid = grid.permute(0, 2, 3, 1)
    return grid


def depth2voxel(img_depth, img_sem, gsize):
    gsize = torch.tensor(gsize).int()
    n, c, h, w = img_depth.size()
    site_z = img_depth[:, 0, int(h/2), int(w/2)] + 3.0
    voxel_sitez = site_z.view(n, 1, 1, 1).expand(n, gsize, gsize, gsize)

    # depth voxel
    grid_mask = generate_grid(gsize, gsize)
    grid_mask = grid_mask.expand(n, gsize, gsize, 2)
    # grid_depth = torch.nn.functional.grid_sample(img_depth, grid_mask,align_corners=True)
    grid_depth = torch.nn.functional.grid_sample(img_depth, grid_mask, mode='nearest', align_corners=True)
    voxel_depth = grid_depth.expand(n, gsize, gsize, gsize)
    voxel_depth = voxel_depth - voxel_sitez

    # semantic voxel
    grid_s = torch.nn.functional.grid_sample(img_sem, grid_mask, mode='nearest', align_corners=True)
    voxel_s = grid_s.expand(n, gsize, gsize, gsize)
    k = int(gsize/2) + 1
    voxel_s = grid_s.expand(n, k, gsize, gsize)
    gound_s = torch.zeros([n, gsize-k,gsize, gsize], dtype=torch.float) # set as ground
    voxel_s = torch.cat((gound_s, voxel_s.cpu()), 1)

    # occupancy voxel
    voxel_grid = torch.arange(-gsize/2, gsize/2, 1).float()
    voxel_grid = voxel_grid.view(1, gsize, 1, 1).expand(n, gsize, gsize, gsize)
    voxel_ocupy = torch.ge(voxel_depth, voxel_grid).float().cpu()
    voxel_ocupy[:,gsize-1,:,:] = 0
    voxel_ocupy = voxel_ocupy

    # distance voxel
    voxel_dx = grid_mask[0,:,:,0].view(1,1,gsize,gsize).expand(n,gsize,gsize,gsize).float()*float(gsize/2.0)
    voxel_dy = grid_mask[0,:,:,1].view(1,1,gsize,gsize).expand(n,gsize,gsize,gsize).float()*float(gsize/2.0)
    voxel_dz = voxel_grid

    voxel_dis = voxel_dx.mul(voxel_dx) + voxel_dy.mul(voxel_dy) + voxel_dz.mul(voxel_dz)
    voxel_dis = voxel_dis.add(0.01)   # avoid 1/0 = nan
    voxel_dis = voxel_dis.mul(voxel_ocupy)
    voxel_dis = torch.sqrt(voxel_dis) - voxel_ocupy.add(-1.0).mul(float(gsize)*0.9)

    voxel_s = voxel_s.mul(voxel_ocupy) - voxel_ocupy.add(-1.0).mul(255)
    return voxel_dis, voxel_s


def voxel2pano(voxel_dis, voxel_s, size_pano, ori=torch.Tensor([0])):
    PI = 3.1415926535
    r, c = [size_pano[0], size_pano[1]]
    n, s, t, tt = voxel_dis.size()
    k = int(s/2)

    # rays
    ori = ori.view(n, 1).expand(n, c).float()
    x = torch.arange(0, c, 1).float().view(1, c).expand(n, c)
    y = torch.arange(0, r, 1).float().view(1, r).expand(n, r)
    lon = x * 2 * PI/c + ori - PI
    lat = PI/2.0 - y * PI/r
    sin_lat = torch.sin(lat).view(n, 1, r, 1).expand(n, 1, r, c)
    cos_lat = torch.cos(lat).view(n, 1, r, 1).expand(n, 1, r, c)
    sin_lon = torch.sin(lon).view(n, 1, 1, c).expand(n, 1, r, c)
    cos_lon = torch.cos(lon).view(n, 1, 1, c).expand(n, 1, r, c)
    vx =  cos_lat.mul(sin_lon)
    vy = -cos_lat.mul(cos_lon)
    vz =  sin_lat
    vx = vx.expand(n, k, r, c)
    vy = vy.expand(n, k, r, c)
    vz = vz.expand(n, k, r, c)

    #
    voxel_dis = voxel_dis.contiguous().view(1, n*s*s*s)
    voxel_s = voxel_s.contiguous().view(1, n*s*s*s)

    # sample voxels along pano-rays
    d_samples = torch.arange(0, float(k), 1).view(1, k, 1, 1).expand(n, k, r, c)
    samples_x = vx.mul(d_samples).add(k).long()
    samples_y = vy.mul(d_samples).add(k).long()
    samples_z = vz.mul(d_samples).add(k).long()
    samples_n = torch.arange(0, n, 1).view(n, 1, 1, 1).expand(n, k, r, c).long()
    samples_indices = samples_n.mul(s*s*s).add(samples_z.mul(s*s)).add(samples_y.mul(s)).add(samples_x)
    samples_indices = samples_indices.view(1, n*k*r*c)
    samples_indices = samples_indices[0,:]

    # get depth pano
    samples_depth = torch.index_select(voxel_dis, 1, samples_indices)
    samples_depth = samples_depth.view(n, k, r, c)
    min_depth = torch.min(samples_depth, 1)
    pano_depth = min_depth[0]
    pano_depth = pano_depth.view(n, 1, r, c)

    # get sem pano
    idx_z = min_depth[1].cpu().long()
    idx_y = torch.arange(0, r, 1).view(1, r, 1).expand(n, r, c).long()
    idx_x = torch.arange(0, c, 1).view(1, 1, c).expand(n, r, c).long()
    idx_n = torch.arange(0, n, 1).view(n, 1, 1).expand(n, r, c).long()
    idx = idx_n.mul(k*r*c).add(idx_z.mul(r*c)).add(idx_y.mul(c)).add(idx_x).view(1, n*r*c)
    samples_s = torch.index_select(voxel_s, 1, samples_indices)
    pano_sem = torch.index_select(samples_s, 1, idx[0,:]).view(n,1,r,c).float()

    return pano_depth, pano_sem


def geo_projection(sate_depth, sate_sem, ori, sate_gsd=0.5, pano_size=[512, 1024]):
    gsize = int(sate_depth.size()[3] * sate_gsd)
    voxel_d, voxel_s = depth2voxel(sate_depth, sate_sem, gsize)
    ori = torch.Tensor([ori/180 * 3.1415926535])
    pano_depth, pano_sem = voxel2pano(voxel_d, voxel_s, pano_size, ori)
    pano_depth = pano_depth.mul(1.0/119.8)
    return pano_depth, pano_sem


def geo_transform_image(paths):
    height_img_path, semantic_img_path = paths
    if not osp.exists(height_img_path):
        print(f'File not existed: {height_img_path}')
        return
    if not osp.exists(semantic_img_path):
        print(f'File not existed: {semantic_img_path}')
        return

    global shared_geo_semantic_img_dir
    img_name = osp.basename(height_img_path)
    if osp.exists(osp.join(shared_geo_semantic_img_dir, osp.splitext(img_name)[0] + ".jpg")):
        return

    height_img = io.imread(height_img_path)
    height_img_tensor = transforms.ToTensor()(height_img[..., None])
    height_img_tensor = torch.stack([height_img_tensor])

    sem_img = cv2.imread(semantic_img_path)
    sem_img = cv2.cvtColor(sem_img, cv2.COLOR_BGR2GRAY)
    sem_img_tensor = transforms.ToTensor()(sem_img.astype(np.float32))
    sem_img_tensor = torch.stack([sem_img_tensor])

    ori = 180.0
    geo_height_img, geo_semantic_img = geo_projection(height_img_tensor, sem_img_tensor, ori, sate_gsd=0.268)
    geo_semantic_img = geo_semantic_img.permute(0, 2, 3, 1)[0].data.cpu().numpy()
    geo_semantic_img_path = osp.join(shared_geo_semantic_img_dir, osp.splitext(img_name)[0] + ".jpg")
    cv2.imwrite(geo_semantic_img_path, geo_semantic_img.astype(np.uint8))
    # ret, mask = cv2.threshold(geo_height_img, 1, 255, cv2.THRESH_BINARY)
    # geo_semantic_img_path = osp.join(shared_geo_semantic_img_dir, osp.splitext(img_name)[0] + ".jpg")
    # cv2.imwrite(geo_semantic_img_path, mask.astype(np.uint8))


def main(height_dir, semantic_dir, output_dir):
    global shared_geo_semantic_img_dir
    shared_geo_semantic_img_dir = output_dir
    process_func = geo_transform_image

    to_process_names = [osp.splitext(_)[0] for _ in os.listdir(height_dir)]
    to_process_paths = [
        (osp.join(height_dir, _+'.tif'), osp.join(semantic_dir, _+'.png'))
        for _ in to_process_names
        if osp.exists(osp.join(height_dir, _+'.tif')) and osp.exists(osp.join(semantic_dir, _+'.png'))]
    print(f"Totally {len(to_process_paths)} images to process.")
    with Pool(32) as p:
        p.map(process_func, to_process_paths)


if __name__ == '__main__':
    script_name = sys.argv[0].strip()
    if len(sys.argv) != 4:
        print(f"Usage: python {script_name} [height_dir] [semantic_dir] [output_dir]")
        sys.exit(0)

    height_dir = sys.argv[1].strip()
    semantic_dir = sys.argv[2].strip()
    output_dir = sys.argv[3].strip()
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    main(height_dir, semantic_dir, output_dir)


# python data/omnicity/geo_semantic_preprocess.py \
#     /mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/annotation-height-all/ \
#     /mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/annotation-landuse/test/ \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/geo-semantic-panorama/test/