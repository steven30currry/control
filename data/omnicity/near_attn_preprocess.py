import os
import sys
import cv2
import math
import numpy as np
import os.path as osp
from skimage import io
from scipy import sparse
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


def depth2voxel(img_depth, gsize):
    gsize = torch.tensor(gsize).int()
    n, c, h, w = img_depth.size()
    site_z = img_depth[:, 0, int(h/2), int(w/2)] + 3.0
    voxel_sitez = site_z.view(n, 1, 1, 1).expand(n, gsize, gsize, gsize)

    # depth voxel
    grid_mask = generate_grid(gsize, gsize)
    grid_mask = grid_mask.expand(n, gsize, gsize, 2)
    grid_depth = torch.nn.functional.grid_sample(img_depth, grid_mask,align_corners=True)
    voxel_depth = grid_depth.expand(n, gsize, gsize, gsize)
    voxel_depth = voxel_depth - voxel_sitez

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
    return voxel_dis


def voxel2pano(voxel_dis, size_pano, ori=torch.Tensor([0])):
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
    return pano_depth


def geo_projection(sate_depth, ori, sate_gsd=0.5, pano_size=[512, 1024]):
    gsize = sate_depth.size()[3] * sate_gsd
    voxel_d = depth2voxel(sate_depth, gsize)
    ori = torch.Tensor([ori/180.0 * 3.1415926535])
    pano_depth = voxel2pano(voxel_d, pano_size, ori)
    pano_depth = pano_depth.mul(1.0 / (0.9 * gsize))
    return pano_depth


def calc_distance(row1, col1, row2, col2):
    return math.sqrt(math.pow(row1-row2, 2.0) + math.pow(col1-col2, 2.0))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def score_func(distance):
    score = 1.0 - sigmoid(0.5*distance)
    return score if score >= 1e-2 else 0


def generate_weight_map(pano_size, tgt_row, tgt_col):
    (pano_height, pano_width) = pano_size
    result_map = np.zeros(pano_size)
    for row in range(pano_height):
        for col in range(pano_width):
            distance = calc_distance(row, col, tgt_row, tgt_col)
            result_map[row, col] = score_func(distance)
    return result_map


def process_func(item):
    height_img_path, angle, distance, output_path = item
    if osp.exists(output_path):
        return

    # geo transformation
    height_img = io.imread(height_img_path)
    height_img_tensor = transforms.ToTensor()(height_img[..., None])
    height_img_tensor = torch.stack([height_img_tensor])

    ori = 180.0
    geo_height_img = geo_projection(height_img_tensor, ori, sate_gsd=0.268).permute(0, 2, 3, 1)[0].data.cpu().numpy()

    # angle calculation
    pano_width = 1024
    pano_height = 512
    resized_pano_width = pano_width // 8
    resized_pano_height = pano_height // 8
    PI = math.pi

    width_pixel_per_angle = pano_width / 360.0
    theta_angle = round(angle)
    d = distance

    target_columns = []
    float_angle = 360.0 / resized_pano_width
    for alpha_angle in np.arange(0, 360, float_angle):
        k = int(pano_height / 2)

        d1 = geo_height_img[k][int(alpha_angle*width_pixel_per_angle) : int((alpha_angle+1)*width_pixel_per_angle)].squeeze(-1).mean()
        d1 = d1 * 0.9 * 512 * 0.268

        if theta_angle < 90 or theta_angle > 270:
            center_angle = (90 - alpha_angle - theta_angle) % 360
        else:
            center_angle = (alpha_angle + theta_angle - 90) % 360

        center = math.radians(center_angle)
        alpha = math.radians(alpha_angle)
        theta = math.radians(theta_angle)
        d2 = math.sqrt(d*d + d1*d1 - 2*d*d1*math.cos(center))
        cos_val = (d*d+d2*d2-d1*d1) / (2.0*d*d2)
        if cos_val < -1:
            cos_val = -1
        if cos_val > 1:
            cos_val = 1
        gamma = math.acos(cos_val)

        if theta_angle < 90 or theta_angle > 270:
            if alpha > (PI/2.0 - theta) % (2.0*PI) and alpha <= (3.0*PI/2.0 - theta) % (2.0*PI):
                beta = (3.0*PI/2.0 - theta - gamma) % (2.0*PI)
            else:
                beta = (3.0*PI/2.0 - theta + gamma) % (2.0*PI)
        else:
            if alpha > (3.0*PI/2.0 - theta) % (2.0*PI) and alpha <= (5.0*PI/2.0 - theta) % (2.0*PI):
                beta = (3.0*PI/2.0 - theta + gamma) % (2.0*PI)
            else:
                beta = (3.0*PI/2.0 - theta - gamma) % (2.0*PI)
        beta_angle = round(beta / PI * 180.0)
        target_columns.append(round(beta_angle*width_pixel_per_angle/8))

    # attention map generation
    pano_pixel_num = resized_pano_height * resized_pano_width
    full_weight_map = np.zeros((pano_pixel_num, pano_pixel_num))
    for src_col in range(resized_pano_width):
        tgt_col = target_columns[src_col]
        if tgt_col < 0:
            tgt_col = 0
        elif tgt_col >= resized_pano_width:
            tgt_col = resized_pano_width - 1

        for src_row in range(resized_pano_height):
            tgt_row = src_row
            weight_index = src_row * resized_pano_width + src_col
            result_map = generate_weight_map((resized_pano_height, resized_pano_width), tgt_row, tgt_col)
            full_weight_map[weight_index, :] = result_map.reshape(-1)
    full_weight_map=sparse.csr_matrix(full_weight_map)
    sparse.save_npz(output_path,full_weight_map)


def main(input_file, output_dir, output_file):
    # train
    # height_img_dir = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/annotation-height/annotation-height-train'
    # test
    height_img_dir = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/annotation-height/annotation-height-test'

    to_process_items = []
    writer = open(output_file, 'w')
    for line in open(input_file, 'r').readlines():
        segs = line.strip().split('\t')
        center_img_name = osp.basename(segs[1].strip())[:-4]
        center_height_img_path = osp.join(height_img_dir, center_img_name + '.tif')
        assert osp.exists(center_height_img_path)
        near_img_name = osp.basename(segs[3].strip())[:-4]
        near_height_img_path = osp.join(height_img_dir, near_img_name + '.tif')
        if not osp.exists(near_height_img_path):
            print(f"{near_img_name} doesn't have height image in {height_img_dir}")
        angle = float(segs[4].strip())
        distance = float(segs[5].strip())
        to_process_items.append((center_height_img_path, angle, distance, osp.join(output_dir, f"{center_img_name}.npz")))
        segs[-1] = osp.join(output_dir, f"{center_img_name}.npz")
        writer.write('\t'.join(segs) + '\n')
    writer.flush()
    writer.close()

    print(f"Totally {len(to_process_items)} images to process.")
    with Pool(32) as p:
        p.map(process_func, to_process_items)


if __name__ == '__main__':
    script_name = sys.argv[0].strip()
    if len(sys.argv) != 4:
        print(f"Usage: python {script_name} [input_file] [output_dir] [output_file]")
        sys.exit(0)

    input_file = sys.argv[1].strip()
    output_dir = sys.argv[2].strip()
    output_file = sys.argv[3].strip()
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    main(input_file, output_dir, output_file)


# python data/omnicity/near_attn_preprocess.py \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_full_multihint_test.csv \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/nearest-panorama-weight/test \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_attn_multihint_test.csv

# python data/omnicity/near_attn_preprocess.py \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_full_multihint_test.csv \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/nearest-panorama-weight/test \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_attn_multihint_test.csv