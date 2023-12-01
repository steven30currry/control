import os
import sys
import json


def main(cond_img_dir, pano_img_dir, input_file, output_file):
    input_data = [_.strip() for _ in open(input_file, 'r').readlines()]
    print(f"Totally {len(input_data)} images in {input_file}")

    output_data = []
    for img_name in input_data:
        item_id = img_name.split('_')[0]
        cond_img_path = os.path.join(cond_img_dir, img_name)
        pano_img_path = os.path.join(pano_img_dir, img_name)

        if not os.path.exists(cond_img_path):
            print(f'{img_name} not existed in {cond_img_dir}')
            continue
        if not os.path.exists(pano_img_path):
            print(f'{img_name} not existed in {pano_img_dir}')
            continue
        output_data.append(f"{item_id}\t{cond_img_path}\t{pano_img_path}")

    with open(output_file, 'w') as writer:
        writer.write('\n'.join(output_data))


if __name__ == '__main__':
    script_name = sys.argv[0].strip()
    if len(sys.argv) != 4:
        print(f"Usage: python {script_name} [input_file] [mode] [output_file]")
        sys.exit(0)

    input_file = sys.argv[1].strip()
    mode = sys.argv[2].strip()
    output_file = sys.argv[3].strip()

    pano_img_dir = '/mnt/petrelfs/share_data/chenyuankun/omnicity/street-level/rotated-image-panorama/'
    if mode == 'polar_sate':
        cond_img_dir = '/mnt/petrelfs/share_data/chenyuankun/omnicity/satellite-level/polar-image-satellite/'
    elif mode == 'geo_height':
        cond_img_dir = '/mnt/petrelfs/share_data/chenyuankun/omnicity/street-level/geo-height-panorama/'
    elif mode == 'mirror':
        cond_img_dir = '/mnt/petrelfs/share_data/chenyuankun/omnicity/street-level/rotated-image-panorama/'

    main(cond_img_dir, pano_img_dir, input_file, output_file)

# mirror
# python data/omnicity/generate_data_file_v2.py \
#     /mnt/petrelfs/share_data/chenyuankun/omnicity/random_train_cleaned2.txt \
#     mirror \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/mirror/random-cleaned2-train.csv
# python data/omnicity/generate_data_file_v2.py \
#     /mnt/petrelfs/share_data/chenyuankun/omnicity/random_test_cleaned2.txt \
#     mirror \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/mirror/random-cleaned2-test.csv

# geo_height
# python data/omnicity/generate_data_file_v2.py \
#     /mnt/petrelfs/share_data/chenyuankun/omnicity/random_train_cleaned2.txt \
#     geo_height \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height/random-cleaned2-train.csv
# python data/omnicity/generate_data_file_v2.py \
#     /mnt/petrelfs/share_data/chenyuankun/omnicity/random_test_cleaned2.txt \
#     geo_height \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height/random-cleaned2-test.csv

# polar_sate
# python data/omnicity/generate_data_file_v2.py \
#     /mnt/petrelfs/share_data/chenyuankun/omnicity/random_train_cleaned2.txt \
#     polar_sate \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-train.csv
# python data/omnicity/generate_data_file_v2.py \
#     /mnt/petrelfs/share_data/chenyuankun/omnicity/random_test_cleaned2.txt \
#     polar_sate \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv