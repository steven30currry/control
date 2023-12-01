import os
import sys
import json
import numpy as np


def main(sate_img_dir, pano_img_dir, input_file, output_file):
    input_data = json.loads(open(input_file, 'r').read())

    valid_img_pairs = []
    for record in input_data:
        if record['sate_img_name'] not in record['related_pano_img_names']:
            continue
        if len(record['related_pano_img_distances']) <= 1:
            continue

        distances = np.array(record['related_pano_img_distances'])
        index = np.argsort(distances)[1]
        nearest_img_name = record['related_pano_img_names'][index]
        nearest_angle = record['relative_angle_horizon'][index]
        nearest_distance = distances[index]
        valid_img_pairs.append((record['sate_img_name'], nearest_img_name, nearest_angle, nearest_distance))
    print(f"Totally {len(valid_img_pairs)} valid images")

    output_data = []
    for (img_name, nearest_img_name, nearest_angle, nearest_distance) in valid_img_pairs:
        item_id = img_name.split('_')[0]
        polar_sate_img_path = os.path.join(sate_img_dir, img_name)
        rotated_pano_img_path = os.path.join(pano_img_dir, img_name)
        nearest_rotated_pano_img_path = os.path.join(pano_img_dir, nearest_img_name)
        assert os.path.exists(polar_sate_img_path)
        assert os.path.exists(rotated_pano_img_path)
        assert os.path.exists(nearest_rotated_pano_img_path)
        output_data.append(f"{item_id}\t{polar_sate_img_path}\t{rotated_pano_img_path}\t{nearest_rotated_pano_img_path}\t{nearest_angle}\t{nearest_distance}")

    with open(output_file, 'w') as writer:
        writer.write('\n'.join(output_data))


if __name__ == '__main__':
    script_name = sys.argv[0].strip()
    if len(sys.argv) != 5:
        print(f"Usage: python {script_name} [input_file] [sate_img_dir] [pano_img_dir] [output_file]")
        sys.exit(0)

    input_file = sys.argv[1].strip()
    sate_img_dir = sys.argv[2].strip()
    pano_img_dir = sys.argv[3].strip()
    output_file = sys.argv[4].strip()
    main(sate_img_dir, pano_img_dir, input_file, output_file)


# python data/omnicity/generate_multi_data_file.py
#   data/omnicity/train-2019-sate-street-angle.json
#   /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/geo-height-panorama/train/
#   /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/rotated-image-panorama/train/
#   /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_full_multihint_train.csv