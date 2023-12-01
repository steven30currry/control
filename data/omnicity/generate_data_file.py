import os
import sys
import json


def main(sate_img_dir, pano_img_dir, input_file, output_file):
    input_data = json.loads(open(input_file, 'r').read())
    valid_img_names = [record['sate_img_name'] for record in input_data
                       if record['sate_img_name'] in record['related_pano_img_names']]
    print(f"Totally {len(valid_img_names)} valid images")

    output_data = []
    for img_name in valid_img_names:
        item_id = img_name.split('_')[0]
        polar_sate_img_path = os.path.join(sate_img_dir, img_name)
        rotated_pano_img_path = os.path.join(pano_img_dir, img_name)
        assert os.path.exists(polar_sate_img_path)
        assert os.path.exists(rotated_pano_img_path)
        output_data.append(f"{item_id}\t{polar_sate_img_path}\t{rotated_pano_img_path}")

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


# python data/omnicity/generate_data_file.py \
#     data/omnicity/val2017-2019_sate-street.json \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/test/polar-image-satellite/ \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/rotated-image-panorama/test/ \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/test.csv