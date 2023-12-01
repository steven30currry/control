import os
import sys
import json
import random


def main(input_file, output_dir):
    # {item_id}\t{polar_sate_img_path}\t{rotated_pano_img_path}
    input_data = open(input_file, 'r').read().splitlines()
    print(f"Totally {len(input_data)} input records")

    input_data = sorted(input_data)
    random.shuffle(input_data)
    train_data = input_data[:12000]
    valid_data = input_data[12000:]
    print(f"Totally {len(train_data)} training records.")
    print(f"Totally {len(valid_data)} validation records.")

    with open(os.path.join(output_dir, 'train.txt'), 'w') as writer:
        writer.write('\n'.join(train_data))
    with open(os.path.join(output_dir, 'valid.txt'), 'w') as writer:
        writer.write('\n'.join(valid_data))


if __name__ == '__main__':
    script_name = sys.argv[0].strip()
    if len(sys.argv) != 3:
        print(f"Usage: python {script_name} [input_file] [output_dir]")
        sys.exit(0)

    random.seed(42)
    input_file = sys.argv[1].strip()
    output_dir = sys.argv[2].strip()
    main(input_file, output_dir)