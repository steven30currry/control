import sys
from PIL import Image

from transformers import AutoProcessor, BlipForConditionalGeneration


def main(in_file, out_file):
    model_name = "Salesforce/blip-image-captioning-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to('cuda')

    new_lines = []
    for line in open(in_file, 'r').readlines():
        img_path = line.strip().split('\t')[-1]
        inputs = processor(images=Image.open(img_path), return_tensors="pt").to('cuda')
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        new_lines.append(line.strip() + '\t' + caption)
        if len(new_lines) % 1000 == 0:
            print(f'Processed {len(new_lines)} images.')

    with open(out_file, 'w') as writer:
        writer.write('\n'.join(new_lines))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python [in_file] [out_file]')
        sys.exit(0)

    in_file = sys.argv[1].strip()
    out_file = sys.argv[2].strip()
    main(in_file, out_file)