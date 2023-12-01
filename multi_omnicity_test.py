import os
import torch
import argparse
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from share import *
from multi_omnicity_dataset import MultiOmniCityDataset
from cldm.logger import ValidImageLogger
from cldm.model import create_model, load_state_dict


def create_argparser():
    defaults = dict(
        config_path = './models/mcldm_v15.yaml',
        model_path = './models/mcontrol_sd15_ini.ckpt',
        image_width = 512,
        image_height = 256,
        source_image_width = 512,
        source_image_height = 256,
        text_prompt = 'a realistic, detailed, high-quality street view panorama image',
        data_file_path = '/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/small-view/test.csv',
        batch_size = 4,
        result_dir = './results',
        logger_freq = 1,
        sample_num = 10000,
        unconditional_guidance_scale = 9.0,
        num_gpus = 8,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    model = create_model(args.config_path).cpu()
    model.load_state_dict(load_state_dict(args.model_path, location='cpu'))
    model = model.cuda()
    model.eval()

    image_size = (args.image_width, args.image_height)
    source_image_size = (args.source_image_width, args.source_image_height)
    dataset = MultiOmniCityDataset(args.data_file_path, args.text_prompt, image_size, source_image_size)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size, shuffle=False)

    logger = ValidImageLogger(args.result_dir, args.logger_freq)
    trainer = pl.Trainer(
        gpus=args.num_gpus,
        strategy="ddp",
        precision=32,
        max_epochs=1,
        callbacks=[logger]
    )
    trainer.validate(model, dataloader)

    # dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)
    # sample_num = min(args.sample_num, len(dataset)) // args.batch_size
    # with torch.no_grad():
    #     for idx, batch in enumerate(dataloader):
    #         if idx > sample_num:
    #             break

    #         names = batch['name']
    #         images = model.log_images(batch, unconditional_guidance_scale=args.unconditional_guidance_scale)
    #         for k in images:
    #             # if k == 'control':
    #             #     continue
    #             images[k] = images[k].detach().cpu()
    #             images[k] = torch.clamp(images[k], -1., 1.)
    #             imgs = (images[k] + 1.0) / 2.0
    #             imgs = torch.split(imgs, 1, dim=0)
    #             for name, img in zip(names, imgs):
    #                 img = img.squeeze(0).transpose(0, 1).transpose(1, 2).numpy()
    #                 img = (img * 255).astype(np.uint8)
    #                 filename = "{}_{}.png".format(name, k)
    #                 Image.fromarray(img).save(os.path.join(args.result_dir, filename))


if __name__ == '__main__':
    main()
