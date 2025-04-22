import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw
import os
import tqdm
import time

from src.core import YAMLConfig

def draw(images, labels, boxes, scores, out_filepath, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        for ii, b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(lab[ii].item()), fill='blue', )

        im.save(out_filepath)


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            # outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    example_inputs = torch.randn((args.batch, 3, *args.imgs)).to(args.device)
    
    orig_size = torch.tensor(args.imgs)[None].to(args.device)
    print('begin warmup...')
    # for i in tqdm(range(args.warmup), desc='warmup....'):
    for i in range(args.warmup):
        model(example_inputs, orig_size)
    
    print('begin test latency...')
    time_arr = []
    
    for i in range(args.testtime):
        start_time = time.time()
        
        model(example_inputs, orig_size)
        
        end_time = time.time()
        time_arr.append(end_time - start_time)
    
    std_time = np.std(time_arr)
    infer_time_per_image = np.sum(time_arr) / (args.testtime * args.batch)
    print(f'(bs:{args.batch})Latency:{infer_time_per_image:.5f}s +- {std_time:.5f}s fps:{1 / infer_time_per_image:.1f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml')
    parser.add_argument('-r', '--resume', type=str, default='')
    parser.add_argument('--batch', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='[height, width] image sizes')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--warmup', default=1000, type=int, help='warmup time')
    parser.add_argument('--testtime', default=2000, type=int, help='test time')

    args = parser.parse_args()
    main(args)

    
    
    
