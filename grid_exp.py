# python train.py dataset.data_type=frame epoch=50 dataset.frame_number=1-2 gpu_idx=2

import os
import subprocess
from itertools import chain, combinations
from argparse import ArgumentParser

frame_numbers = (1, 2, 3, 4)
def get_subsets(s):
    """Generate all subsets of a set s."""
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

# Example usage:
subsets = get_subsets(set(frame_numbers))
# print(subsets)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu_idx', default=1, type=int)
    parser.add_argument('--attack', default='none', type=str)
    parser.add_argument('--max_iteration', default=50, type=int)
    args = parser.parse_args()
    
    # training stage
    if args.attack == 'none':
        for frame_numbers in subsets:
            # print(frame_numbers)
            # continue
            if len(frame_numbers) == 0:
                continue
            num_epochs = 100 // len(frame_numbers)
            frame_arg_str = '-'.join(map(str, frame_numbers))
            print("Running with frame numbers:", frame_arg_str, "for", num_epochs, "epochs")
            subprocess.call(f'python train.py \
                        dataset.data_type=frame \
                        epoch={num_epochs} \
                        dataset.frame_number={frame_arg_str} \
                        gpu_idx={args.gpu_idx}', \
                        shell=True
                    )
    
    # attack stage
    else:
        for frame_numbers in subsets:
            if len(frame_numbers) < 2:
                continue
            frame_arg_str = '-'.join(map(str, frame_numbers))
            subprocess.call(f'python main.py \
                        dataset.data_type=event \
                        dataset.frame_number={frame_arg_str} \
                        use_wandb=True \
                        max_iteration={args.max_iteration} \
                        attack.init_alpha_mode={args.attack} \
                        optimizer=sgd \
                        gpu_idx={args.gpu_idx} \
                        num_pic=100', \
                        shell=True
                    )
        
    # os.system("/usr/bin/shutdown")