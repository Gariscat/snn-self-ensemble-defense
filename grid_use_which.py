# python train.py dataset.data_type=frame epoch=50 dataset.frame_number=1-2 gpu_idx=2

import os
import subprocess
from itertools import chain, combinations
from argparse import ArgumentParser

# frame_numbers = (1, 2, 3, 4)
FrameNumbers = (1, 2, 3, 4, 6, 8, 12, 16)
def get_subsets(s):
    """Generate all subsets of a set s."""
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

# Example usage:
subsets = get_subsets(set(FrameNumbers))
# print(subsets)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu_idx', default=0, type=int)
    parser.add_argument('--attack', default='none', type=str)
    parser.add_argument('--max_iteration', default=100, type=int)
    parser.add_argument('--render', action="store_true")
    args = parser.parse_args()
    
    # training stage
    if args.attack == 'none':
        for frame_numbers in subsets:
            # print(frame_numbers)
            # continue
            if len(frame_numbers) != 4:
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
            if 1 in frame_numbers:  # ignore the case for latency == 1
                continue
            if len(frame_numbers) != 3:  # ignore the case for ensemble size != 3
                continue
            frame_arg_str = '-'.join(map(str, frame_numbers))
            for use_which, use_frame_id in enumerate(frame_numbers):
                ## frame_arg_str = '-'.join(map(str, frame_numbers))
                subprocess.call(f'python main.py \
                            dataset.data_type=event \
                            dataset.frame_number={frame_arg_str} \
                            use_which_frame_number={use_which} \
                            wandb_mode=disabled \
                            max_iteration={args.max_iteration} \
                            attack.init_alpha_mode={args.attack} \
                            optimizer=sgd \
                            gpu_idx={args.gpu_idx} \
                            num_pic=288 \
                            render={args.render}', \
                            shell=True
                        )
        
    # os.system("/usr/bin/shutdown")