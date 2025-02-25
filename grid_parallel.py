import subprocess
import torch
import os
import subprocess
from itertools import chain, combinations
from argparse import ArgumentParser
from typing import List

# frame_numbers = (1, 2, 3, 4)
FrameNumbers = (1, 2, 3, 4, 6, 8, 12, 16)
def get_subsets(s):
    """Generate all subsets of a set s."""
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

# Example usage:
subsets = get_subsets(set(FrameNumbers))
# print(subsets)
    
import multiprocessing as mp
import subprocess

def run_job(python_cmd, gpu_id):
    """
    Runs a single experiment on the specified GPU.
    """
    command = f"{python_cmd} gpu_idx={gpu_id}"
    # print(f"GPU {gpu_id} starting run \"{python_cmd}\"")
    subprocess.run(command, shell=True)

def main(cmds: List[str]):
    ## total_runs = 20
    n_gpus = torch.cuda.device_count()
    # Distribute runs evenly among GPUs
    ## runs_per_gpu = total_runs // n_gpus

    # Create a pool for each GPU. Here, pool size = 1 means one job at a time per GPU.
    pools = []
    for gpu in range(n_gpus):
        pool = mp.Pool(processes=2)  # Change this number if you want more concurrency on a single GPU.
        pools.append(pool)
    
    # Assign runs to the corresponding GPU pool
    # for gpu, pool in pools:
    #     for i in range(runs_per_gpu):
    #         run_id = gpu * runs_per_gpu + i
    #         pool.apply_async(run_job, args=(run_id, gpu))
    for cmd_idx, cmd in enumerate(cmds):
        gpu_id = cmd_idx % n_gpus
        pools[gpu_id].apply_async(run_job, args=(cmd, gpu_id))
    
    # Close and join all pools
    for pool in pools:
        pool.close()
        
    for pool in pools:
        pool.join()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu_idx', default=0, type=int)
    parser.add_argument('--attack', default='none', type=str)
    parser.add_argument('--max_iteration', default=20, type=int)
    parser.add_argument('--render', action="store_true")
    parser.add_argument("--add_position_label_mode", default="target", type=str)
    parser.add_argument("--ensemble_size", default=3, type=int)
    args = parser.parse_args()
    
    cmds = []
    # training stage
    if args.attack == 'none':
        for frame_numbers in subsets:
            # print(frame_numbers)
            # continue
            if len(frame_numbers) != 2:
                continue
            num_epochs = 100 // len(frame_numbers)
            frame_arg_str = '-'.join(map(str, frame_numbers))
            print("Running with frame numbers:", frame_arg_str, "for", num_epochs, "epochs")
            cmds += [f'python train.py \
                        dataset.data_type=frame \
                        epoch={num_epochs} \
                        dataset.frame_number={frame_arg_str}']
        print("All runs:")
        print(cmds)
    
    # attack stage
    else:
        for frame_numbers in subsets:
            if 1 in frame_numbers:  # ignore the case for latency == 1
                continue
            if len(frame_numbers) != args.ensemble_size:  # ignore the case for ensemble size != 3
                continue
            frame_arg_str = '-'.join(map(str, frame_numbers))
            for use_which, use_frame_id in enumerate(frame_numbers):
                ## frame_arg_str = '-'.join(map(str, frame_numbers))
                cmds += [f'python main.py \
                            dataset.data_type=event \
                            dataset.frame_number={frame_arg_str} \
                            use_which_frame_number={use_which} \
                            wandb_mode=disabled \
                            max_iteration={args.max_iteration} \
                            attack.init_alpha_mode={args.attack} \
                            attack.add_position_label_mode={args.add_position_label_mode} \
                            optimizer=sgd \
                            optimizer.lr=5e-2 \
                            num_pic=200 \
                            render={args.render}']
        # print("All runs:")
        # print(cmds)
        
    main(cmds)