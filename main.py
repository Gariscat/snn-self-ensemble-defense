import os
from pprint import pprint
from copy import deepcopy
import io
from typing import Dict, Literal, List
from utils.instatiate_utils import get_model
from spikingjelly.activation_based import functional
from render import *

os.environ["WANDB_MODE"] = "disabled"
import time

import hydra
import torch
from ignite.utils import manual_seed
from omegaconf import DictConfig, OmegaConf

import wandb
from attacker import GumbelAttacker
from attacks.ours.probability_space.frame_generator import FrameGenerator
from attacks.ours.probability_space.probability_attack import ProbabilityAttacker
from utils.general_utils import *
from utils.instatiate_utils import get_dataloaders, get_event_generator
from utils.metrics import Metrics, TotalMetrics
from utils.optimizer import CosineAnnealing, get_lr_scheduler, get_optimizer
import json

def test_on_ensemble(
    cfg: Dict,
    model: nn.Module,
    event_pair: Dict[str, np.lib.npyio.NpzFile], # one for original, one for adversarial
    ## ens_mode: Literal["vote", "avg_logits", "avg_prob"]
):
    device = get_device(cfg["gpu_idx"])
    fr_processors = []
    for fr_n in cfg["dataset"]["frame_number"]:
        fr_processors.append(
            FrameGenerator(
                split_by=cfg["dataset"]["split_by"],
                frame_number=fr_n,
                frame_size=cfg["dataset"]["frame_size"],
            )
        )

    ## indices, values = pre_process(outcome["original_event"], device)
    ## print(indices.shape, values.shape)
    
    logits_list_for_ori = [
        event_to_logits(model, event_pair["original_event"], fr_processor, device) for fr_processor in fr_processors
    ]
    ori_pred = logits_list_to_pred(logits_list_for_ori, mode=cfg["ens_mode"])
    
    logits_list_for_adv = [
        event_to_logits(model, event_pair["adversarial_event"], fr_processor, device) for fr_processor in fr_processors
    ]
    adv_pred = logits_list_to_pred(logits_list_for_adv, mode=cfg["ens_mode"])
    
    return ori_pred, adv_pred

@hydra.main(config_path="./config", config_name="default", version_base=None)
def main(dictconfig: DictConfig):
    cfg: dict = OmegaConf.to_container(dictconfig, resolve=True)  # type:ignore
    manual_seed(cfg["seed"])
    cfg["use_grad_scaling"] = cfg["use_grad_scaling"] if cfg["use_amp"] else False
    assert cfg["dataset"]["data_type"] == "event"
    cfg["project"] = "GumbelSoftmax-attack"
    # print(cfg['optimizer']['lr'])
    # exit()
    
    torch.manual_seed(cfg["seed"])
    
    attack_on_ensemble = isinstance(cfg["dataset"]["frame_number"], str)
    if attack_on_ensemble: # multiple latency for one SNN
        ensemble_res = [[0, 0], [0, 0]]
        cfg["dataset"]["frame_number"] = list(map(int, cfg["dataset"]["frame_number"].split('-')))
    else:
        cfg["dataset"]["frame_number"] = [cfg["dataset"]["frame_number"], ]
        
    os.environ["WANDB_MODE"] = cfg["wandb_mode"]
    # init one run for all image
    wandb.init(
        config=cfg,
        project=cfg["project"],
        ## entity=cfg["entity"],
        # name=cfg["name"],
        reinit=True,
    )
    wandb.define_metric("*", step_metric="global_step")

    # prepare device, dataset and model
    device = get_device(cfg["gpu_idx"])
    # use a temporary cfg dict for dataset since the we might have multiple frame number
    # the attack process could only target at one frame number in each run
    tmp_cfg = deepcopy(cfg)
    tmp_cfg["model"]["frame_number"] = \
        tmp_cfg["dataset"]["frame_number"] = cfg["dataset"]["frame_number"][cfg["use_which_frame_number"]]
    _, _, test_data = get_dataloaders(
        **tmp_cfg["dataset"],
        transform=cfg["transform"],
    )
    model = get_model(cfg, device, attack=True)

    # init some generator
    event_generator = get_event_generator(cfg)
    frame_processor = FrameGenerator(
        split_by=tmp_cfg["dataset"]["split_by"],
        frame_number=tmp_cfg["dataset"]["frame_number"],
        frame_size=tmp_cfg["dataset"]["frame_size"],
    )

    (
        correct_num,
        correct_events,
        true_labels,
        true_values,
        true_indices,
        true_frame_list,
        model_acc
    ) = remove_misclassification(cfg, test_data, model, frame_processor, device)
    print(f"Correct number: {correct_num}")
    target_labels = get_target_label(cfg, true_labels, device, cfg["seed"])
    auxiliary_event_dict: dict = generate_auxiliary_samples_for_each_sample(
        cfg, true_labels, correct_events
    )
    
    # Prepare a sample for each sample to be attacked.
    metrics = Metrics(total_num_correct=correct_num)
    start = time.time()
    
    # # Specify the file path
    # file_path = f"alpha_gpu_{cfg['gpu_idx']}.txt"

    # try:
    #     # Remove the file
    #     os.remove(file_path)
    #     print(f"File {file_path} has been removed.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    
    # START ATTACK
    for i, (event, true_label, target_label, true_frame, true_indice) in enumerate(
        zip(correct_events, true_labels, target_labels, true_frame_list, true_indices)
    ):
        assert event['t'].shape[0] == event['x'].shape[0] == event['y'].shape[0] == event['p'].shape[0]

        target_label_for_addition_position = get_target_label_for_add_position(
            add_position_label_mode=cfg["attack"]["add_position_label_mode"],
            target_label=target_label,
            num_class=cfg["dataset"]["num_class"],
            seed=cfg["seed"]
        )
        alpha_dict = {
            "events": event,
            "device": device,
            "init_alpha_mode": cfg["attack"]["init_alpha_mode"],
            "event_dict": auxiliary_event_dict,
            "target_label": target_label_for_addition_position,
            "target_position_ratio": cfg["attack"]["target_position_ratio"],
            "true_label": true_label,
            "num_class": cfg["dataset"]["num_class"],
        }
        (
            true_value_for_attack,
            hard_true_value,
            probability_attacker,
            optimizer_alpha,
            lr_scheduler,
            scaler,
            temperture_tau_scheduler,
        ) = prepare_attack(
            cfg=cfg,
            alpha_dict=alpha_dict,
            event_generator=event_generator,
            frame_processor=frame_processor,
        )
        # print(event['p'].shape, true_value_for_attack.shape, hard_true_value.shape)
        # continue
        wandb.watch(probability_attacker, log="all", log_freq=10)

        # initial an attacker
        attacker = GumbelAttacker(
            cfg=cfg,
            probability_attacker=probability_attacker,
            target_label=target_label,
            orginal_value=true_value_for_attack,
            optimizer_alpha=optimizer_alpha,
            model=model,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
        )

        # start attack
        attack_metrics = TotalMetrics()
        one_step_metrics = None
        epoch = 0
        
        event = modify_timestamp(event, event['t']-event['t'].min())
        
        for epoch in range(1, cfg["max_iteration"] + 1):
            # attack one step
            one_step_metrics, results = attacker.attack_one_step(epoch=epoch)

            results["hard_event_indices"] = true_indice.repeat_interleave(
                cfg["attack"]["sample_num"], dim=0
            )

            # judge if success
            preds = torch.argmax(results["logits"], dim=-1)
            if is_success(preds, target_label, cfg["targeted"]):
                metrics.update_metrics(
                    cfg=cfg,
                    preds=preds.cpu(),
                    target_label=target_label.cpu(),
                    results=results,
                    true_value=hard_true_value,
                    iters=epoch,
                    i=i,
                    one_step_metrics=one_step_metrics,
                    true_frame=true_frame,
                    true_label=true_label,
                )
                metrics.event_number_orginal_list.append(event["p"].shape[0])

                print(
                    f"[succ] Image:{i}, target:{target_label.item()}, iteration:{epoch}"
                )
                # print("original:")
                # print(event['p'])
                # print("adversarial:")
                # print(results["hard_event_values"].cpu().numpy())
                if cfg["attack"]["init_alpha_mode"] == "default":  # no new event added
                    adv_event_p = results["hard_event_values"].cpu().flatten().numpy().astype(int)
                    assert adv_event_p.shape[0] == event['p'].shape[0], f"{adv_event_p.shape[0]} != {event['p'].shape[0]}"
                    assert_arrays_differ(adv_event_p, event['p'])
                    adv_event = modify_polarity(event, adv_event_p)
                else:
                    adv_event_p = results["hard_event_values"].cpu().flatten().numpy().astype(int)
                    _indices = attacker.probability_attacker.get_indices()
                    adv_event_t = _indices[:, 0].cpu().flatten().numpy()
                    adv_event_x = _indices[:, 1].cpu().flatten().numpy()
                    adv_event_y = _indices[:, 2].cpu().flatten().numpy()
                    
                    adv_event_t -= adv_event_t.min()
                    
                    adv_event = {
                        "t": adv_event_t,
                        "x": adv_event_x,
                        "y": adv_event_y,
                        "p": adv_event_p,
                    }
                    
                sample_index = i
                
                if cfg["render"]:
                    if cfg["dataset"]["name"] == "gesture-dvs":
                        frame_size = (128, 128)
                    elif cfg["dataset"]["name"] == "nmnist":
                        frame_size = (34, 34)
                    elif cfg["dataset"]["name"] == "cifar10-dvs":
                        frame_size = (128, 128)
                    filename = f"renders/ia-{cfg['attack']['init_alpha_mode']}-ap-{cfg['attack']['add_position_label_mode']}_sample_{sample_index}_cls_{true_label}_use_{cfg['use_which_frame_number']}_on_{cfg['dataset']['frame_number']}.mp4"
                    
                    print("start timestamps:", event['t'].min(), adv_event['t'].min())
                    print("end timestamps", event['t'].max(), adv_event['t'].max())
                    print("count:", event['t'].shape[0], adv_event['t'].shape[0])
                    if cfg["attack"]["init_alpha_mode"] == "default":
                        # events_to_combined_video(event, adv_event, frame_size=frame_size, filename=filename)
                        events_to_combined_video_normalized(event, adv_event, frame_size=frame_size, filename=filename)
                    else:
                        # ori_filename = filename.replace('.mp4', '_ori.mp4')
                        # adv_filename = filename.replace('.mp4', '_adv.mp4')
                        # events_to_separate_videos(event, adv_event, time_window=2048, frame_size=frame_size, filenames=(ori_filename, adv_filename))
                        
                        # events_to_combined_video_unsynced(event, adv_event, frame_size=frame_size, filename=filename)
                        events_to_combined_video_normalized(event, adv_event, frame_size=frame_size, filename=filename)
                
                event_pair = {
                    "original_event": event,
                    "adversarial_event": adv_event,
                    "sample_index": sample_index,
                    "true_label": true_label,
                }
                
                if attack_on_ensemble:
                    ori_pred, adv_pred = test_on_ensemble(cfg=cfg, model=model, event_pair=event_pair)
                    _i = ori_pred==true_label
                    _j = adv_pred==true_label
                    ensemble_res[_i][_j] += 1
                    
                break

            # update attack_metrics
            attack_metrics.update_metrics(
                gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
                **one_step_metrics,
            )

            if epoch % cfg["log_loss_interval"] == 0:
                attack_metrics.upload(
                    step=epoch,
                    name="",
                    image_id=i,
                )
                attack_metrics.reset()

            temperture_tau_scheduler.step(iters=epoch)
            model.zero_grad()
            torch.cuda.empty_cache()
        else:
            print(f"[fail] Image:{i}, target:{target_label.item()}, iter:{epoch}")

        torch.cuda.synchronize()
        del probability_attacker, optimizer_alpha

    end = time.time()
    metrics.finalize_metrics(start=start, end=end, cfg=cfg)
    
    if attack_on_ensemble:
        print("Ensemble result:")
        success_cnt = ensemble_res[1][1] + ensemble_res[1][0] + ensemble_res[0][1] + ensemble_res[0][0]
        print("Both correct:", ensemble_res[1][1])
        print("Both wrong:", ensemble_res[0][0])
        print("Original correct, adv wrong:", ensemble_res[1][0])
        print("Original wrong, adv correct:", ensemble_res[0][1])
        
        if success_cnt == 0:
            wandb.log({"no_successful_attack": True})
        else:
            wandb.log({"both_correct": ensemble_res[1][1] / success_cnt})
            wandb.log({"both_wrong": ensemble_res[0][0] / success_cnt})
            wandb.log({"ori_correct_adv_wrong": ensemble_res[1][0] / success_cnt})
            wandb.log({"ori_wrong_adv_correct": ensemble_res[0][1] / success_cnt})
            
            with open(f"logs/fn-ia-{cfg['attack']['init_alpha_mode']}-ap-{cfg['attack']['add_position_label_mode']}-{cfg['dataset']['frame_number']}-use-{[cfg['use_which_frame_number']]}.json", 'w') as f:
                json.dump({
                    "both_correct:": ensemble_res[1][1] / success_cnt,
                    "both_wrong:": ensemble_res[0][0] / success_cnt,
                    "ori_correct_adv_wrong": ensemble_res[1][0] / success_cnt,
                    "ori_wrong_adv_correct": ensemble_res[0][1] / success_cnt,
                    "model_acc": model_acc,
                }, f)
            


def prepare_attack(
    cfg: dict,
    alpha_dict: dict,
    event_generator,
    frame_processor,
):
    """
    Prepares the attack by initializing the necessary components and parameters.

    Args:
        cfg (dict): Configuration dictionary containing attack settings.
        alpha_dict (dict): Dictionary containing alpha values.
        event_generator: Event generator object.
        frame_processor: Frame processor object.

    Returns:
        Tuple: A tuple containing the following elements:
            - true_value_for_attack: True value for the attack.
            - hard_true_value: Hard true value for the attack.
            - event2value_attacker: event2value_attacker object.
            - optimizer_alpha: Optimizer for alpha values.
            - lr_scheduler: Learning rate scheduler.
            - scaler: Gradient scaler.
            - temperture_tau_scheduler: Temperature tau scheduler.
    """
    event2value_attacker = ProbabilityAttacker(
        attack_cfg=cfg["attack"],
        alpha_dict=alpha_dict,
        event_generator=event_generator,
        frame_processor=frame_processor,
        seed=cfg["seed"],
        gpu_idx=cfg["gpu_idx"],
    )

    hard_true_value, true_value_for_attack = get_true_values(
        event2value_attacker.alpha.clone(), use_soft_event=cfg["use_soft_event"]
    )

    # preprocess
    optimizer_alpha = get_optimizer(
        event2value_attacker.parameters(), **cfg["optimizer"]
    )
    lr_scheduler = get_lr_scheduler(optimizer_alpha, **cfg["scheduler"])
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # type:ignore
    temperture_tau_scheduler = CosineAnnealing(
        optimizer=event_generator,
        initial_value=cfg["attack"]["min_tau"],
        final_value=cfg["attack"]["max_tau"],
        decay_step=cfg["attack"]["decay_step"],
    )

    return (
        true_value_for_attack,
        hard_true_value,
        event2value_attacker,
        optimizer_alpha,
        lr_scheduler,
        scaler,
        temperture_tau_scheduler,
    )
    
def modify_timestamp(event: np.lib.npyio.NpzFile, new_t: np.ndarray):
    _data_dict = {key: event[key].copy() for key in event}
    _data_dict['t'] = new_t
    _memory_buffer = io.BytesIO()
    # Save the data into the in-memory buffer (not to disk)
    np.savez(_memory_buffer, **_data_dict)
    # Load from the in-memory buffer to create a new NpzFile object
    _memory_buffer.seek(0)  # Reset buffer position
    adv_event = np.load(_memory_buffer)
    return adv_event

def modify_polarity(event: np.lib.npyio.NpzFile, new_p: np.ndarray):
    _data_dict = {key: event[key].copy() for key in event}
    # _data_dict['p'] = results['adv_sample']
    _data_dict['p'] = new_p
    _memory_buffer = io.BytesIO()
    # Save the data into the in-memory buffer (not to disk)
    np.savez(_memory_buffer, **_data_dict)
    # Load from the in-memory buffer to create a new NpzFile object
    _memory_buffer.seek(0)  # Reset buffer position
    adv_event = np.load(_memory_buffer)
    return adv_event

def assert_arrays_differ(arr1, arr2):
    # Ensure arrays have the same length
    assert arr1.shape == arr2.shape, "Arrays must have the same shape."
    # Check if there is at least one index where values differ
    if np.any(arr1 != arr2):
        return True
    else:
        raise AssertionError("No differing values found between arrays.")
    
def event_to_logits(model: nn.Module, event: np.lib.npyio.NpzFile, frame_processor: FrameGenerator, device):
    functional.reset_net(model)
    indices, values = pre_process(event, device)
    frames = frame_processor.forward(
        event_indices=indices, event_values=values, use_soft=False
    )
    logits = model(frames.to(device)).mean(0).flatten()
    ## print(logits.shape)
    return logits

def logits_list_to_pred(logits_list: List[torch.Tensor], mode=Literal["vote", "avg_logits", "avg_prob"]) -> torch.Tensor:
    if mode == "vote":
        return torch.mode(torch.stack(logits_list).argmax(-1)).values.item()
    elif mode == "avg_logits":
        return sum(logits_list).argmax(-1).item()
    elif mode == "avg_prob":
        return sum([torch.softmax(x, -1) for x in logits_list]).argmax(-1).item()
    else:
        raise NotImplementedError("Invalid ensemble mode")


if __name__ == "__main__":
    main()