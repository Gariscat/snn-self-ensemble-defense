This is responsory is modified from the repo of the paper "Exploring Vulnerabilities in Spiking Neural Networks: Direct Adversarial Attacks on Raw Event Data", from ECCV 2024.

## Requirements

```
torch ==  2.0.0+cu118
torchvision == 0.15.1+cu118
spikingjelly == 0.0.0.0.14
snntorch == 0.6.4
wandb == 0.16.1
numpy == 1.23.5
```

To install all the requirements, run:

```bash
pip install -r requirements.txt
```


## File Structure

- `data/`: contains the datasets used in the paper.
- `model/`: contains the models used in the paper.
- `attacks/`: contains the attacks used in the paper.
- `config/`: the configuration file to set the parameters.
- `utils/`: the utils folder.
- `main.py`: the main file to run the attack.
- `attacker.py`: the attacker class files.
- `requirements.txt`: the requirements file.
- `train.py`: train code.
- `README.md`: this file.

## Training
Use ```train.py``` and pass ```dataset.frame_number={x-y-z}``` to control the desired latencies.

## Attack
Use ```main.py``` and pass the same argument as above. If the model is trained on multiple latencies, then the Self-Ensemble defense is conducted automatically after each successful attack (on single latency). Please use ```attack.init_alpha_mode``` and ```attack.add_position_label_mode``` to control the attacker's behavior.
