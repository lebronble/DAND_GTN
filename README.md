# DAND_GTN

PyTorch implementation of DAND-GTN for emotion gait recognition experiments.

## Project Structure

- `main.py`: training and evaluation entry point.
- `model/`: DAND-GTN model and loss-fusion modules.
- `feeders/`: data loading utilities.
- `config/`: experiment configuration files.

## Environment

The current `requirements.txt` is a Conda environment specification. Create the environment with:

```bash
conda env create -f requirements.txt
conda activate DAND_GTN
```

## Data

Dataset files are not committed to Git. Place them under `datasets/` following the paths in the config files:

- `config/train_emotion_gait.yaml`
- `config/train_elmd.yaml`

## Training

Emotion gait:

```bash
python main.py --config config/train_emotion_gait.yaml
```

ELMD:

```bash
python main.py --config config/train_elmd.yaml
```

Training outputs are written to `runs/` and `work_dir/`, which are ignored by Git.
