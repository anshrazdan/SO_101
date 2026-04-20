# SO_101 Robot ACT Training Project

This repo contains the Python and MATLAB code for training an ACT-style action chunking policy on SO-101 robot demonstration data.

The model learns from:

- wrist camera images
- base camera images
- robot joint states
- text instruction labels
- future robot joint action chunks

The current task label in the dataset is:

```text
hang the towel
```

## Important Dataset Note

The training data is not stored in GitHub because it is very large. The `data/` folder must be shared separately, for example with Google Drive, OneDrive, Dropbox, or an external drive.

After cloning the repo and copying the dataset, the folder should look like this:

```text
SO_101/
  README.md
  requirements.txt
  project_architecture.md
  python_src/
  matlab/
  data/
    pilot_dataset/
      episode_1/
        metadata.csv
        frame_000.png
        base_frame_000.png
        ...
      episode_2/
        metadata.csv
        frame_000.png
        base_frame_000.png
        ...
      ...
```

Each episode folder contains:

- `metadata.csv`: joint state, action, timestamp, instruction, and image filenames
- `frame_*.png`: wrist camera images
- `base_frame_*.png`: base camera images

The joint columns in `metadata.csv` are:

```text
state_0, state_1, state_2, state_3, state_4, state_5
action_0, action_1, action_2, action_3, action_4, action_5
```

`state_*` columns are the robot joint positions at that frame.

`action_*` columns are the joint movement/action targets.

## Requirements

Install:

- Python 3.10 or newer
- Git
- A Python virtual environment
- The Python packages in `requirements.txt`

The Python dependencies are:

```text
torch
opencv-python
lerobot
```

If you are only training/evaluating from an existing dataset, you need `torch` and `opencv-python`.

If you are recording new robot data, you also need `lerobot` and access to the SO-101 robot hardware.

## Setup On A New Computer

Clone the repo:

```powershell
git clone https://github.com/anshrazdan/SO_101.git
cd SO_101
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Copy the shared `data/` folder into the repo root.

Confirm this path exists:

```text
SO_101/data/pilot_dataset/
```

## Training The ACT Model

Train the ACT-style action chunking model:

```powershell
python python_src/training/train_real.py
```

The training script uses all episode folders inside:

```text
data/pilot_dataset/
```

The model predicts a chunk of 10 future actions. Each action has 6 joint values, so the model output shape is:

```text
batch_size x 10 x 6
```

After training, the model weights are saved to:

```text
data/act_model.pth
```

Training loss is saved to:

```text
data/real_training_loss.csv
```

## Evaluating The ACT Model

After training, run:

```powershell
python python_src/training/evaluate_real_model.py
```

This writes predictions to:

```text
data/act_model_predictions.csv
```

The CSV includes:

- true next action
- predicted next action
- predicted 10-step action chunk

## MATLAB Plotting

From MATLAB, use these scripts:

```text
matlab/plot_real_training_loss.m
matlab/plot_prediction_vs_truth.m
matlab/plot_predictions_vs_truth.m
matlab/view_pilot_images.m
```

The prediction plotting scripts expect:

```text
data/act_model_predictions.csv
```

The training loss plot expects:

```text
data/real_training_loss.csv
```

## Recording New Data

To record a teleoperated SO-101 episode:

```powershell
python python_src/utils/teleop_record_episode.py
```

Default hardware settings in the script:

```text
leader port: COM5
follower port: COM6
wrist camera index: 1
base camera index: none unless provided
```

To include a base camera:

```powershell
python python_src/utils/teleop_record_episode.py --base-camera-index 0
```

To change the wrist camera:

```powershell
python python_src/utils/teleop_record_episode.py --wrist-camera-index 2
```

New episodes are saved under:

```text
data/pilot_dataset/episode_<number>/
```

## Useful Utility Scripts

Preview cameras:

```powershell
python python_src/utils/preview_cameras.py
```

List cameras:

```powershell
python python_src/list_cameras.py
```

Read robot joint state:

```powershell
python python_src/utils/read_joint_state.py
```

Analyze dataset metadata:

```powershell
python python_src/utils/analyze_metadata.py
```

Clean duplicate idle metadata rows for episode 1:

```powershell
python python_src/utils/clean_metadata.py
```

## Files Not Included In GitHub

These are intentionally ignored:

```text
data/
.venv/
matlab_venv/
.vscode/
```

`data/` is ignored because the dataset is very large. Share it separately.

Virtual environments are ignored because each computer should create its own environment.

## Common Problems

### Missing metadata file

If training fails with a missing metadata error, the dataset is probably not in the right place.

Make sure this exists:

```text
data/pilot_dataset/episode_1/metadata.csv
```

### Missing Python package

If Python says a package is missing, activate the virtual environment and reinstall:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### GitHub repo does not include data

That is expected. The dataset folder is ignored and must be shared separately.

### GitHub password does not work

GitHub no longer accepts normal passwords for Git pushes over HTTPS. Use a GitHub Personal Access Token instead.

## Main Project Files

```text
python_src/datasets/real_robot_dataset.py
```

Loads episode data, images, robot states, and ACT action chunks.

```text
python_src/training/train_real.py
```

Trains the ACT-style action chunking transformer.

```text
python_src/training/evaluate_real_model.py
```

Evaluates the trained ACT model and exports predictions.

```text
python_src/utils/teleop_record_episode.py
```

Records new robot demonstration episodes.

```text
matlab/
```

MATLAB scripts for plotting, inspecting, and reporting results.
