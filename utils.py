import torch
import glob2

def get_data(data_root, class_list):
    train_list = []
    for char in class_list:
        train_path = glob2.glob(os.path.join(data_root, char, "**", "*.png")) + glob2.glob(os.path.join(data_root, char, "**", "*.jpg"))
        train_list +=train_path

    return train_list

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def save(results_folder, milestone, step, model, ema_model):
    data = {
        'step': step,
        'model': model.state_dict(),
        'ema': ema_model.state_dict()
    }
    torch.save(data, str(results_folder / f'model-{milestone}.pt'))