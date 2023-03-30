import torch
import torchvision
import argparse
import copy
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False
from pathlib import Path

from utils import *
from dataset import Dataset
from model import Unet, GaussianDiffusion, EMA

def main():
    args = create_argparser().parse_args()
    device = torch.device('cuda:%d'%(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(device)

    with open(args.use_class) as f:
        use_list = [s.strip() for s in f.readlines()]

    print('num class : ', len(use_list))
    print('use use : ')
    print(use_list)

    train_list = get_data(use_list)
    print('train list : ', len(train_list))

    ds = Dataset(use_list, train_list, args.img_size)
    dl = cycle(torch.utils.data.DataLoader(ds, batch_size = args.batch_size, shuffle=True, num_workers=2, pin_memory=True))

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size = args.img_size,
        channels=3,
        classes = len(use_list),
        timesteps = args.timesteps,   # number of steps
        loss_type = args.loss_type    # L1 or L2
    ).to(device)

    ema = EMA(args.ema_decay)
    ema_model = copy.deepcopy(diffusion)
    ema_model.load_state_dict(diffusion.state_dict())


    #define constant
    step_start_ema = 2000
    save_and_sample_every = 1000
    update_ema_every = 10
    gradient_accumulate_every = args.gradient_accumulate
    train_num_steps = args.num_steps
    opt = torch.optim.Adam(diffusion.parameters(), lr=args.learing_late)
    step = 0

    # if args.fp16:
    #     (model, ema_model), opt = amp.initialize([model, ema_model], opt, opt_level='O1')

    results_folder = Path(args.save_foler)
    results_folder.mkdir(exist_ok = True)

    while step < train_num_steps:
        diffusion.train()
        for i in range(gradient_accumulate_every):
            data, label = next(dl)
            data, label = data.to(device), label.to(device)
            loss = diffusion(data, label)
            loss = loss / gradient_accumulate_every
            print(f'{step}: {loss.item()}')
            loss.backward()

        opt.step()
        opt.zero_grad()

        if step % update_ema_every == 0:
            if step < step_start_ema:
                ema_model.load_state_dict(diffusion.state_dict())
            else:
                ema.update_model_average(ema_model, diffusion)


        if step != 0 and step % save_and_sample_every == 0:
            diffusion.eval()
            milestone = step // save_and_sample_every
            batches = num_to_groups(36, args.batch_size)
            all_images_list = list(map(lambda n: ema_model.sample(batch_size=n), batches))
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            torchvision.utils.save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)
            save(results_folder, milestone, step, diffusion, ema_model)

        step += 1

    print('training completed')



def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument('-data_root', type=str, default="dataset")
    parser.add_argument('-use_class', type=str, default="class.txt")
    parser.add_argument("--save_foler", type=str, default="result")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--loss_type", type=str, default="l1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--gradient_accumulate", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    # parser.add_argument("--fp16", type=bool, default=False)

    return parser


if __name__ == '__main__':
    main()