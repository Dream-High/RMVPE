import os

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from src import MIR1K, E2E, cycle, summary, SAMPLE_RATE, FL
from evaluate import evaluate


def train(alpha, gamma):
    logdir = 'runs/Pitch_FL' + str(alpha) + '_' + str(gamma)
    seq_l = 2.55

    hop_length = 20

    learning_rate = 5e-4
    batch_size = 16
    clip_grad_norm = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MIR1K('/cache/whj/dataset/MIR1K', hop_length, seq_l, ['train'])
    print(len(train_dataset))
    validation_dataset = MIR1K('/cache/whj/dataset/MIR1K', hop_length, None, ['test'])
    print(len(validation_dataset))

    data_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    print(len(data_loader))
    validation_interval = len(data_loader)
    iterations = len(data_loader) * 100
    learning_rate_decay_steps = len(data_loader) * 5
    learning_rate_decay_rate = 0.98
    resume_iteration = None

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    if resume_iteration is None:
        model = nn.DataParallel(E2E(int(hop_length / 1000 * SAMPLE_RATE), 4, 1, (2, 2))).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    summary(model)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    RPA, RCA, OA, VFA, VR, it = 0, 0, 0, 0, 0, 0

    for i, data in zip(loop, cycle(data_loader)):
        audio = data['audio'].to(device)
        pitch_label = data['pitch'].to(device)
        pitch_pred = model(audio)
        loss = FL(pitch_pred, pitch_label, alpha, gamma)

        print(i, end='\t')
        print('loss_total:', loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        scheduler.step()
        writer.add_scalar('loss/loss_pitch', loss.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                metrics = evaluate(validation_dataset, model.module, hop_length, device)
                for key, value in metrics.items():
                    writer.add_scalar('stage_pitch/' + key, np.mean(value), global_step=i)
                rpa = np.mean(metrics['RPA'])
                rca = np.mean(metrics['RCA'])
                oa = np.mean(metrics['OA'])
                vr = np.mean(metrics['VR'])
                vfa = np.mean(metrics['VFA'])
                if rpa >= RPA:
                    RPA, RCA, OA, VR, VFA, it = rpa, rca, oa, vr, vfa, i
                    with open(os.path.join(logdir, 'result.txt'), 'a') as f:
                        f.write(str(i) + '\t')
                        f.write(str(RPA) + '\t')
                        f.write(str(RCA) + '\t')
                        f.write(str(OA) + '\t')
                        f.write(str(VR) + '\t')
                        f.write(str(VFA) + '\n')
                    torch.save(model.module, os.path.join(logdir, f'model-1-{i}.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
            model.train()

        if i - it > len(data_loader) * 10:
            break


alpha_list = [6, 7, 8, 9, 10]
for alpha in alpha_list:
    print('' * 250)
    print(alpha)
    print('' * 250)
    train(alpha, 0)
