import numpy as np
import torch
from tqdm import tqdm

from collections import defaultdict
from src import to_local_average_cents, bce, SAMPLE_RATE, WINDOW_LENGTH
from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy
from mir_eval.melody import voicing_recall, voicing_false_alarm
import torch.nn.functional as F


def evaluate(dataset, model, hop_length, device, pitch_th=0.0):
    metrics = defaultdict(list)
    for data in dataset:
        audio = data['audio'].to(device)
        pitch_label = data['pitch'].to(device)
        pitch_pred = torch.zeros_like(pitch_label)
        slices = range(0, pitch_pred.shape[0], 128)
        for start_steps in slices:
            end_steps = start_steps + 127
            start = int(start_steps * hop_length * SAMPLE_RATE / 1000)
            end = int(end_steps * hop_length * SAMPLE_RATE / 1000) + WINDOW_LENGTH
            if end_steps >= pitch_pred.shape[0]:
                t_audio = F.pad(audio[start:end], (0, end - start - len(audio[start:end])), mode='constant')
                t_pitch_pred = model(t_audio.reshape((-1, t_audio.shape[-1]))).squeeze(0)
                pitch_pred[start_steps:end_steps + 1] = t_pitch_pred[:pitch_pred.shape[0] - end_steps - 1]
            else:
                t_audio = audio[start:end]
                t_pitch_pred = model(t_audio.reshape((-1, t_audio.shape[-1]))).squeeze(0)
                pitch_pred[start_steps:end_steps + 1] = t_pitch_pred

        loss = bce(pitch_pred, pitch_label)
        metrics['loss'].append(loss.item())

        cents_pred = to_local_average_cents(pitch_pred.cpu().numpy(), None, pitch_th)
        # cents_pred = to_viterbi_cents(pitch_pred.cpu().numpy())
        # print()
        cents_label = to_local_average_cents(pitch_label.cpu().numpy(), None, pitch_th)
        # cents_label = to_viterbi_cents(pitch_label.cpu().numpy())
        # print()

        freq_pred = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        freq = np.array([10 * (2 ** (cent / 1200)) if cent else 0 for cent in cents_label])

        time_slice = np.array([i*hop_length/1000 for i in range(len(cents_label))])
        ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, freq, time_slice, freq_pred)

        rpa = raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
        rca = raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
        oa = overall_accuracy(ref_v, ref_c, est_v, est_c)
        vfa = voicing_false_alarm(ref_v, est_v)
        vr = voicing_recall(ref_v, est_v)
        metrics['RPA'].append(rpa)
        metrics['RCA'].append(rca)
        metrics['OA'].append(oa)
        metrics['VFA'].append(vfa)
        metrics['VR'].append(vr)
        # if rpa < 0.9:
        print(data['file'], ':\t', rpa, '\t', oa)

    return metrics
