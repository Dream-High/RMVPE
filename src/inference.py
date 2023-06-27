import numpy as np
import torch


class Inference:
    def __init__(self, model, seg_len, seg_frames, hop_length, batch_size, device):
        super(Inference, self).__init__()
        self.model = model.eval()
        self.seg_len = seg_len
        self.seg_frames = seg_frames
        self.batch_size = batch_size
        self.hop_length = hop_length
        self.device = device

    def inference(self, audio):
        with torch.no_grad():
            padded_audio = self.pad_audio(audio)
            segments = self.en_frame(padded_audio)
            hidden_vec_segments, out_segments = self.forward_in_mini_batch(self.model, segments)
            out_segments = self.de_frame(out_segments, type_seg='pitch')[:(len(audio) // self.hop_length + 1)]
            hidden_vec_segments = self.de_frame(hidden_vec_segments, type_seg='pitch')[:(len(audio) // self.hop_length + 1)]
            return hidden_vec_segments, out_segments

    def pad_audio(self, audio):
        audio_len = len(audio)
        seg_nums = int(np.ceil(audio_len / self.seg_len)) + 1
        pad_len = seg_nums * self.seg_len - audio_len + self.seg_len // 2
        padded_audio = torch.cat([torch.zeros(self.seg_len // 4).to(self.device), audio,
                                  torch.zeros(pad_len - self.seg_len // 4).to(self.device)])
        return padded_audio

    def en_frame(self, audio):
        audio_len = len(audio)
        assert audio_len % (self.seg_len // 2) == 0

        audio = torch.cat([torch.zeros(1024).to(self.device), audio, torch.zeros(1024).to(self.device)])
        segments = []
        start = 0
        while start + self.seg_len <= audio_len:
            segments.append(audio[start:start + self.seg_len + 2048])
            start += self.seg_len // 2
        segments = torch.stack(segments, dim=0)
        return segments

    def forward_in_mini_batch(self, model, segments):
        hidden_vec_segments = []
        out_segments = []
        segments_num = segments.shape[0]
        batch_start = 0
        while True:
            # print('#', end='\t')
            if batch_start + self.batch_size >= segments_num:
                batch_tmp = segments[batch_start:].shape[0]
                segment_in = torch.cat([segments[batch_start:],
                                        torch.zeros_like(segments)[:self.batch_size-batch_tmp].to(self.device)], dim=0)
                hidden_vec, out_tmp = model(segment_in)
                hidden_vec_segments.append(hidden_vec[:batch_tmp])
                out_segments.append(out_tmp[:batch_tmp])
                break
            else:
                segment_in = segments[batch_start:batch_start+self.batch_size]
                hidden_vec, out_tmp = model(segment_in)
                hidden_vec_segments.append(hidden_vec)
                out_segments.append(out_tmp)
            batch_start += self.batch_size
        hidden_vec_segments = torch.cat(hidden_vec_segments, dim=0)
        out_segments = torch.cat(out_segments, dim=0)
        return hidden_vec_segments, out_segments

    def de_frame(self, segments, type_seg='audio'):
        output = []
        if type_seg == 'audio':
            for segment in segments:
                output.append(segment[self.seg_len // 4: int(self.seg_len * 0.75)])
        else:
            for segment in segments:
                output.append(segment[self.seg_frames // 4: int(self.seg_frames * 0.75)])
        output = torch.cat(output, dim=0)
        return output
