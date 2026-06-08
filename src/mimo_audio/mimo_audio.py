# Copyright 2025 Xiaomi Corporation.
import time
import random
import torch
import torchaudio

from typing import Union
from torchaudio.transforms import MelSpectrogram
from transformers import (
    AutoTokenizer,
    GenerationConfig
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .process_speechdata import InputSegment
from ..mimo_audio_tokenizer import MiMoAudioTokenizer
from .templates import asr_en_templates, asr_zh_templates
from .modeling_mimo_audio import (
    MiMoAudioArguments,
    MiMoAudioForCausalLM,
    MiMoSampler,
    MiMoStopper,
)


class MimoAudio:

    def __init__(
        self,
        model_path: str,
        mimo_audio_tokenizer_path: str,
        device: str | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.path = model_path
        self.mimo_audio_tokenizer_path = mimo_audio_tokenizer_path

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            self.path
        )
        self.padding_idx = int(self.tokenizer.pad_token_id)

        special_tokens = [
            "<|sosp|>",
            "<|eosp|>",
            "<|empty|>",
            "<|Human|>",
            "<|SpeechLM|>",
            "<|sostm|>",
            "<|eostm|>",
            "<|eot|>",
        ]
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                print(f"Add special tokens {token} to tokenizer.vocab")
                self.tokenizer.add_tokens([token], special_tokens=True)

        self.sosp_idx = self.tokenizer.convert_tokens_to_ids("<|sosp|>")
        self.eosp_idx = self.tokenizer.convert_tokens_to_ids("<|eosp|>")
        self.empty_token = self.tokenizer.convert_tokens_to_ids("<|empty|>")
        self.sostm_idx = self.tokenizer.convert_tokens_to_ids("<|sostm|>")
        self.eostm_idx = self.tokenizer.convert_tokens_to_ids("<|eostm|>")
        self.eot_idx = self.tokenizer.convert_tokens_to_ids("<|eot|>")
        self.im_start_idx = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_idx = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        model_args = MiMoAudioArguments(
            model_name_or_path=self.path,
            sosp_idx=self.sosp_idx,
            eosp_idx=self.eosp_idx,
            empty_idx=self.empty_token,
            sostm_idx=self.sostm_idx,
            eostm_idx=self.eostm_idx,
            eot_idx=self.eot_idx,
        )

        start_loading_time = time.monotonic()
        self.model = MiMoAudioForCausalLM.from_pretrained(
            self.path,
            args=model_args,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        )

        self.group_size=self.model.config.group_size
        self.audio_channels=self.model.config.audio_channels
        self.delay_pattern = self.model.config.delay_pattern
        self.vocab_size = self.model.config.vocab_size

        self.speech_zeroemb_idx = self.model.speech_empty_ids

        self.model.eval()
        print(
            f"Model loaded in {time.monotonic() - start_loading_time:.2f} seconds, device: {self.device}"
        )

        self.generate_kwargs = {
            "max_length": 8192,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        self.default_global_sampler = MiMoSampler(
            do_sample=True, temperature=0.6, top_k=50, top_p=0.95
        )
        self.default_local_sampler = MiMoSampler(
            do_sample=True, temperature=0.9, top_k=50, top_p=0.95
        )

        self.task_sampler_configs = {
            "asr": {
                "global": MiMoSampler(do_sample=False, temperature=1.0, top_p=1.0),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95)
            },
        }

        start_loading_mimo_audio_tokenizer_time = time.monotonic()
        self.mimo_audio_tokenizer = MiMoAudioTokenizer.from_pretrained(self.mimo_audio_tokenizer_path)

        self.mimo_audio_tokenizer.eval().bfloat16().to(self.device)
        print(
            f"MiMo-Audio Tokenizer loaded in {time.monotonic() - start_loading_mimo_audio_tokenizer_time:.2f} seconds, device: {self.device}"
        )

        # Initialize mel spectrogram transform for consistent processing
        self.mel_transform = MelSpectrogram(
            sample_rate=self.mimo_audio_tokenizer.config.sampling_rate,
            n_fft=self.mimo_audio_tokenizer.config.nfft,
            hop_length=self.mimo_audio_tokenizer.config.hop_length,
            win_length=self.mimo_audio_tokenizer.config.window_size,
            f_min=self.mimo_audio_tokenizer.config.fmin,
            f_max=self.mimo_audio_tokenizer.config.fmax,
            n_mels=self.mimo_audio_tokenizer.config.n_mels,
            power=1.0,
            center=True,
        ).to(self.device)

    def get_task_sampler(self, task_name):
        if task_name not in self.task_sampler_configs:
            return {
                "global": self.default_global_sampler,
                "local": self.default_local_sampler
            }
        return self.task_sampler_configs[task_name]

    def wav2mel(self, wav):
        spec = self.mel_transform(wav[None, :])
        return torch.log(torch.clip(spec, min=1e-7)).squeeze()

    def resample_audio_if_needed(self, wav_tensor: torch.Tensor, original_sr: int):
        target_sr = self.mimo_audio_tokenizer.config.sampling_rate
        if original_sr != target_sr:
            wav_tensor = torchaudio.functional.resample(
                wav_tensor, original_sr, target_sr
            )
        return wav_tensor

    def group_by_length(self, features: torch.Tensor, lengths: torch.Tensor, max_length: int):
        if features.size(0) != lengths.sum().item():
            raise ValueError(f"Feature size mismatch: {features.size(0)} vs {lengths.sum().item()}")

        split_points = []
        current_sum = 0

        for i, seq_len in enumerate(lengths):
            if current_sum + seq_len > max_length and current_sum > 0:
                split_points.append(i)
                current_sum = seq_len.item()
            else:
                current_sum += seq_len.item()

        # Convert split points to group sizes
        group_sizes = []
        prev = 0
        for point in split_points:
            group_sizes.append(point - prev)
            prev = point
        if prev < len(lengths):
            group_sizes.append(len(lengths) - prev)

        len_groups = torch.split(lengths, group_sizes)
        feature_sizes = [group.sum().item() for group in len_groups]
        feature_groups = torch.split(features, feature_sizes)

        return feature_groups, len_groups

    def encode_batch(self, input_features: torch.Tensor, input_lens: torch.Tensor, max_length: int = 256000):
        feature_groups, len_groups = self.group_by_length(input_features, input_lens, max_length)

        encoded_parts = []
        for features, lengths in zip(feature_groups, len_groups):
            with torch.no_grad():
                codes, _ = self.mimo_audio_tokenizer.encoder.encode(
                    input_features=features.to(self.device),
                    input_lens=lengths.to(self.device),
                    return_codes_only=True
                )
                encoded_parts.append(codes)

        return torch.cat(encoded_parts, dim=-1)

    def preprocess_input(
        self,
        input: Union[str, torch.Tensor],
    ):
        if isinstance(input, torch.Tensor):
            wav = input
        else:
            wav, sr = torchaudio.load(input)
            if wav.ndim == 2:
                wav = wav.mean(dim=0)
            wav = self.resample_audio_if_needed(wav, sr)
        wav = wav.to(self.device)

        # Split waveform into 30s chunks, tokenize each separately, then concatenate codes
        target_sr = self.mimo_audio_tokenizer.config.sampling_rate
        chunk_samples = 30 * target_sr
        n_fft = self.mimo_audio_tokenizer.config.nfft

        total_samples = wav.shape[-1]
        code_parts = []
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            # Merge a too-short trailing chunk (would break mel reflect padding)
            # into the current one.
            if 0 < total_samples - end < n_fft:
                end = total_samples
            chunk = wav[start:end]
            # Zero-pad if the entire audio is shorter than n_fft.
            if chunk.shape[-1] < n_fft:
                chunk = torch.nn.functional.pad(chunk, (0, n_fft - chunk.shape[-1]))
            mel = self.wav2mel(chunk).transpose(0, 1)  # (seq_len, n_mels)
            codes_chunk = self.encode_batch(
                input_features=mel,
                input_lens=torch.tensor([mel.size(0)]),
            )
            code_parts.append(codes_chunk)
            start = end

        codes_packed = torch.cat(code_parts, dim=-1)
        codes = codes_packed.transpose(0, 1).detach().cpu()
        audio_codes = codes[:, :self.audio_channels]

        # Pad the sequence to be a multiple of group_size by repeating the last frame
        num_timesteps = audio_codes.shape[0]
        if num_timesteps % self.group_size != 0:
            padding_needed = self.group_size - (num_timesteps % self.group_size)
            last_tokens = audio_codes[-1:, :] # Keep dim for repeat
            padding_tokens = last_tokens.repeat(padding_needed, 1)
            audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)

        audio_tokenized = audio_codes.reshape(-1)

        return audio_tokenized

    def get_input_ids(self, prompt):
        input_ids = [
            seg.to_input_id(
                self.tokenizer,
                self.group_size,
                self.audio_channels,
            )
            for seg in prompt
        ]
        input_ids = torch.cat(input_ids, dim=1)
        return input_ids.to(self.device)


    def get_asr_sft_prompt(
        self,
        input: Union[None, str] = None,
        audio_tag="",
    ):
        audio_tokenized = self.preprocess_input(input)

        if '<chinese>' in audio_tag:
            template = random.choice(asr_zh_templates)
        elif '<english>' in audio_tag:
            template = random.choice(asr_en_templates)
        else:
            template = random.choice(asr_zh_templates + asr_en_templates)

        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=template,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<think>\n\n</think>\n{audio_tag}",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        ]
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids


    @torch.no_grad()
    def forward(
        self,
        input_ids,
        stopping_criteria=None,
        min_new_tokens=0,
        max_new_tokens=8192,
        task_name=None,
    ):

        task_sampler = self.get_task_sampler(task_name)

        generation_kwargs = self.generate_kwargs.copy()
        generation_config = GenerationConfig(**generation_kwargs)

        input_ids = input_ids.T.reshape(1, -1) # [B, flattened(T, audio_channels + 1)]

        prompt_length = input_ids.shape[1] // (self.audio_channels+1)

        max_length = prompt_length // self.group_size + max_new_tokens
        min_length = prompt_length // self.group_size + min_new_tokens

        if stopping_criteria is not None:
            for criterion in stopping_criteria:
                if isinstance(criterion, MiMoStopper):
                    criterion.max_length = max_length
                    criterion.min_length = min_length

        generated_ids = self.model.generate(
            input_ids,
            generation_config,
            stopping_criteria=stopping_criteria,
            global_sampler=task_sampler["global"],
            local_sampler=task_sampler["local"],
        )

        generated_ids = generated_ids.int().cpu().reshape(-1, self.audio_channels+1).T[:, prompt_length:]

        text = generated_ids[0, ::self.group_size][:-1]
        detokenized_text = self.tokenizer.decode(text, skip_special_tokens=False).strip().replace("<|empty|>", "").replace("<|eot|>", "").replace("<|eostm|>", "")
        print("Text channel:\t", detokenized_text)

        return detokenized_text

    def asr_sft(self, audio, audio_tag=""):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_asr_sft_prompt(audio, audio_tag)
        result = self.forward(input_ids, stopping_criteria=stopping_criteria, task_name="asr")
        if '<chinese>' in result or '<english>' in result:
            result = result.replace('<chinese>', '').replace('<english>', '').strip()
        return result
