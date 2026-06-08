#!/usr/bin/env python3
"""
MiMo-V2.5-ASR 轉錄腳本
將長音頻切成固定段落分別轉錄，輸出 SRT 和 TXT 格式
"""
import sys
import os
import time
import argparse

import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mimo_audio.mimo_audio import MimoAudio


def format_time_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_audio_mono(path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr


def split_into_chunks(wav: torch.Tensor, sr: int, chunk_sec: float) -> list[tuple[float, float, torch.Tensor]]:
    """回傳 list of (start_sec, end_sec, chunk_tensor)"""
    total = wav.shape[0]
    chunk_samples = int(chunk_sec * sr)
    chunks = []
    pos = 0
    while pos < total:
        end = min(pos + chunk_samples, total)
        start_sec = pos / sr
        end_sec = end / sr
        chunks.append((start_sec, end_sec, wav[pos:end]))
        pos = end
    return chunks


def transcribe_file(
    audio_path: str,
    model_path: str,
    tokenizer_path: str,
    chunk_sec: float = 30.0,
    language: str = "auto",
    output_dir: str | None = None,
):
    # 選裝置
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[裝置] {device}")

    # 載入模型
    print("[1/4] 載入 MiMo-V2.5-ASR 模型...")
    t0 = time.time()
    model = MimoAudio(model_path, tokenizer_path, device=device)
    print(f"      模型載入完成，耗時 {time.time()-t0:.1f}s")

    # 語言 tag
    tag_map = {"zh": "<chinese>", "en": "<english>", "auto": ""}
    audio_tag = tag_map.get(language, "")
    print(f"[2/4] 語言 tag: '{audio_tag or 'Auto'}'")

    # 載入並切段音頻
    print(f"[3/4] 載入音頻：{audio_path}")
    wav, sr = load_audio_mono(audio_path)
    total_sec = wav.shape[0] / sr
    print(f"      總時長：{total_sec/60:.1f} 分鐘，每段 {chunk_sec}s")
    chunks = split_into_chunks(wav, sr, chunk_sec)
    print(f"      共 {len(chunks)} 段")

    # 逐段轉錄
    print(f"[4/4] 開始逐段轉錄...")
    import gc
    results = []  # list of (start, end, text)
    for i, (start_sec, end_sec, chunk_wav) in enumerate(chunks):
        seg_start = time.time()
        print(f"  [{i+1:3d}/{len(chunks)}] {format_time_srt(start_sec)} → {format_time_srt(end_sec)}", end="", flush=True)

        # 暫存到記憶體，透過 tensor 直接傳入
        text = model.asr_sft(chunk_wav, audio_tag=audio_tag)
        text = text.strip()

        elapsed = time.time() - seg_start
        print(f"  ({elapsed:.1f}s)  {text[:50]}{'...' if len(text)>50 else ''}")

        if text:
            results.append((start_sec, end_sec, text))

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    # 輸出路徑
    base_name = os.path.basename(audio_path)
    out_dir = output_dir or os.path.dirname(audio_path)
    srt_path = os.path.join(out_dir, base_name + ".mimo.srt")
    txt_path = os.path.join(out_dir, base_name + ".mimo.txt")

    # 寫 SRT
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(results, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_time_srt(start)} --> {format_time_srt(end)}\n")
            f.write(f"{text}\n\n")

    # 寫 TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        for _, _, text in results:
            f.write(text + "\n")

    print(f"\n✅ 完成！")
    print(f"   SRT → {srt_path}")
    print(f"   TXT → {txt_path}")
    print(f"   字幕數：{len(results)}")
    return srt_path, txt_path


def main():
    parser = argparse.ArgumentParser(description="MiMo-V2.5-ASR 轉錄為 SRT/TXT")
    parser.add_argument("audio", help="音頻檔案路徑")
    parser.add_argument("--model-path", default="./models/MiMo-V2.5-ASR")
    parser.add_argument("--tokenizer-path", default="./models/MiMo-Audio-Tokenizer")
    parser.add_argument("--chunk-sec", type=float, default=300.0, help="每段秒數（預設300秒）")
    parser.add_argument("--language", choices=["auto", "zh", "en"], default="zh")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    transcribe_file(
        audio_path=args.audio,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        chunk_sec=args.chunk_sec,
        language=args.language,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
