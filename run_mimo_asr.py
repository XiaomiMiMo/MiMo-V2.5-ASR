# Copyright 2025 Xiaomi Corporation.
import argparse
import os
import time

import gradio as gr
import torch

from src.mimo_audio.mimo_audio import MimoAudio
from src.memory_guard import (
    check_memory_before_load,
    format_memory_status,
    MODEL_MIN_REQUIRED_GB,
    MODEL_RECOMMENDED_GB,
)


LANGUAGE_TAGS = {
    "Auto": "",
    "Chinese": "<chinese>",
    "English": "<english>",
}


class ASRGenerator:
    def __init__(self, model):
        self.model = model

    def transcribe(self, audio_path, audio_tag=""):
        return self.model.asr_sft(audio_path, audio_tag=audio_tag)


class MiMoV25ASRInterface:
    def __init__(self):
        self.model = None
        self.asr_generator = None
        self.device = None
        self.model_initialized = False

    def initialize_model(self, model_path=None, tokenizer_path=None):
        try:
            # ── Memory check (before loading anything) ─────────────────────────
            mem_check = check_memory_before_load(
                required_gb=MODEL_MIN_REQUIRED_GB,
                recommended_gb=MODEL_RECOMMENDED_GB,
            )
            print(mem_check.format_report())

            if not mem_check.ok:
                # Critical — refuse to load to avoid OOM crash
                error_msg = (
                    f"[記憶體不足，拒絕載入]\n"
                    f"{mem_check.message}\n"
                    f"{mem_check.details}\n\n"
                    f"請先關閉其他佔用大量記憶體的應用程式後再試。"
                )
                print(error_msg)
                return error_msg

            if mem_check.level == "warning":
                print(f"[記憶體警告] {mem_check.message}")

            # ── Device selection ───────────────────────────────────────────────
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            if not model_path:
                model_path = "./models/MiMo-V2.5-ASR"
            if not tokenizer_path:
                tokenizer_path = "./models/MiMo-Audio-Tokenizer"

            print(f"Model path: {model_path}")
            print(f"Tokenizer path: {tokenizer_path}")
            print(f"Device: {self.device}")

            self.model = MimoAudio(model_path, tokenizer_path, device=str(self.device))
            self.asr_generator = ASRGenerator(self.model)

            self.model_initialized = True
            success_msg = f"模型載入成功！\n{mem_check.message}"
            print(success_msg)
            return success_msg

        except MemoryError as e:
            error_msg = (
                f"記憶體不足，載入中止：{str(e)}\n"
                f"請關閉其他應用程式後重試。"
            )
            print(error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"Model loading failed: {str(e)}"
            print(error_msg)
            return error_msg

    def transcribe(self, uploaded_audio, recorded_audio, language_choice):
        if not self.model_initialized:
            yield "", "Error: Model not initialized, please load the model first."
            return

        audio_path = uploaded_audio or recorded_audio
        if audio_path is None:
            yield "", "Error: Please upload an audio file or record from your microphone."
            return

        audio_tag = LANGUAGE_TAGS.get(language_choice, "")

        try:
            print(f"Performing chunked ASR task:")
            print(f"  Audio: {audio_path}")
            print(f"  Language: {language_choice} (tag='{audio_tag}')")

            import torchaudio
            import gc
            from src.memory_guard import get_process_memory_gb

            yield "", "正在載入與處理音訊檔案..."
            wav, sr = torchaudio.load(audio_path)
            if wav.ndim == 2:
                wav = wav.mean(dim=0)
            
            target_sr = 16000
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            
            total_duration = wav.shape[0] / target_sr
            print(f"Audio duration: {total_duration:.2f} seconds")

            # Split into chunks of 300 seconds
            chunk_sec = 300.0
            chunk_samples = int(chunk_sec * target_sr)
            total_samples = wav.shape[0]
            
            chunks = []
            pos = 0
            while pos < total_samples:
                end = min(pos + chunk_samples, total_samples)
                chunks.append(wav[pos:end])
                pos = end

            num_chunks = len(chunks)
            print(f"Split into {num_chunks} chunks.")

            transcripts = []
            start_time = time.time()

            for i, chunk_wav in enumerate(chunks):
                chunk_start_sec = i * chunk_sec
                chunk_end_sec = min((i + 1) * chunk_sec, total_duration)

                proc_mem = get_process_memory_gb()
                status_msg = (
                    f"正在轉錄段落 {i+1}/{num_chunks} ({chunk_start_sec:.1f}s ~ {chunk_end_sec:.1f}s)...\n"
                    f"已用時間: {time.time() - start_time:.1f}s\n"
                    f"記憶體佔用: {proc_mem:.1f} GB"
                )
                yield " ".join(transcripts), status_msg

                # Transcribe the chunk
                chunk_text = self.asr_generator.transcribe(chunk_wav, audio_tag=audio_tag)
                chunk_text = chunk_text.strip()

                if chunk_text:
                    transcripts.append(chunk_text)

                # Regular memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()

            final_transcript = " ".join(transcripts)
            elapsed = time.time() - start_time
            proc_mem = get_process_memory_gb()
            status_msg = (
                f"轉錄完成！共 {num_chunks} 個段落，總耗時 {elapsed:.2f} 秒。\n"
                f"音訊總長度: {total_duration:.1f} 秒\n"
                f"最終記憶體佔用: {proc_mem:.1f} GB"
            )
            yield final_transcript, status_msg

        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            print(error_msg)
            yield "", error_msg

    def create_interface(self, default_model_path="", default_tokenizer_path=""):
        import inspect
        blocks_kwargs = {"title": "MiMo-V2.5-ASR Speech Recognition"}
        theme_obj = gr.themes.Soft()
        if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
            blocks_kwargs["theme"] = theme_obj
            theme_to_launch = None
        else:
            theme_to_launch = theme_obj

        with gr.Blocks(**blocks_kwargs) as iface:
            iface.theme_to_launch = theme_to_launch
            gr.Markdown("# MiMo-V2.5-ASR: Robust Speech Recognition")
            gr.Markdown(
                "Upload an audio file **or** record directly from your microphone. "
                "Supports Chinese, English, Chinese dialects, code-switch, singing, "
                "noisy environments, and multi-speaker scenarios."
            )

            # ── Memory status banner ─────────────────────────────────────────
            with gr.Accordion("🖥️ 系統記憶體狀態", open=True):
                mem_display = gr.Textbox(
                    label="記憶體監控",
                    value=format_memory_status(),
                    interactive=False,
                    lines=10,
                )
                mem_refresh_btn = gr.Button("🔄 更新記憶體狀態", size="sm")
                mem_refresh_btn.click(
                    fn=format_memory_status,
                    outputs=[mem_display],
                )

            with gr.Tabs():
                with gr.TabItem("Model Configuration"):
                    gr.Markdown("### Model initialization configuration")

                    with gr.Row():
                        with gr.Column():
                            model_path = gr.Textbox(
                                label="Model path",
                                placeholder="Leave blank to use default path: ./models/MiMo-V2.5-ASR",
                                value=default_model_path,
                                lines=2,
                            )
                            tokenizer_path = gr.Textbox(
                                label="Tokenizer path",
                                placeholder="Leave blank to use default path: ./models/MiMo-Audio-Tokenizer",
                                value=default_tokenizer_path,
                                lines=2,
                            )
                            init_btn = gr.Button(
                                "Initialize model", variant="primary", size="lg"
                            )

                        with gr.Column():
                            init_status = gr.Textbox(
                                label="初始化狀態",
                                interactive=False,
                                lines=8,
                                placeholder="點擊『初始化模型』按鈕開始載入...",
                            )
                            gr.Markdown("### 系統資訊")
                            gpu_available = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
                            device_name = "CUDA GPU" if torch.cuda.is_available() else ("Apple Silicon MPS (統一記憶體)" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "CPU")
                            import platform
                            gr.Textbox(
                                label="裝置資訊",
                                value=(
                                    f"加速器: {'是' if gpu_available else '否'} — {device_name}\n"
                                    f"架構: {platform.machine()} / {platform.system()}\n"
                                    f"模型最低記憶體需求: {MODEL_MIN_REQUIRED_GB:.0f} GB\n"
                                    f"模型建議記憶體需求: {MODEL_RECOMMENDED_GB:.0f} GB"
                                ),
                                interactive=False,
                                lines=5,
                            )

                with gr.TabItem("Speech Recognition"):
                    gr.Markdown("### Automatic Speech Recognition")

                    with gr.Row():
                        with gr.Column():
                            uploaded_audio = gr.Audio(
                                label="Upload Audio File",
                                type="filepath",
                                sources=["upload"],
                                interactive=True,
                            )
                            recorded_audio = gr.Audio(
                                label="Or Record from Microphone",
                                type="filepath",
                                sources=["microphone"],
                                interactive=True,
                            )
                            language_choice = gr.Radio(
                                label="Language Tag",
                                choices=list(LANGUAGE_TAGS.keys()),
                                value="Auto",
                                info=(
                                    "Auto: automatic language detection (recommended for "
                                    "code-switched speech). Select Chinese or English to "
                                    "bias the model toward that language."
                                ),
                            )
                            transcribe_btn = gr.Button(
                                "Transcribe", variant="primary", size="lg"
                            )

                        with gr.Column():
                            import inspect
                            textbox_kwargs = {
                                "label": "Transcription",
                                "lines": 10,
                                "interactive": False,
                                "placeholder": "Transcription result will appear here...",
                            }
                            if "show_copy_button" in inspect.signature(gr.Textbox.__init__).parameters:
                                textbox_kwargs["show_copy_button"] = True
                            elif "buttons" in inspect.signature(gr.Textbox.__init__).parameters:
                                textbox_kwargs["buttons"] = ["copy"]
                            output_text = gr.Textbox(**textbox_kwargs)
                            status = gr.Textbox(
                                label="Status",
                                lines=4,
                                interactive=False,
                                placeholder="Processing status will be shown here...",
                            )
                            with gr.Row():
                                clear_btn = gr.Button("Clear", size="sm")

            def _init_and_refresh_mem(p, t):
                status_msg = self.initialize_model(p or None, t or None)
                mem_status = format_memory_status()
                return status_msg, mem_status

            init_btn.click(
                fn=_init_and_refresh_mem,
                inputs=[model_path, tokenizer_path],
                outputs=[init_status, mem_display],
            )

            transcribe_btn.click(
                fn=self.transcribe,
                inputs=[uploaded_audio, recorded_audio, language_choice],
                outputs=[output_text, status],
            )

            def clear_all():
                return None, None, "Auto", "", ""

            clear_btn.click(
                fn=clear_all,
                outputs=[
                    uploaded_audio,
                    recorded_audio,
                    language_choice,
                    output_text,
                    status,
                ],
            )

        return iface


def main():
    parser = argparse.ArgumentParser(description="MiMo-V2.5-ASR Gradio Demo")
    parser.add_argument("--model-path", default=None, help="Path to the MiMo ASR model")
    parser.add_argument("--tokenizer-path", default=None, help="Path to the MiMo audio tokenizer")
    parser.add_argument("--host", default="127.0.0.1", help="Server address")
    parser.add_argument("--port", type=int, default=7898, help="Port")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print("Launch MiMo-V2.5-ASR demo...")
    interface = MiMoV25ASRInterface()

    if args.model_path or args.tokenizer_path:
        print("Initializing model from command-line paths...")
        print(interface.initialize_model(args.model_path, args.tokenizer_path))

    print("Create Gradio interface...")
    iface = interface.create_interface(
        default_model_path=args.model_path or "",
        default_tokenizer_path=args.tokenizer_path or "",
    )

    print(f"Launch service - {args.host}:{args.port}")
    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "debug": args.debug,
    }
    if hasattr(iface, "theme_to_launch") and iface.theme_to_launch is not None:
        launch_kwargs["theme"] = iface.theme_to_launch

    iface.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
