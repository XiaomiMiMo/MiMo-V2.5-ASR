# Copyright 2025 Xiaomi Corporation.
import argparse
import os
import time

import gradio as gr
import torch

from src.mimo_audio.mimo_audio import MimoAudio


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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if not model_path:
                model_path = "./models/MiMo-V2.5-ASR"
            if not tokenizer_path:
                tokenizer_path = "./models/MiMo-Audio-Tokenizer"

            print(f"Model path: {model_path}")
            print(f"Tokenizer path: {tokenizer_path}")

            self.model = MimoAudio(model_path, tokenizer_path)
            self.asr_generator = ASRGenerator(self.model)

            self.model_initialized = True
            print("Model loaded successfully!")
            return "Model loaded successfully!"

        except Exception as e:
            error_msg = f"Model loading failed: {str(e)}"
            print(error_msg)
            return error_msg

    def transcribe(self, uploaded_audio, recorded_audio, language_choice):
        if not self.model_initialized:
            return "", "Error: Model not initialized, please load the model first."

        audio_path = uploaded_audio or recorded_audio
        if audio_path is None:
            return "", "Error: Please upload an audio file or record from your microphone."

        audio_tag = LANGUAGE_TAGS.get(language_choice, "")

        try:
            print(f"Performing ASR task:")
            print(f"  Audio: {audio_path}")
            print(f"  Language: {language_choice} (tag='{audio_tag}')")

            start = time.time()
            transcript = self.asr_generator.transcribe(audio_path, audio_tag=audio_tag)
            elapsed = time.time() - start

            status_msg = (
                f"Transcription completed in {elapsed:.2f}s\n"
                f"Input audio: {os.path.basename(audio_path)}\n"
                f"Language tag: {language_choice}"
            )
            return transcript, status_msg

        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            print(error_msg)
            return "", error_msg

    def create_interface(self, default_model_path="", default_tokenizer_path="", init_status_msg=""):
        with gr.Blocks(title="MiMo-V2.5-ASR Speech Recognition", theme=gr.themes.Soft()) as iface:
            gr.Markdown("# MiMo-V2.5-ASR: Robust Speech Recognition")
            gr.Markdown(
                "Upload an audio file **or** record directly from your microphone. "
                "Supports Chinese, English, Chinese dialects, code-switch, singing, "
                "noisy environments, and multi-speaker scenarios."
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
                                label="Initialization status",
                                interactive=False,
                                lines=6,
                                value=init_status_msg,
                                placeholder="Click the initialize model button to start...",
                            )
                            gr.Markdown("### System information")
                            gr.Textbox(
                                label="Device information",
                                value=f"GPU available: {'Yes' if torch.cuda.is_available() else 'No'}",
                                interactive=False,
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
                            output_text = gr.Textbox(
                                label="Transcription",
                                lines=10,
                                interactive=False,
                                placeholder="Transcription result will appear here...",
                                show_copy_button=True,
                            )
                            status = gr.Textbox(
                                label="Status",
                                lines=4,
                                interactive=False,
                                placeholder="Processing status will be shown here...",
                            )
                            with gr.Row():
                                clear_btn = gr.Button("Clear", size="sm")

            init_btn.click(
                fn=lambda p, t: self.initialize_model(p or None, t or None),
                inputs=[model_path, tokenizer_path],
                outputs=[init_status],
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
    parser.add_argument("--host", default="0.0.0.0", help="Server address")
    parser.add_argument("--port", type=int, default=7898, help="Port")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print("Launch MiMo-V2.5-ASR demo...")
    interface = MiMoV25ASRInterface()

    init_status_msg = ""
    if args.model_path or args.tokenizer_path:
        print("Initializing model from command-line paths...")
        init_status_msg = interface.initialize_model(args.model_path, args.tokenizer_path)
        print(init_status_msg)

    print("Create Gradio interface...")
    iface = interface.create_interface(
        default_model_path=args.model_path or "",
        default_tokenizer_path=args.tokenizer_path or "",
        init_status_msg=init_status_msg,
    )

    print(f"Launch service - {args.host}:{args.port}")
    iface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
