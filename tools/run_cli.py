import os
from argparse import ArgumentParser
from pathlib import Path

import pyrootutils
import torch
from loguru import logger
import numpy as np
import soundfile

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.vqgan.inference import load_model as load_decoder_model
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
from tools.webui import build_app
from tools.webui.inference import get_inference_wrapper
from fish_speech.utils.file import audio_to_bytes

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.5",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="firefly_gan_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--theme", type=str, default="light")
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--audio-file", type=str, default=None)
    parser.add_argument("--reference-text-file", type=str, default=None)
    parser.add_argument("--reference-audio-file", type=str, default=None)
    

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    # Check if MPS or CUDA is available
    if torch.backends.mps.is_available():
        args.device = "mps"
        logger.info("mps is available, running on mps.")
    elif not torch.cuda.is_available():
        logger.info("CUDA is not available, running on CPU.")
        args.device = "cpu"

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    logger.info("Loading VQ-GAN model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("Decoder model loaded, warming up...")

    # Create the inference engine
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )


    logger.info(f"Text File {args.text_file}")
    logger.info(f"Audio File {args.audio_file}")
    logger.info(f"Reference Text File {args.reference_text_file}")
    logger.info(f"Reference Audio File {args.reference_audio_file}")

    text = "Hello, how are you?"
    if args.text_file:
        with open(args.text_file, 'r') as file:
            text = file.read()

    reference_text = ""
    if args.reference_text_file:
        with open(args.reference_text_file, 'r') as file:
            reference_text = file.read()            
    

    results = inference_engine.inference(
            ServeTTSRequest(
                text=text,
                references=[
                    ServeReferenceAudio(
                        text=reference_text,
                        audio=audio_to_bytes(args.reference_audio_file),
                    )
                ],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                format="wav",
            )
        )
    for result in results:
        logger.info(f"Result: code: {result.code}, error: {result.error}")
        if result.code == 'final' and result.audio is not None:
            soundfile.write(args.audio_file, result.audio[1], result.audio[0])
            



    
