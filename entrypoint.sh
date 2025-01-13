#!/bin/bash

CUDA_ENABLED=${CUDA_ENABLED:-true}
DEVICE=""

if [ "${CUDA_ENABLED}" != "true" ]; then
    DEVICE="--device cpu"
fi

exec python tools/run_cli.py ${DEVICE} --reference-audio-file reference/reference_audio_01.wav --reference-text-file reference/reference_text_01.txt  --text-file working/sample_01.txt --audio-file /app/data/sample_01.wav
