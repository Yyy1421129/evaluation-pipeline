# Evaluation System for Speech Tasks

## Overview

This evaluation system is designed to assess multiple speech-related tasks, including:

- ASR (Automatic Speech Recognition)
- SER (Speech Emotion Recognition)
- GR (Gender Recognition)
- S2TT (Speech-to-Text Translation)
- SLU (Spoken Language Understanding)
- SD (Speaker Diarization)
- SA-ASR (Speaker-Attributed ASR)

## Input Format

### Ground Truth (GT) (ASR, SER, GR, S2TT, SLU)
Input is a JSON file, each entry like:
```json
{
  "key": "6930-76324-0026",
  "task": "ASR",
  "target": "ISNT HE THE GREATEST FOR GETTING INTO ODD CORNERS",
  "path": "/aistor/sjtu/hpc_stor01/home/yangyi/data/asr/test/data/data_wav.7.ark:166773289"
}
```

### Model Prediction
A text file, each line:
```
<id>    <predicted_text>
```
Example:
```
10226_10111_000000    after his nap timothy lazily stretched first one gray velvet foot then another strolled indolently to his plate turning over the food carefully selecting choice bits nosing out that which he scorned upon the clean hearth
```
For SD tasks, input format should be .rttm. For SA-ASR tasks, input format should be .stm

## Task Requirements

- For each task, compare model predictions with GT according to task-specific metrics.
- Support batch evaluation and summary statistics.
- Modular design for easy extension to new tasks.

## Usage

1. Place GT JSON and prediction files in the directory.
2. Run the evaluation scripts for the desired task.
3. Results will be output as summary tables and metrics.

## Extensibility

- Add new task modules by following the input/output conventions.
- Metrics and evaluation logic should be encapsulated per task.

---

