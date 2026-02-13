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
<key>    <predicted_text>
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

<details>

<summary>ASR part(Examples): Matrix: wenet wer/cer mer</summary>

python run_evaluation.py <gt_json> <pred_txt> —language zh(Optional)

```json
{
  "key": "en1",
  "task": "ASR",
  "target": "The price is $12.50."
  "path": /fake_path/audio1
}
```
Normalized result: The price is twelve dollars fifty cents
```json
{
  "key": "1",
  "task": "ASR",
  "target": "今天是2023-10-27"
  "path": /fake_path/audio5
}
```
Normalized result: 今天是二千零二十三年十月二十七日  
For codeswitch ASR tasks, language param should be ‘cs’.  
Returns: MER(mixture error rate) WER CER (Calculate separately in Chinese and English)

</details>

<details>

<summary>SER part(Examples): Matrix: acc</summary>

python run_evaluation.py <gt_json> <pred_txt> —ser_mapping {}(Optional)  
Default ser_mapping : {"neu": 0, "hap": 1, "ang": 2, "sad": 3} (IEMOCAP)  

```json
{
	"task": "ser", 
	"key": "utt_001", 
	"target": "happy", 
	"text": "I am so glad to see you!"
	"path": /fake_path/audio4
}
```

</details>

<details>

<summary>GR part(Examples): Matrix: acc</summary>

python run_evaluation.py <gt_json> <pred_txt> —gr_mapping {}(Optional)  
Default gr_mapping : {"man": 0, "woman": 1}

```json
{
	"task": "gr", 
	"key": "gr5", 
	"target": "man",
	"path": /fake_path/audio9
}
```

</details>

<details>

<summary>S2TT part(Examples): Matrix: sacrebleu BLEU CHRF</summary>

python run_evaluation.py <gt_json> <pred_txt> —language zh(Optional)  
Parameter language: the language of labels and prediction results

```json
{
	"key": "4", 
	"task": "ASR", 
	"target": "温度是-5"
	"path": /fake_path/audio9
}
```

</details>

<details>

<summary>SLU part(Examples): Matrix: acc</summary>

python run_evaluation.py <gt_json> <pred_txt>  
Parameter language: the language of labels and prediction results

```json
{
	"key": "pause_perception_a0937fe0-6ba7-4797-b698-8bad74a19b91", 
	"task": "slu", 
	"target": "Slow", 
	"prompt": "Which word is most likely followed by a pause in the audio? If there is no pause, select 'No pause'.\nA. crossing\nB. children\nC. Slow\nD. No pause\nPlease answer with the letter only.", 
	"path": "fakepath/audio1.wav"
}
```
Prediction(raw): 

```json
pause_perception_a0937fe0-6ba7-4797-b698-8bad74a19b91	The answer is C
pause_perception_b1234567-aaaa-bbbb-cccc-123456789abc	After careful listening, I choose D
pause_perception_c9876543-cccc-dddd-eeee-987654321fed	I think the answer is A
pause_perception_d1111111-aaaa-bbbb-cccc-123456789abc	B
pause_perception_e2222222-cccc-dddd-eeee-987654321fed	running
```
Prediction(processed):

```json
pause_perception_a0937fe0-6ba7-4797-b698-8bad74a19b91	Slow
pause_perception_b1234567-aaaa-bbbb-cccc-123456789abc	No pause
pause_perception_c9876543-cccc-dddd-eeee-987654321fed	children
pause_perception_d1111111-aaaa-bbbb-cccc-123456789abc	children
pause_perception_e2222222-cccc-dddd-eeee-987654321fed	running
```

</details>

<details>

<summary>SD part(Examples): Matrix: meeteval DER</summary>

python run_evaluation.py <gt_rttm> <pred_rttm> —task sd  
Pay attention to the different input format: .rttm

```json
SPEAKER session1 1 0.00 2.00 <NA> <NA> spk1 <NA> <NA>
SPEAKER session1 1 2.00 2.00 <NA> <NA> spk2 <NA> <NA>
SPEAKER session1 1 4.00 1.00 <NA> <NA> spk1 <NA> <NA>
SPEAKER session1 1 5.00 2.00 <NA> <NA> spk2 <NA> <NA>
```

</details>

<details>

<summary>SA-ASR part(Examples): Matrix: meeteval cpWER</summary>

python run_evaluation.py <gt_stm> <pred_stm> —task sa-asr  
Pay attention to the different input format: .stm

```json
session1 1 spk1 0.00 2.00 hello world
session1 1 spk2 2.00 4.00 how are you
session1 1 spk1 4.00 5.00 fine thank you
session1 1 spk2 5.00 7.00 see you later
```

</details>

## Extensibility

- Add new task modules by following the input/output conventions.
- Metrics and evaluation logic should be encapsulated per task.

### How to Extend (Add a New Task)

1. **Create your evaluation logic**  
   Add a new function or branch in `evaluation/evaluator.py` for your task, following the style of existing tasks.

2. **Register your task**  
   Add your task to the `TASK_MAP` in `run_evaluation.py`, for example:
   ```python
   TASK_MAP = {
       # ...
       "mytask": "mytask_eval",
   }
   ```

3. **Document your task**  
   Add a description of your task’s input/output format and usage to the README.

4. **(Optional) Add test data**  
   Place sample ground truth and prediction files for your task in the `tests/` directory.

#### Example: Add a new task "MyTask"

In `evaluation/evaluator.py`, add:
```python
elif task_name == "mytask_eval":
    ref_labels = []
    hyp_labels = []
    with open(data["ref_file"], 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                key, label = parts
                ref_labels.append(label.strip().lower())
    with open(data["hyp_file"], 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                key, label = parts
                hyp_labels.append(label.strip().lower())
    correct = sum(1 for r, h in zip(ref_labels, hyp_labels) if r == h)
    acc = correct / len(ref_labels) if ref_labels else 0
    print(f"[MYTASK] Accuracy: {acc:.4f} ({correct}/{len(ref_labels)})")
    return acc
```

In `run_evaluation.py`, add to `TASK_MAP`:
```python
"mytask": "mytask_eval",
```

By following these steps, anyone can easily contribute a new task module to this evaluation framework.

---
