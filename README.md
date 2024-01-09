# MusIm

## Introduction

## Usage

```python
from transformers import AutoProcessor
from musim.hf import MusImPipeline

ast_proc = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", low_cpu_mem_usage=True)

m2i = MusImPipeline.from_pretrained("juliagsy/musim")
```

## Music-conditioned Image Generation

### Generation

```python
from PIL import Image
from IPython.core.display import display

input_wav = "<your-music>"

wav = ast_proc(input_wav.tolist(), sampling_rate=16000, return_tensors="pt")
wav = wav.to("cuda")

gen_image = m2i(wav)
gen_image_d = Image.fromarray(gen_image)
display(gen_image_d)
```


### Example 1

[Input wav](examples/wav_5.wav)

Generated image:

<img width="40%" src="examples/img_2.png">


### Example 2

[Input wav](examples/wav_5.wav)

Generated image:

<img width="40%" src="examples/img_5.png">