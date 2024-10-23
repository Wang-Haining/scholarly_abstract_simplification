# Simplifying Scholarly Abstracts for Accessible Digital Libraries

This repository accompanies our manuscript *Simplifying Scholarly Abstracts for Accessible Digital Libraries* submitted to JCDL2024.

## Demo
Play with our models reported in the manuscript on [Colab](https://colab.research.google.com/drive/1tNC2GicYKW9ffMGeEV3539Qy_mobOSYS?usp=sharing).

## Results
The generations from different models are hosted in the folder `eval_results_temp_0.01` with informative names. 
Using a temperature of 0.01 across generations is only to reduce possible technical problems in decoding and is virtually equivalent to a temperature of 0, as described in the manuscript.

## Corpus
Due to copyright restrictions, we cannot share the Scientific Abstract-Significance Statement (SASS) corpus publicly. 
Please feel free to contact us for access to the corpus for academic use.
To examine the corpus statistics, you can run the script `corpus_stats.py` after unzipping the corpus file to the folder `resources`.

## Models
Below are links to the fine-tuned models hosted on Hugging Face hubs:
- [OLMo-1B-SFT-SASS](https://huggingface.co/AI4Library/olmo-1b-sft-sass)
- [Gemma-2B-SFT-SASS](https://huggingface.co/AI4Library/gemma-2b-sft-sass) (Note, Gemma-2B requires permission from Google to use. See License.)
- [Phi-2-SFT-SASS](https://huggingface.co/AI4Library/phi-2-sft-sass) (Note, not suggested for use in practice due to its performance)

See our [Demo](https://colab.research.google.com/drive/1tNC2GicYKW9ffMGeEV3539Qy_mobOSYS?usp=sharing) for use.

## Reproduction

Be sure to reproduce our environment first with:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Training 
You can use the script `sft.py` to train your own models. 
Here is an example we used for training OLMo-1B using the SASS corpus on a single Nvidia A40 GPU:
```bash
python -m sft --model olmo-1b --per_device_train_batch_size 4
```
See more examples in the `runs` folder with script names prefixed with `sft_`.

### Evaluation
Download our checkpoints from [Zenodo](todo) and unzip them into the project folder. A typical checkpoint folder has a name like `ckpts/sft_OLMo-1B-hf/checkpoint-940`.
Then you can use the script `eval_outputs.py` to rerun the generation and evaluation. Here is an example we used for evaluating Gemma-2B's performance:
```bash
python -m eval_outputs --model gemma-2b --temperature 0.01
```
See more examples in the `runs` folder with script names prefixed with `eval_`.

### Word Accessibility Estimator
We reproduced the model used by [Riddell & Igarashi (2021)](https://aclanthology.org/2021.ranlp-1.133/) using the English Wikipedia corpus and provided the trained word accessibility estimator in pickle format at `word_freq/wa_model.pkl`.
The estimator for an arbitrary English word will be loaded when running `eval_outputs.py`.

If you are interested in reproducing our word accessibility estimator, consider the following scripts in the created environment:
```bash
python -m calculate_token_frequency  # calculate ground truth from wiki_en
python -m estimate_token_frequency  # fit a ridge regression
```
### Zero-shot Performance of OpenAI's Models
The outputs from GPT-3.5/GPT-4o and the logs are hosted in the folder `eval_results_temp_0.01`.
You need to supply your own OPENAI_API_KEY before running the script `eval_openai_models.py` to reproduce the results.

## License
Our scripts are under the 0BSD license. OLMo-1B is licensed under Apache-2.0, and Phi-2 is under MIT.
Gemma-2B has its own [license](https://ai.google.dev/gemma/terms) and requires permission from Google to use our fine-tuned Gemma-2B.

## Contact
[Haining Wang](mailto:hw56@indiana.edu)

## Citation
```latex
@article{wang2024simplifying,
  title={Simplifying Scholarly Abstracts for Accessible Digital Libraries},
  author={Wang, Haining and Clark, Jason},
  journal={arXiv preprint arXiv:2408.03899},
  year={2024}
}
```
