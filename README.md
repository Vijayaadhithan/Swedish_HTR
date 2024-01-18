# Swedish_HTR

## This repo is for the course Project in Data Science.

## Overview

This project focuses on transcribing Swedish handwritten manuscripts using a combination of the Laypa framework for layout analysis, Loghi HTR for text recognition through transfer learning, and auxiliary Minion gadgets for various preprocessing and post-processing tasks.

## Workflow

## Text Detection and Baseline Identification:
* Laypa framework detects text areas and baselines in Swedish handwritten manuscripts.
## Text Segmentation and Transcription:
Text lines are segmented into smaller pieces.
* Loghi HTR transcribes these pieces into Unicode text using a trained model based on transfer learning.
## PageXML Handling:
* Transcribing results are saved sequentially to a PageXML file.
* The final result rearranges the transcribing order based on logical sequence information from Laypa detection.

## Results

## Dataset
Our dataset, the Labours Memory dataset, comprises digitized versions of annual and financial reports of blue-collar worker unions from the 1880s. To maintain a consistent writing style, we only include image data from the 1880s to the 1960s. The dataset consists of 1725 images, with 90% serving as the training set and 10% as the validation set. The dataset split is random.

## Experimental Setup
## Platform and Hardware

The model is trained and validated on the Alvis platform, utilizing an Nvidia A100 40G GPU. The key components of the workflow run within Apptainer containers, covering training, inference, and peripheral operations. TensorFlow 2 is the chosen deep learning framework.

## Training Methodology

* The training process consists of two stages.
* In the first stage, we conduct 15 epochs with a batch size of 32, a learning rate of 3e-4, and no early stopping mechanism. The character list is provided by the pre-trained model.
* The second stage involves a reduced learning rate of 1e-4, introduces early stopping, and maintains other setups. The character list in this stage is altered based on a new character list generated in the first stage.

## Robustness and Consistency

All reported results are averaged over five tests to ensure the robustness of the model. Identical setups are maintained for all tests, maximizing integrity and comparability.

## Evaluation Results
The evaluation results demonstrate the effectiveness of our proposed method.

## Quantitative Metrics

Character Error Rate (CER): 2.56%
Word Error Rate (WER): 44.59%


#### This project is based on Loghi HTR workflow and transfer learning.
## Usage
1. Pull this repo
2. Download all models provided in the link below (best_nov_30 and laypa model). best_nov_30 is HTR model for transcribing, it is a folder while laypa model is used for baseline detection, it contains two files (one pth file and one yaml conf file).
[best_nov_30](https://1drv.ms/f/s!AhLc1l9ln_UugvBxsrxiS2OCpx5f3w?e=NjdVju)
[laypa model](https://1drv.ms/f/s!AhLc1l9ln_UugvEF1UOXzfOyDoWtwA?e=CjMQWx)
3. Downlaod 3 Apptainer images (laypa.sif, htr.sif, tool.sif).
[images](https://1drv.ms/f/s!AhLc1l9ln_UugvEGg39WPvRkaehvdg?e=lnt6hl)
4. Edit certain directories in shell scripts like apptainer-pipe.sh into model files in your system.
5. To try the model, use apptainer-pipe.sh and put images you want to process into the *image* folder
6. Run apptainer-pipe.sh, and after running it, you should expect to see a *page* folder inside *image* folder, where you can find pairs of baseline detection results and transcribing results.
