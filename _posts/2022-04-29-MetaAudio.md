---
title: 'MetaAudio: A Benchmark Breakdown'
date: 2022-04-29
permalink: /posts/2022/04/MetaAudio_blog/
tags:
  - few-shot learning
  - meta-learning
  - benchmark
  - acoustics 
---

'MetaAudio: A Few-Shot Classification Benchmark' was released in early April. It contains a variety of benchmark results for researchers to beat in the future. This blog aims to be a more easily digestible breakdown of the work. All of the code for MetaAudio can be found [here](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark)

## Aims of the Work
As it currently stands, the majority of work focusing on few-shot learning exists within computer vision. MetaAudio aims to be an accompanying benchmark to those that already exist, hopefully reducing algorithmic bias toward the image domain. To provide this, we investigate a variety of experimental settings, drawing parallels where possible to existing benchmarks. 

## Data & Setup
### Datasets 
Over the full benchmark we experiment with 7 unique datasets, 5 of which we split up class-wise for both training and evaluation, and 2 of which we set aside for cross-dataset testing. Of these 7, 3 are fixed length (all samples same size e.g. 5 seconds) and 4 are variable length (sample length varies over the full dataset). 

| **Name**                     | **Setting**     | **$N^o$ Classes** | **$N^o$ Samples** | **Format** | **Sample Length** | **Use**         |
|:----------------------------:|:---------------:|:-----------------:|:-----------------:|:----------:|:-----------------:|:---------------:|
| ESC-50                       | Environmental   | 50                | 2,000             | Fixed      | 5s                | Meta-train/test |
| NSynth                       | Instrumentation | 1006              | 305,978           | Fixed      | 4s                | Meta-train/test |
| FDSKaggle18                  | Mixed           | 41                | 11,073            | Variable   | 0.3s - 30s        | Meta-train/test |
| VoxCeleb1                    | Voice           | 1251              | 153,516           | Variable   | 3s - 180s         | Meta-train/test |
| BirdCLEF 2020                | Bird Song       | 960               | 72,305            | Variable   | 3s - 30m          | Meta-train/test |
| BirdCLEF 2020 (Pruned)       | Bird Song       | 715               | 63,364            | Variable   | 3s - 180s         | Meta-train/test |
| Watkins Marine Mammal Sounds | Marine Mammals  | 32                | 1698              | Variable   | 0.1 - 150s        | Meta-test       |
| SpeechCommandsV2             | Spoken Word     | 35                | 105,829           | Fixed      | 1s                | Meta-test       |

The extra dataset included is a pruned version of the BirdClef set. As its contained samples span from 30 seconds to 30 minutes, hardware requirements were significantly higher. This was to an extent that we considered it a barrier of entry when trying to train and utilise networks on the set. To overcome this we pose a pruned version where samples longer than 180 seconds are removed along with classes that contain fewer than 50 examples. 


### Processing & Input
Pre-processing was kept simple with only a few major steps. The first of these was z-normalisation on each raw audio sample individually. We then converted all raw time-series into log-mel spectrograms using identical parameters across both samples and datasets. For the final normalisation on the sampled spectrograms we performed some prior experimentation, looking specifically at at per-sample, channel-wise and global. In general both channel and global performed similarly, with each taking the edge in some cases. For simplicity we opted to use global normalisation across all samples and experiments in the work. 

![Alt text](/images/MetaAudio_blog_post/spectrogram_transform.svg)
*image_caption*

#### Variable Length Sampling

figure for here 

## Experiments
### Within Dataset Evaluation

figure for here 

### Joint Training
#### Sampling Types
figures for here (2)

### Other Notes
Data augmentation was not used 





## Reproduction & Use


<>  images/MetaAudio_blog_post/