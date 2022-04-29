---
title: 'MetaAudio: A Benchmark Breakdown'
date: 2022-04-29
permalink: /posts/2022/04/MetaAudio_blog/
tags:
  - few-shot learning
  - meta-learning
  - acoustics 
---

'MetaAudio: A Few-Shot Classification Benchmark' was released in early April. It contains a variety of benchmark results for researchers to beat in the future. This blog aims to be a more easily digestible breakdown of the work. 

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



### Processing & Input
#### Variable Length Sampling


## Experiments
### Within Dataset Evaluation


### Joint Training
#### Sampling Types

####





## Reproduction & Use


<>  images/MetaAudio_blog_post/