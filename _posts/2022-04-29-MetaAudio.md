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

## Few-Shot Audio Classification
At test-time, few-shot classification performance is evaluated using N-way k-shot tasks, where N is the novel classes (never seen by the model before) and k is the number of examples per each of those classes given for use. For instance, each of the tasks T<sub>1</sub> -> T<sub>3</sub> represented in Figure 1 are 5-way 1-shot tasks. Each task is composed of two parts, the support which is trained on in some way (from explicit gradient descent to prototypes in some embedded space) at test time and query, the evaluation component. Parameterising these tasks with N and k also allows for more in-depth analysis of sample complexity.  

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/MetaAudio_blog_post/Few_shot_audio_classification.svg" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Figure 1: Example tasks for few-shot audio classification with spectrograms. Tasks contain both a support which we train on and a query which we evaluate on</span>
</span>

As well as testing like this, most of the algorithms considered also train with these few-shot tasks (called episodic training). The only exception to this is SimpleShot which trains with more traditional batching over the training classes, after which its linear head is stripped and the remainder of the model used as a feature extractor.  

## Data, Algorithms & Setup
### Datasets 
Over the full benchmark we experiment with 7 unique datasets, 5 of which we split up class-wise for both training and evaluation, and 2 of which we set aside for cross-dataset testing. Of these 7, 3 are fixed length (all samples same size e.g. 5 seconds) and 4 are variable length (sample length varies over the full dataset). 

| **Name**                     | **Setting**     | **$N^o$ Classes** | **$N^o$ Samples** | **Format** | **Sample Length** | **Use**         |
|:----------------------------:|:---------------:|:-----------------:|:-----------------:|:----------:|:-----------------:|:---------------:|
| [ESC-50](https://www.karolpiczak.com/papers/Piczak2015-ESC-Dataset.pdf)                      | Environmental   | 50                | 2,000             | Fixed      | 5s                | Meta-train/test |
| [NSynth](https://arxiv.org/abs/1704.01279)                       | Instrumentation | 1006              | 305,978           | Fixed      | 4s                | Meta-train/test |
| [FSDKaggle18](https://arxiv.org/abs/1807.09902)                  | Mixed           | 41                | 11,073            | Variable   | 0.3s - 30s        | Meta-train/test |
| [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)                    | Voice           | 1251              | 153,516           | Variable   | 3s - 180s         | Meta-train/test |
| [BirdClef2020](https://www.imageclef.org/BirdCLEF2020)                | Bird Song       | 960               | 72,305            | Variable   | 3s - 30m          | Meta-train/test |
| BirdCLEF 2020 (Pruned)       | Bird Song       | 715               | 63,364            | Variable   | 3s - 180s         | Meta-train/test |
| [Watkins Marine Mammal Sound Database](https://cis.whoi.edu/science/B/whalesounds/index.cfm) | Marine Mammals  | 32                | 1698              | Variable   | 0.1 - 150s        | Meta-test       |
| [SpeechCommandsV2](https://arxiv.org/abs/1804.03209)             | Spoken Word     | 35                | 105,829           | Fixed      | 1s                | Meta-test       |

The extra dataset included is a pruned version of the BirdClef set. As its contained samples span from 30 seconds to 30 minutes, hardware requirements were significantly higher. This was to an extent that we considered it a barrier of entry when trying to train and utilise networks on the set. To overcome this we pose a pruned version where samples longer than 180 seconds are removed along with classes that contain fewer than 50 examples. 

Each dataset used for both meta-train and meta-test are class-wise split into training, validation and test sets with a ratio of 7/1/2. 

### Processing & Input
Pre-processing was kept simple with only a few major steps. The first of these was z-normalisation on each raw audio sample individually. We then converted all raw time-series into log-mel spectrograms using identical parameters across both samples and datasets. 

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/MetaAudio_blog_post/spectrogram_transform.svg" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Figure 2: Pre-processing pipeline for data samples</span>
</span>

For the final normalisation on the sampled spectrograms we performed some prior experimentation, looking specifically at at per-sample, channel-wise and global. In general both channel and global performed similarly, with each taking the edge in some cases. For simplicity we opted to use global normalisation across all samples and experiments in the work. 

##### Variable Length Samples 
For variable length samples we first split the raw audio clip into *L* second sub-clips before later converting each to log-mel spectrograms. All of these sub-clips are then stacked and stored as one single file, which can be later sampled. If any sub-clips is less than *L* we repeat the clip and clip to the required length. The exact value of *L* is left as a hyperparameter which we investigate the effects of, however For the majority of the experiments in this version of MetaAudio it is set to 5 seconds, primarily to aid joint training and cross-dataset experimentation. All of this processing is done entirely offline, similarly to teh fixed length setting. 

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/MetaAudio_blog_post/variable_spec_raw.svg" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Figure 3: Variable length clips are first split into sub-clips of *L* seconds. From here the sub-clips are individually converted to spectrograms. If any clips fall short they are repeated until the required length in the spectrogram space. </span>
</span>

There are a few reasons we choose this specific variable length pipeline. Firstly, splitting the clip up and then converting to spectrograms prevents data leakage between sub-clips compared to the alternative, where the full spectrogram is created and then split up. Additionally the stacking of sub-clips in one file allows us to use the same file whether the sample is selected as either a support or a query. If a variable length sample is chosen as a support vector, one of the sub-clips is randomly chosen for use. If selected as a query all sub-clips are predicted over, with a majority vote system deciding on the final assigned class. 

### Algorithms
Due to the large amount of algorithmic literature for meta-learning, MetaAudio is not exhaustive in the algorithms tested. To overcome this as best as possible, we chose a representative few, spanning baselines, metric learners and gradient-based approaches. The exact list considered so far are as follows:

  -  [FO-MAML](https://arxiv.org/abs/1703.03400)
  -  [FO-Meta-Curvature](https://arxiv.org/abs/1902.03356)
  -  [Prototypical Networks](https://arxiv.org/abs/1703.05175)
  -  [SimpleShot](https://arxiv.org/abs/1911.04623)
  -  [Meta-Baseline](https://arxiv.org/abs/2003.04390)

 First order methods are used for the gradient-based learners as during initial experimentation, using 2nd order gradients yielded either similar or lower performance.

## Experiments
### Within Dataset Evaluation

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/MetaAudio_blog_post/within_dataset.svg" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Figure 4:  </span>
</span>
figure for here 

### Joint Training
#### Within Dataset Sampling
figures for here (2)
<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/MetaAudio_blog_post/joint_train_not_mixed.svg" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Figure 4:  </span>
</span>

#### Free Dataset Sampling

### Joint Training to Cross-Dataset


### Massive Pre-Train


### Other Notes
Data augmentation was not used 





## Reproduction & Use

