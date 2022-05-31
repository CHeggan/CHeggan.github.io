---
title: 'MetaAudio: A Benchmark Breakdown'
date: 2022-05-31
permalink: /posts/2022/04/MetaAudio_blog/
tags:
  - few-shot learning
  - meta-learning
  - benchmark
  - acoustics 
---

'MetaAudio: A Few-Shot Classification Benchmark' was released in early April. It contains a variety of benchmark results for researchers to beat in the future. This blog aims to be a more easily digestible breakdown of the work. All of the code for MetaAudio can be found [here](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark)

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/MetaAudio_blog_post/MetaAudio Logo.svg" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;"></span>
</span>

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

<p align="center">
    Table 1: High level details of all datasets considered in MetaAudio
</p>

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

Each dataset used for both meta-train and meta-test are randomly class-wise split into training, validation and test sets with a ratio of 7/1/2. This random class-wise splitting with no fold validation is ot ideal but due to the expense in meta and few-shot learning, it is commonplace within the community. 

### Processing & Input
Pre-processing was kept simple with only a few major steps. The first of these was z-normalisation on each raw audio sample individually. We then converted all raw time-series into log-mel spectrograms using identical parameters across both samples and datasets. 

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/MetaAudio_blog_post/spectrogram_transform.svg" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Figure 2: Pre-processing pipeline for data samples</span>
</span>

For the final normalisation on the sampled spectrograms we performed some prior experimentation, looking specifically at at per-sample, channel-wise and global. In general both channel and global performed similarly, with each taking the edge in some cases. For simplicity we opted to use global normalisation across all samples and experiments in the work. 

##### Variable Length Samples 
For variable length samples we first split the raw audio clip into *L* second sub-clips before later converting each to log-mel spectrograms. All of these sub-clips are then stacked and stored as one single file, which can be later sampled. If any sub-clips is less than *L* we repeat the clip and clip to the required length. The exact value of *L* is left as a hyperparameter which we investigate the effects of, however For the majority of the experiments in this version of MetaAudio it is set to 5 seconds, primarily to aid joint training and cross-dataset experimentation. All of this processing is done entirely offline, similarly to the fixed length setting. 

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
Throughout all of the experiments, the training, validation and testing subsets defined by random class-wise splitting are kept constant. This is not only an important step for this baseline work but also for future researchers tackling the problems set out here.
### General
#### Evaluation Method & Optimiser
Keeping with current and past few-shot works, we evaluate a single end-to-end trained model, obtaining average classification accuracies and a 95% confidence interval over 10,000 randomly sampled tasks from the meta-test set. Although not as informative as something like k-fold validation in more traditional problems, this has become the main metric simply due to the computational and time expense of meta-learning training. For training, we used the Adam optimiser with a non-adaptive learning rate. 

#### Model
Motivated by the increasing performance gaps between basic CNNs and sequentially informed models in traditional acoustic classification as well as some local verification of potential performance gains, we opted to use a lightweight CRNN model as out backbone architecture. Specifically, the CRNN contains a 4-block convolutional backbone (1-64-64-64) with an attached 1-layer non-bidirectional RNN containing 64 hidden units. The number of outputs in the final linear layer is either of size N-way or, in the case of metric learning and baseline
methods, 64. 

### Within Dataset Evaluation
The first ste of experiments carried out looked at within dataset meta-training, validation and testing. How this class-wise split is formatted is demonstrated in Figure 4. This type of pipeline is the most specific and computationally heavy, as each dataset has a model trained for it and it alone. Generally this goes against goals of generalised representation learners, however provides us with a strong baseline of performance for each of the datasets and algorithms. 

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/MetaAudio_blog_post/within_dataset.svg" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Figure 4: An example of meta-train, validation and testing, using a bird song dataset like BirdClef2020. Icons are used to more visually demonstrate the possible classes and how they might relate to one another. </span>
</span>

In general we found that gradient-based (GBML) learners like [MAML](https://arxiv.org/abs/1703.03400) and [Meta-Curvature](https://arxiv.org/abs/1902.03356) outperformed both the baseline models and metric learners. 

<p align="center">
    Table 2: Baseline Within Dataset Results
</p>

| **Dataset**                  | **FO-MAML**     | **FO-Meta-Curvature** | **ProtoNets** | **SimpleShot CL2N** | **Meta_baseline** |
|:----------------------------:|:---------------:|:---------------------:|:-------------:|:-------------------:|:-----------------:|
| ESC-50                       | 74.66 ± 0.42    | **76.17 ± 0.41**      | 68.83 ± 0.38  | 68.82 ± 0.39        |  71.72 ± 0.38     |
| NSynth                       | 93.85 ± 0.24    |**96.47 ± 0.19**       | 95.23 ± 0.19  |90.04 ± 0.27         | 90.74 ± 0.25      |
| FSDKaggle18                  | **43.45 ± 0.46**| 43.18 ± 0.45          | 39.44 ± 0.44  |42.03 ± 0.42         |40.27 ± 0.44       |
| VoxCeleb1                    | 60.89 ± 0.45    | **63.85 ± 0.44**      |59.64 ± 0.44   |48.50 ± 0.42         |55.54 ± 0.42       |
| BirdCLEF 2020 (Pruned)       |  56.26 ± 0.45   |**61.34 ± 0.46**       | 56.11 ± 0.46  | 57.66 ± 0.43        | 57.28 ± 0.41      |
|Avg Algorithm Rank            |2.4              |**1.2**                    |3.8            |4.0                  |3.6                |


This is immediately in contrast with the performance comparisons shown in the [SimpleShot](https://arxiv.org/abs/1911.04623) work with images, where the simple baseline was able to beat out a variety of GBML approaches. Specifically we observe Meta-Curavture performing strongest on 4/5 datasets, with MAML taking the final 1/5. We propose that this is due to the GBML methods’ adaption mechanism, updating feature representation at each meta-test episode, making them particularly useful for tasks with high inter-class/episode variance. Meanwhile the others must rely on a fixed feature extractor that cannot adapt to each unique episode

### Joint Training
The general idea for our joint training experiments is to train concurrently on all of our available datasets, hopefully leading to some implicit data-driven regularisation of the network. After training a network this way, we apply it on the individual test splits of the datasets used. This can be seen in Figure 5 and 6, where although training is mixed in some way, testing still occurs in each datasets meta-test split. We also apply these models to two held-out sets, all of which we use as meta-test.  

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/MetaAudio_blog_post/joint_train_not_mixed.svg" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Figure 5: Joint training scenario with within dataset sampling. Can see that individual meta-train tasks are still confined to containing classes from one dataset. Meta-test for each set is the same meta-test found in within dataset generalisation. </span>
</span>

We indentified two distinct ways to train with all datasets simultaneously, one where any individual task can only contain supports and queries from one of the included datasets (which we call **within dataset sampling**), and one in which samples contained within a task are unconstrained (**free dataset sampling**). 

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/MetaAudio_blog_post/Joint Train - Free Sampling.svg" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Figure 6: Joint training scenario with free dataset sampling. Meta-train tasks can now contain classes from more than one dataset and are unconfined. Meta-test for each set is the same meta-test found in within dataset generalisation and joint training with within dataset sampling. </span>
</span>

Comparing the performance of both sampling techniques against within-dataset evaluation, we see a general degradation of performance, with only ESC-50 and Kaggle18 improving (both from free dataset sampling). The difference varies heavily in magnitude between both the datasets and sampling routine used alike. 

<p align="center">
    Table 3: Joint Training (Within Dataset Sampling)
</p>

| **Dataset**                     | **FO-MAML**     | **FO-Meta-Curvature** | **ProtoNets** | **SimpleShot CL2N** | **Meta_baseline** |
|:----------------------------:|:---------------:|:-----------------:|:-----------------:|:----------:|:-----------------:|
| ESC-50                      |  68.68 ± 0.45 |**72.43 ± 0.44** |61.49 ± 0.41| 59.31 ± 0.40 |62.79 ± 0.40
| NSynth                       | 81.54 ± 0.39 |82.22 ± 0.38 |78.63 ± 0.36 |**89.66 ± 0.41** |85.17 ± 0.31
| FSDKaggle18                 | 39.51 ± 0.44 |**41.22 ± 0.45**| 36.22 ± 0.40 |37.80 ± 0.40 |34.04 ± 0.40
| VoxCeleb1                   | **51.41 ± 0.43** |51.37 ± 0.44| 50.74 ± 0.41| 40.14 ± 0.41| 39.18 ±0.39
| BirdCLEF 2020 (Pruned)       |  **47.69 ± 0.45** |47.39 ± 0.46| 46.49 ± 0.43| 35.69 ± 0.40 |37.40 ± 0.40
| Watkins                      | 57.75 ± 0.47| **57.76 ± 0.47**| 49.16 ± 0.43| 52.73 ± 0.43| 52.09 ± 0.43
| SpeechCommands V1 |25.09 ± 0.40 |**26.33 ± 0.41**| 24.31 ± 0.36 |24.99 ± 0.35| 24.18 ± 0.36
|Avg Algorithm Rank |2.0| **1.6** |4.0 |3.4| 4.0

The generally mixed results here mirror other studies (full references in paper) and reflect the tradeoff between generally increasing the amount of training data available and the increased difficulty of learning a single model capable of simultaneous high performance on diverse data domains. This is some evidence that MetaAudio compliments existing works in providing a challenging benchmark to test future meta-learners' ability to fit diverse audio types, as well as enabling few-shot recognition of new categories. 

<p align="center">
    Table 4: Joint Training (Free Dataset Sampling)
</p>

| **Dataset**                     | **FO-MAML**     | **FO-Meta-Curvature** | **ProtoNets** | **SimpleShot CL2N** | **Meta_baseline** |
|:----------------------------:|:---------------:|:-----------------:|:-----------------:|:----------:|:-----------------:|
| ESC-50                      |  **76.24 ± 0.42** |75.72 ± 0.42 |68.63 ± 0.39 |59.04 ± 0.41| 61.53 ± 0.40|
| NSynth                       | 77.71 ± 0.41 |83.51 ± 0.37 |79.06 ± 0.36 |**90.02 ± 0.27** |85.04 ± 0.31|
| FSDKaggle18                 |  44.85 ± 0.45 |**45.46 ± 0.45**| 41.76 ± 0.41 |38.12 ± 0.40 |35.90 ± 0.38|
| VoxCeleb1                   | 39.52 ± 0.42 |39.83 ± 0.43| 40.74 ± 0.39| **42.66 ± 0.41**| 36.63 ± 0.38|
| BirdCLEF 2020 (Pruned)       | **46.76 ± 0.45**| 46.41 ± 0.46 |44.70 ± 0.42| 37.96 ± 0.40| 32.29 ± 0.38|
| Watkins                      | **60.27 ± 0.47** |58.19 ± 0.47| 48.56 ± 0.42 |54.34 ± 0.43| 53.23 ± 0.43|
| SpeechCommands V1 |**27.29 ± 0.42**| 26.56 ± 0.42| 24.30 ± 0.35 |24.74 ± 0.35 |23.88 ± 0.35|
|Avg Algorithm Rank |**2.1**| **2.1** |3.4 |3.0| 4.3|

Additionally, we contrast how the joint training episode sampling routines compare. For our main datasets, we observe 3/5 of the top results were obtained using the free sampling method, with the 2 outliers belonging to VoxCeleb and BirdClef - evidence that their tasks require significantly different and specific model parameterisation, as the within dataset task sampling would allow more opportunity to learn these more specialised features.


### Joint Training to Cross-Dataset
For the held-out cross-dataset tasks (Watkins, SpeechCommands V1), we also see the strongest performance coming from the free sampling routine, where it outperforms its within dataset counterpart by ∼2% in both held-out sets. As for the absolute performances obtained on the held-out sets, we see that our joint training transfers somewhat-effectively, with the model in one case attaining a respectable 50-60% and another obtaining accuracies only 5% above random.

### Massive Pre-Train
A full meta-learning pipeline for a specific dataset can be expensive. Transferring some pre-trained feature extractor and using a cheap linear classifier for each task could be cheaper due to the spreading of cost over multiple downstream tasks. In this direction we employed Audio Spectrogram Transformers from [this work](https://arxiv.org/abs/2104.01778), which were trained on the large ImageNet and AudioSet datasets. On top of the features that we obtained form these models, we applied both nearest centroid and linear SVM classification. 

<p align="center">
    Table 5: Massive Pre-training to linear classifier
</p>

|                 |  AST ImageNet                       ||  AST ImageNet & AudioSet                       || From Table 2 |
| Dataset         | SVM  | SimpleShot CL2N              | SVM  | SimpleShot CL2N  | SimpleShot (CL2N)|
|:----------------------------:|:---------------:|:-----------------:|:-----------------:|:----------:|:-----------------:|
| ESC-50                      |  61.12 ± 0.41 |60.41 ± 0.41| 61.61 ± 0.41| 64.48 ± 0.41| **68.82 ± 0.39** |
| NSynth                       | 64.26 ± 0.41| 66.68 ± 0.41 |62.62 ± 0.42| 63.78 ± 0.42| **90.04 ± 0.27**|
| FSDKaggle18                 | 34.01 ± 0.40 |33.52 ± 0.39| 38.38 ± 0.41| 38.76 ± 0.41| **42.03 ± 0.42** |
| VoxCeleb1                   |27.26 ± 0.36| 28.09 ± 0.37| 27.45 ± 0.36| 28.79 ± 0.38| **48.50 ± 0.42** |
| BirdCLEF 2020 (Pruned)       | 30.84 ± 0.37| 33.04 ± 0.41| 33.17 ± 0.38| 36.41 ± 0.42 |**57.66 ± 0.43**|
| Avg Rank |4.2| 3.8| 3.6| 2.4| **1.0**|
| Watkins                      |**55.91 ± 0.42** |55.40 ± 0.42| 51.46 ± 0.42| 51.81 ± 0.42 | N/A|
| SpeechCommands V1 | 26.24 ± 0.36 |26.46 ± 0.37| **30.69 ± 0.38**| 30.24 ± 0.38| N/A |
|Avg Rank |2.5| 2.5| 2.5| 2.5| N/A|

These results reveal a few interesting insights. Firstly and perhaps unsurprisingly, we observe that the features pre-trained on both AudioSet and ImageNet outperform those trained on ImageNet alone. The small margin between these however is perhaps surprising, showing that image-derived features provide most of the information needed to interpret spectrograms.  

Comparing these results to the in-domain training presented in Table 2, we observe substantial performance drops across the board, with the possible exceptions of ESC-50 and Kaggle18. In their best cases, NSynth, VoxCeleb and BirdClef all take drops in performance of ∼20% due to dataset shift between general purpose pre-training and our specific tasks, such as musical instruments, speech or bird song recognition. While the performance hit due to domain-shift is expected, these results are surprising as AudioSet is a much larger dataset, and the AST transformer is a much larger architecture than the CRNN used in Table 2. Within the image domain, comparable experimental settings show a clear win by simply applying larger pre-raining datasets and models along with simple readouts, compared to conducting within-domain meta-learning. This confirms the value of Meta-Audio as an important benchmark for assessing meta-learning contributions that cannot easily be replicated by larger architectures and more data. Performance on our held-out sets shows a more mixed set of results, with ImageNet only pre-training favouring Watkins, and ImageNet + AudioSet pre-training setting a new SOTA for SpeechCommands.

## Reproduction & Use
This work aims to be a benchmark upon which people can build and improve, in that sense we outline how to best use MetaAudio. There are two main ways this benchmark can be approached. One in which new ideas are actively being tested, and one in which reproduction and/or immediate extensions to MetaAudio is the goal. Both of these goals start with the [code repo](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark) available for MetaAudio. 

### Testing New Ideas
If simply testing new ideas against this benchmark, then following this format should work:
 - Obtain the datasets of interest (sources can be found [here](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark/tree/main/Dataset%20Processing)) - We recommend testing and reporting with all of the datasets to avoid claims of selection bias 
 - Go through the dataset preprocessing pipelines (.wav -> .npy raw -> .npy spec). All of the code for this as well as detailed descriptions of the scripts can be found [here](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark/tree/main/Dataset%20Processing. Note that some datasets like BirdClef & Watkins require an additional step of cleaning
 - Obtain the .npy class split files from [here](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark/tree/main/Dataset%20Splits). Unless doing specific experiments with sample length and meta-data, we strongly recommend using the so-called 'Baseline Splits'
 - If you already have some few-shot sampler for classification tasks that is set to sample from a folder-of-class-folders structure, then this should be all that is required from the MetaAudio repo. If this sampler is missing however, custom built classes can be found [here](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark/tree/main/Examples/MAML_ESC/dataset_)
- The evaluation metric used in MetaAudio is the average and 95% confidence interval taken over 10,000 randomly sampled tasks from the meta-test set. For fair and easy comparison, we recommend this to other researchers

### Reproduction & Immediate Extensions
Generally reproduction and immediate extensions will require the use of more of the code base than just benchmarking. Starting off, the steps outlined in the 'Testing New Ideas' should be followed. This wil end in having all of the datasets properly processed and setup for experimental work. On top of this, these steps may be helpful in starting off:
 - Environment replication. Within the main README file [here](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark), the enrichment file and instructions on how to load it into conda can be found
 - Examples of some experiments can be found [here](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark/tree/main/Examples). These include MAML for ESC-50 and ProtoNets for Kaggle18
 - The code included in the example experiments should be sufficient for both reproduction and code add-ons/extra experiments
 
## Conclusion
MetaAudio both frames and benchmarks a variety of interesting few-shot audio classification problems. These span datasets from a variety of sound domains, from environmental sounds to bird song, and settings from within-domain meta-learning to massive pre-training for features. Our presented results showed a variety of things, however the most important of these was that few-shot audio behaves significantly differently from the much more well-studied image domain. 

From here, our hope is that the community makes use of MetaAudio as a tool to more extensively round out the evaluation of novel meta and few-shot learning techniques alike. 

**Thank you very much for reading!**
