---
title: "MetaAudio: A Few-Shot Audio Classification Benchmark"
collection: publications
permalink: /publication/2009-10-01-paper-title-number-1
excerpt: 'This paper establishes a novel few-shot audio classification benchmark'
date: 2022-04-03
venue: 'ICANN22'
paperurl: 'https://arxiv.org/abs/2204.02121'
citation: 'Heggan, C., Budgett, S., Hospedales, T., Yaghoobi, M. (2022). MetaAudio: A Few-Shot Audio Classification Benchmark. In: Pimenidis, E., Angelov, P., Jayne, C., Papaleonidas, A., Aydin, M. (eds) Artificial Neural Networks and Machine Learning – ICANN 2022. ICANN 2022. Lecture Notes in Computer Science, vol 13529. Springer, Cham. https://doi.org/10.1007/978-3-031-15919-0_19'
---
**Abstract:** Currently available benchmarks for few-shot learning (machine learning with few training examples) are limited in the domains they cover, primarily focusing on image classification. This work aims to alleviate this reliance on image-based benchmarks by offering the first comprehensive, public and fully reproducible audio based alternative, covering a variety of sound domains and experimental settings. We compare the few-shot classification performance of a variety of techniques on seven audio datasets (spanning environmental sounds to human-speech). Extending this, we carry out in-depth analyses of joint training (where all datasets are used during training) and cross-dataset adaptation protocols, establishing the possibility of a generalised audio few-shot classification algorithm. Our experimentation shows gradient-based meta-learning methods such as MAML and Meta-Curvature consistently outperform both metric and baseline methods. We also demonstrate that the joint training routine helps overall generalisation for the environmental sound databases included, as well as being a somewhat-effective method of tackling the cross-dataset/domain setting.

[Paper on arXiv](https://arxiv.org/abs/2204.02121) <br/>
[Paper on Springer](https://link.springer.com/chapter/10.1007/978-3-031-15919-0_19#Ack1) <br/>
[Code](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark) <br/>
[Papers With Code Entry](https://paperswithcode.com/paper/metaaudio-a-few-shot-audio-classification) <br/>

**Recommended citation (BibTex):**" @InProceedings{10.1007/978-3-031-15919-0_19,
author="Heggan, Calum
and Budgett, Sam
and Hospedales, Timothy
and Yaghoobi, Mehrdad",
editor="Pimenidis, Elias
and Angelov, Plamen
and Jayne, Chrisina
and Papaleonidas, Antonios
and Aydin, Mehmet",
title="MetaAudio: A Few-Shot Audio Classification Benchmark",
booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2022",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="219--230",
isbn="978-3-031-15919-0"
}"