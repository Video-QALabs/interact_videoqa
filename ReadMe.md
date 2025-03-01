# Updates
- [16 February 2025] Paper submitted for conference.
- [28 February 2025] Website is Live.


# *InterAct VideoQA*: A Benchmark Dataset for Video Question Answering in Traffic Intersection Monitoring
<div>
<a href="#"> Paper</a> |
<a href="https://interact-videoqa.github.io/InterActVideoQA/">Website</a> |
<a href="#">Data</a> |
<a href="#" target="_blank">Doc </a>
</div>
<hr>
<img src="https://github.com/user-attachments/assets/1a99e4f5-f7a0-4016-8abe-9d6116d0f553"/>
Traffic monitoring is crucial for urban mobility, road safety, and intelligent transportation systems (ITS). Deep learning has advanced video-based traffic monitoring through video question answering (VideoQA) models, enabling structured insight extraction from traffic videos. However, existing VideoQA models struggle with the complexity of real-world traffic scenes, where multiple concurrent events unfold across spatiotemporal dimensions. To address these challenges, this paper introduces InterAct VideoQA, a curated dataset designed to benchmark and enhance VideoQA models for traffic monitoring tasks. The InterAct VideoQA dataset comprises 8 hours of real-world traffic footage collected from diverse intersections, segmented into 10-second video clips, with over 25,000 question-answer (QA) pairs covering spatiotemporal dynamics, vehicle interactions, incident detection, and other critical traffic attributes. State-of-the-art VideoQA models are evaluated on InterAct VideoQA, exposing challenges in reasoning over fine-grained spatiotemporal dependencies within complex traffic scenarios. Additionally, fine-tuning these models on InterAct VideoQA yields notable performance improvements, demonstrating the necessity of domain-specific datasets for VideoQA. InterAct VideoQA is publicly available as a benchmark dataset to facilitate future research in real-world-deployable VideoQA models for intelligent transportation systems.
<img src="https://github.com/user-attachments/assets/95651208-d9c4-4644-9740-a7156e7dd5b8"/>

# Download 
_Will be released soon_

# Dataset Overview

InterAct VideoQA dataset,comprises 28,800 question-answer pairs across various reasoning categories. A higher concentration appears in counting, attribute recognition, and event reasoning, followed by counterfactual inference and reverse reasoning (3a). Figures 3(b)-(d) illustrate the dataset's emphasis on vehicular-related questions, the dominance of attribution and event reasoning categories, and the distribution of question types (“what,” “where,” and “how”). This structured approach supports the analysis of complex, multi-event traffic scenarios requiring robust spatio-temporal reasoning. A rigorous human and GPT-assisted validation process ensures the consistency, accuracy, and reliability of all annotations.

# Folder Structure
_Will be released soon_

# Baseline

<img src="https://github.com/user-attachments/assets/79c84ec5-015e-4487-be42-2d70286152d8"> </img>

The main implementations for these models for InterAct VideoQA can be found here.

[VideoLLama2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) |
[LlavaNext-Video](https://github.com/LLaVA-VL/LLaVA-NeXT) | 
[Qwen2-VL-7B-hf](https://github.com/QwenLM/Qwen2.5-VL)

# License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

# Citation

```

  ```
