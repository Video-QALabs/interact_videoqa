# Updates

- [7 March 2025] Dataset description released
- [3 March 2025] Training Code and Sample Dataset Released
- [28 February 2025] Website is Live.
- [16 February 2025] Paper submitted for a conference [Under Review].

# *InterAct VideoQA*: A Benchmark Dataset for Video Question Answering in Traffic Intersection Monitoring
<div>
<a href="#"> Paper</a> |
<a href="https://interact-videoqa.github.io/InterActVideoQA/">Website</a> |
<a href="https://huggingface.co/datasets/joeWabbit/InterAct_Video_Reasoning_Rich_Video_QA_for_Urban_Traffic/blob/main/README.md">Data</a> |
<a href="https://interact-videoqa.github.io/InterActVideoQA/docs/InterAct_VideoQADatasetDescription.pdf" target="_blank">Doc </a>
</div>
<hr>
<div style="text-align: center;">
<img src="https://github.com/user-attachments/assets/9b0e8d90-ff04-44f5-a2fc-9fc03e7aaf3a"/>
</div>
<p align="justify">Traffic monitoring is crucial for urban mobility, road safety, and intelligent transportation systems (ITS). Deep learning has advanced video-based traffic monitoring through video question answering (VideoQA) models, enabling structured insight extraction from traffic videos. However, existing VideoQA models struggle with the complexity of real-world traffic scenes, where multiple concurrent events unfold across spatiotemporal dimensions. To address these challenges, this paper introduces InterAct VideoQA, a curated dataset designed to benchmark and enhance VideoQA models for traffic monitoring tasks. The InterAct VideoQA dataset comprises 8 hours of real-world traffic footage collected from diverse intersections, segmented into 10-second video clips, with over 25,000 question-answer (QA) pairs covering spatiotemporal dynamics, vehicle interactions, incident detection, and other critical traffic attributes. State-of-the-art VideoQA models are evaluated on InterAct VideoQA, exposing challenges in reasoning over fine-grained spatiotemporal dependencies within complex traffic scenarios. Fine-tuning these models on InterAct VideoQA also yields notable performance improvements, demonstrating the necessity of domain-specific datasets for VideoQA. InterAct VideoQA is publicly available as a benchmark dataset to facilitate future research in real-world-deployable VideoQA models for intelligent transportation systems.
</p>
# Related Works



<div style="text-align: center;">
<img src="https://github.com/user-attachments/assets/82c93cc6-4f7d-4e35-b38f-5079b1b12ef3"/>
</div>


# Dataset Download
Dataset can be downloaded <a href="https://drive.google.com/drive/folders/1dwbeWHASKkLbLOImyHKE8of8hWCq7bdO?usp=drive_link">here</a>


# Dataset Overview
<p align="justify">
InterAct VideoQA dataset comprises 28,800 question-answer pairs across various reasoning categories. A higher concentration appears in counting, attribute recognition, and event reasoning, followed by counterfactual inference and reverse reasoning.The dataset also illustrate the dataset's emphasis on vehicular-related questions, the dominance of attribution and event reasoning categories, and the distribution of question types (“what,” “where,” and “how”). This structured approach supports the analysis of complex, multi-event traffic scenarios requiring robust spatio-temporal reasoning. A rigorous human and GPT-assisted validation process ensures all annotations' consistency, accuracy, and reliability.
</p>

# Folder Structure
```
data
├── videoannotations.csv
└── Videos
    ├── clip_videos_0.mp4
    ├── clip_videos_1.mp4
    ├── clip_videos_2.mp4
    └── ...

```
# Model Setup 
Please look at the official github page for the models to set up.
- [VideoLLama2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
- [LlavaNext-Video](https://github.com/LLaVA-VL/LLaVA-NeXT)  
- [Qwen2-VL-7B-hf](https://github.com/QwenLM/Qwen2.5-VL)
# Baseline

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/264443ff-05c6-49d2-9d5c-60a0789b6b2d" alt="Image">
</div>

The main implementations for these models for InterAct VideoQA can be found here.

[VideoLLama2](https://github.com/joe-rabbit/interact_videoqa/tree/main/interAct%20VideoQA/VideoLlama2) |
[LlavaNext-Video](https://github.com/joe-rabbit/interact_videoqa/tree/main/interAct%20VideoQA/Llava-Next-Video) | 
[Qwen2-VL-7B-hf](https://github.com/joe-rabbit/interact_videoqa/tree/main/interAct%20VideoQA/Qwen-VL2-7B-hf)

# License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

# Citation

```

  ```
