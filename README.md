# FRAMe

## Abstract
Diabetes compromises bone health through reduced bone mineral density, altered microarchitecture, and disrupted metabolism markers, substantially increasing fracture risk. However, the large number of indicators associated with diabetic pathogenesis, combined with their complex interrelationships, obscures the underlying pathogenic mechanisms, thereby hindering clinical efforts in fracture prediction and the development of targeted therapeutic strategies for diabetic patients. Large Language Models (LLMs), characterized by their vast number of parameters and sophisticated architectures, are capable of reasoning over intricate interdependencies among variables, making them particularly well-suited for the aforementioned scenario. Based on this, this study proposes a Fracture Risk Prediction and Medication Guidance Framework (FRAMe) for diabetic patients. Leveraging state-of-the-art language modeling techniques, FRAMe enables both the prediction of fracture risks in diabetes-related anatomical regions and the provision of personalized pharmacological recommendations. Specifically, (1) FRAMe employs a dual-backbone architecture combining fine-tuned large language models to capture complex interactions between diabetic indicators and bone health. (2) An optimized prompt-engineering strategy integrates multidimensional medical indicators, enhancing prediction accuracy and clinical interpretability. (3) A medical knowledge assessment module encodes clinical diagnostic criteria and risk assessment guidelines into model parameters, ensuring evidence-based probability predictions. Constructed and validated using real clinical dataset from the First Affiliated Hospital of Soochow University, FRAMe demonstrates superior performance compared to traditional machine learning approaches in site-specific fracture risk prediction and medication guidance, providing enhanced interpretability for intelligent diagnosis and treatment of diabetes complications.

## Introduction
This repository provides a comprehensive fracture risk prediction model evaluation and comparison system, supporting performance comparison between large language models and traditional machine learning models.

## Environment Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- For details, see requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Configure Paths
Before running the scripts, please modify the following configurations:

```python
self.model_configs = {
    "Model_1": {
        "path": "/your/model/path",  # 修改为您的模型路径
        "name": "模型名称"
    }
}

self.data_paths = {
    "train_data": "/your/data/train.json",  # 修改为您的数据路径
    "test_data": "/your/data/test.json"
}
```

### 2. Run Evaluation
```bash
# With Prompt version
python scripts/eval-all.py

# Without Prompt version
python scripts/eval-NoPrompt.py
```

## Dataset Description
The dataset has been preprocessed and contains 4 JSON files. For details, see datasets.zip in the release.

## Medical Guidance Evaluation
The medical guidance section is directly evaluated through the llama-factory framework. (GitHub link: https://github.com/hiyouga/LLaMA-Factory)

## 许可证
MIT License
