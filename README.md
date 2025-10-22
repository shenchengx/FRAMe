# FRAMe

## Introduction
This project provides a comprehensive fracture risk prediction model evaluation and comparison system, supporting performance comparison between large language models and traditional machine learning models.

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
