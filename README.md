# FRAMe

## 简介
本项目提供了一套完整的骨折风险预测模型评估对比系统，支持大语言模型和传统机器学习模型的性能对比。

## 环境要求
- Python 3.8+
- CUDA 11.0+ (用于GPU加速)
- 详见 requirements.txt

## 安装
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 配置路径
在运行脚本前，请修改以下配置：

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

### 2. 运行评估
```bash
# 带Prompt版本
python scripts/eval-all.py

# 无Prompt版本
python scripts/eval-NoPrompt.py
```

## 数据集说明
数据集已经过预处理，包含4个JSON文件。详见release中datasets.zip。


## 许可证
MIT License
