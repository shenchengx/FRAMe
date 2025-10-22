#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
骨折风险预测模型评估对比系统
对比三个已训练好的本地大模型和传统机器学习模型在骨折风险预测数据集任务下的评价指标
"""

import os
import json
import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
# 禁用transformers的安全提示
logging.set_verbosity_error()
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, f1_score, roc_auc_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 传统机器学习模型导入
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                            GradientBoostingClassifier, GradientBoostingRegressor,
                            AdaBoostClassifier, AdaBoostRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost未安装，跳过XGBoost模型")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM未安装，跳过LightGBM模型")

#  Monkey patch for input to auto-return 'y'
import builtins
original_input = builtins.input
builtins.input = lambda prompt='': 'y'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedFractureRiskEvaluator:
    def __init__(self):
        """初始化增强评估器"""
        # 骨折部位类别定义
        self.fracture_locations = [
            "骨盆", "肱骨近端", "脊柱-腰椎", "脊柱-胸椎", 
            "髋部-股骨颈", "髋部-粗隆", "腕部-桡骨远端", 
            "腕部-尺骨远端", "脊柱-脊柱", "无", "其他"
        ]
        
        # ===== 需要自定义填写：大模型配置 =====
        self.model_configs = {
            "Model_1": {
                "path": "/path/to/your/model1",  # 请填写您的第一个模型路径
                "name": "模型1名称"
            },
            "Model_2": {
                "path": "/path/to/your/model2",  # 请填写您的第二个模型路径
                "name": "模型2名称"
            },
            "Model_3": {
                "path": "/path/to/your/model3",  # 请填写您的第三个模型路径
                "name": "模型3名称"
            }
        }

        # ===== 需要自定义填写：数据路径配置 =====
        self.data_paths = {
            "train_data": "/path/to/your/train.json",  # 请填写训练数据路径
            "test_data": "/path/to/your/test.json"  # 请填写测试数据路径
        }
        
        # GPU配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 传统分类模型配置
        self.classification_models = {
            "逻辑回归": LogisticRegression(random_state=42, max_iter=1000),
            "决策树": DecisionTreeClassifier(random_state=42, max_depth=10),
            "随机森林": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
            "支持向量机": SVC(random_state=42, probability=True, kernel='rbf'),
            "K近邻": KNeighborsClassifier(n_neighbors=5),
            "朴素贝叶斯": GaussianNB(),
            "梯度提升": GradientBoostingClassifier(random_state=42, max_depth=6),
            "线性判别分析": LinearDiscriminantAnalysis(),
            "AdaBoost": AdaBoostClassifier(random_state=42)
        }
        
        # 传统回归模型配置
        self.regression_models = {
            "线性回归": LinearRegression(),
            "岭回归": Ridge(random_state=42, alpha=1.0),
            "Lasso回归": Lasso(random_state=42, alpha=0.1),
            "弹性网络": ElasticNet(random_state=42, alpha=0.1),
            "决策树回归": DecisionTreeRegressor(random_state=42, max_depth=10),
            "随机森林回归": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10),
            "支持向量回归": SVR(kernel='rbf', C=1.0),
            "K近邻回归": KNeighborsRegressor(n_neighbors=5),
            "梯度提升回归": GradientBoostingRegressor(random_state=42, max_depth=6),
            "AdaBoost回归": AdaBoostRegressor(random_state=42)
        }
        
        # 添加 XGBoost和LightGBM（如果可用）
        if XGBOOST_AVAILABLE:
            self.classification_models["XGBoost"] = xgb.XGBClassifier(random_state=42, max_depth=6)
            self.regression_models["XGBoost回归"] = xgb.XGBRegressor(random_state=42, max_depth=6)
        
        if LIGHTGBM_AVAILABLE:
            self.classification_models["LightGBM"] = lgb.LGBMClassifier(random_state=42, verbose=-1, max_depth=6)
            self.regression_models["LightGBM回归"] = lgb.LGBMRegressor(random_state=42, verbose=-1, max_depth=6)
    
    def load_model(self, model_path):
        """加载本地大模型"""
        try:
            print(f"正在加载模型: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                # 禁用安全提示
                revision="main"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                # 禁用安全提示
                revision="main"
            )
            model.eval()
            return tokenizer, model
        except Exception as e:
            print(f"模型加载失败 {model_path}: {e}")
            return None, None
    
    def load_data(self):
        """加载JSON格式的训练和测试数据"""
        try:
            # 加载训练数据
            with open(self.data_paths["train_data"], 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            # 加载测试数据
            with open(self.data_paths["test_data"], 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            print(f"训练数据样本数: {len(train_data)}")
            print(f"测试数据样本数: {len(test_data)}")
            
            return train_data, test_data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None, None
    
    def extract_features_from_input(self, input_text):
        """从输入文本中提取数值特征"""
        features = {}
        
        # 定义需要提取的数值特征
        numeric_patterns = {
            '年龄': r'就诊年龄\(岁\): (\d+)',
            '身高': r'身高\(cm\): ([\d.]+)',
            '体重': r'体重\(kg\): ([\d.]+)',
            'BMI': r'BMI: ([\d.]+)',
            '钠': r'钠\(mmol/L\): ([\d.]+)',
            '钾': r'钾\(mmol/L\): ([\d.]+)',
            '氯': r'氯\(mmol/L\): ([\d.]+)',
            '钙': r'钙\(mmol/L\): ([\d.]+)',
            '磷': r'磷\(mmol/L\): ([\d.]+)',
            '葡萄糖': r'葡萄糖\(mmol/L\): ([\d.]+)',
            '总蛋白': r'总蛋白\(g/L\): ([\d.]+)',
            '白蛋白': r'白蛋白\(g/L\): ([\d.]+)',
            '肌酐': r'肌酐\(μmol/L\): ([\d.]+)',
            '尿酸': r'尿酸\(umol/L\): ([\d.]+)',
            'C反应蛋白': r'C反应蛋白\(mg/L\): ([\d.>]+)',
            '碱性磷酸酶': r'碱性磷酸酶 \(U/L\): ([\d.]+)'
        }
        
        # 二元特征
        binary_patterns = {
            '性别_男': r'性别: 男',
            '骨松诊断': r'骨松诊断: 1\(有\)',
            '骨折诊断': r'骨折诊断: 1\(有\)',
            '脆性骨折人群': r'脆性骨折人群: 1\(有\)',
            '吸烟': r'吸烟: 1\.0\(有\)',
            '饮酒': r'饮酒: 1\.0\(有\)',
            '外伤史': r'外伤史: 1\(有\)',
            '糖尿病': r'糖尿病: 1\(有\)',
            '高血压': r'高血压: 1\(有\)',
            '脑梗塞': r'脑梗塞/腔隙性脑梗塞: 1\(有\)'
        }
        
        # 提取数值特征
        for feature_name, pattern in numeric_patterns.items():
            match = re.search(pattern, input_text)
            if match:
                try:
                    value = match.group(1).replace('>', '')
                    # 异常值处理
                    parsed_value = float(value)
                    # 基本范围检查
                    if feature_name == '年龄' and (parsed_value < 0 or parsed_value > 120):
                        parsed_value = np.nan
                    elif feature_name == 'BMI' and (parsed_value < 10 or parsed_value > 50):
                        parsed_value = np.nan
                    elif parsed_value < 0:  # 其他负值都设为异常
                        parsed_value = np.nan
                    features[feature_name] = parsed_value
                except:
                    features[feature_name] = np.nan
            else:
                features[feature_name] = np.nan
        
        # 提取二元特征
        for feature_name, pattern in binary_patterns.items():
            features[feature_name] = 1 if re.search(pattern, input_text) else 0
        
        return features
    
    def prepare_traditional_ml_data(self, data):
        """为传统机器学习模型准备数据"""
        print("正在准备传统机器学习数据...")
        
        features_list = []
        probabilities = []
        locations = []
        
        for sample in tqdm(data, desc="提取特征"):
            input_text = sample.get('input', '')
            output_text = sample.get('output', '')
            
            # 提取特征
            features = self.extract_features_from_input(input_text)
            features_list.append(features)
            
            # 提取真实标签
            prob, loc = self.parse_model_output(output_text)
            probabilities.append(prob if prob is not None else np.nan)
            locations.append(loc if loc is not None else "无")
        
        # 转换为DataFrame
        features_df = pd.DataFrame(features_list)
        
        # 数据清洗和预处理
        print(f"原始特征数量: {features_df.shape[1]}")
        print(f"原始样本数量: {features_df.shape[0]}")
        
        # 处理缺失值 - 使用中位数填充数值特征，0填充二元特征
        numeric_features = ['年龄', '身高', '体重', 'BMI', '钠', '钾', '氯', '钙', '磷', 
                           '葡萄糖', '总蛋白', '白蛋白', '肌酐', '尿酸', 'C反应蛋白', '碱性磷酸酶']
        binary_features = ['性别_男', '骨松诊断', '骨折诊断', '脆性骨折人群', '吸烟', 
                          '饮酒', '外伤史', '糖尿病', '高血压', '脑梗塞']
        
        # 填充数值特征的缺失值
        for col in numeric_features:
            if col in features_df.columns:
                median_val = features_df[col].median()
                if pd.isna(median_val):  # 如果中位数也是NaN，用0填充
                    median_val = 0
                features_df[col].fillna(median_val, inplace=True)
        
        # 填充二元特征的缺失值
        for col in binary_features:
            if col in features_df.columns:
                features_df[col].fillna(0, inplace=True)
        
        # 异常值处理 - 使用IQR方法
        for col in numeric_features:
            if col in features_df.columns:
                Q1 = features_df[col].quantile(0.25)
                Q3 = features_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 将异常值设为边界值
                features_df[col] = features_df[col].clip(lower_bound, upper_bound)
        
        # 处理概率标签
        probabilities = np.array(probabilities)
        valid_prob_mask = ~pd.isna(probabilities)
        
        # 将概率转换为0-1范围
        probabilities = probabilities / 100.0  # 从百分比转换为小数
        probabilities = np.clip(probabilities, 0, 1)  # 确保在0-1范围内
        
        print(f"处理后特征数量: {features_df.shape[1]}")
        print(f"有效概率样本数: {valid_prob_mask.sum()}")
        print("特征列表:", list(features_df.columns))
        
        return features_df, probabilities, locations, valid_prob_mask
    
    def train_traditional_models(self, X_train, y_train_reg, y_train_clf, valid_mask_reg, valid_mask_clf):
        """训练传统机器学习模型"""
        print("\n开始训练传统机器学习模型...")
        
        trained_models = {
            'regression': {},
            'classification': {}
        }
        
        # 只使用有效样本训练
        X_train_reg = X_train[valid_mask_reg]
        y_train_reg_valid = y_train_reg[valid_mask_reg]
        
        # 使用RobustScaler代替StandardScaler，对异常值更鲁棒
        scaler = RobustScaler()
        X_train_reg_scaled = scaler.fit_transform(X_train_reg)
        
        # 训练回归模型
        print("训练回归模型...")
        for name, model in tqdm(self.regression_models.items(), desc="回归模型"):
            try:
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train_reg_scaled, y_train_reg_valid)
                trained_models['regression'][name] = model_copy
                print(f"  {name} 训练完成")
            except Exception as e:
                print(f"模型 {name} 训练失败: {e}")
        
        # 处理分类任务
        valid_clf_indices = [i for i, (mask, label) in enumerate(zip(valid_mask_clf, y_train_clf)) 
                            if mask and label != "无" and label is not None]
        
        if len(valid_clf_indices) > 0:
            X_train_clf = X_train.iloc[valid_clf_indices]
            y_train_clf_valid = [y_train_clf[i] for i in valid_clf_indices]
            
            # 标准化分类特征
            scaler_clf = RobustScaler()
            X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
            
            # 编码分类标签
            le = LabelEncoder()
            y_train_clf_encoded = le.fit_transform(y_train_clf_valid)
            
            # 训练分类模型
            print("训练分类模型...")
            for name, model in tqdm(self.classification_models.items(), desc="分类模型"):
                try:
                    model_copy = type(model)(**model.get_params())
                    model_copy.fit(X_train_clf_scaled, y_train_clf_encoded)
                    trained_models['classification'][name] = model_copy
                    print(f"  {name} 训练完成")
                except Exception as e:
                    print(f"模型 {name} 训练失败: {e}")
        else:
            scaler_clf = None
            le = None
            print("没有足够的有效分类样本，跳过分类模型训练")
        
        return trained_models, scaler, scaler_clf, le
    
    def evaluate_traditional_models(self, trained_models, scaler, scaler_clf, le, 
                                   X_test, y_test_reg, y_test_clf, valid_mask_reg, valid_mask_clf):
        """评估传统机器学习模型"""
        print("\n评估传统机器学习模型...")
        
        results = {}
        
        # 评估回归模型
        if len(trained_models['regression']) > 0:
            X_test_reg = X_test[valid_mask_reg]
            y_test_reg_valid = y_test_reg[valid_mask_reg]
            
            if len(X_test_reg) > 0:
                X_test_reg_scaled = scaler.transform(X_test_reg)
                
                for name, model in trained_models['regression'].items():
                    try:
                        y_pred = model.predict(X_test_reg_scaled)
                        
                        # 确保预测值在合理范围内
                        y_pred = np.clip(y_pred, 0, 1)
                        
                        mae = mean_absolute_error(y_test_reg_valid, y_pred)
                        mse = mean_squared_error(y_test_reg_valid, y_pred)
                        rmse = np.sqrt(mse)
                        
                        results[name] = {
                            '回归指标': {
                                "样本数": len(y_test_reg_valid),
                                "MAE": mae,
                                "MSE": mse,
                                "RMSE": rmse
                            }
                        }
                        print(f"  {name}: RMSE = {rmse:.4f}")
                    except Exception as e:
                        print(f"回归模型 {name} 评估失败: {e}")
        
        # 评估分类模型
        if len(trained_models['classification']) > 0 and scaler_clf is not None and le is not None:
            valid_clf_indices = [i for i, (mask, label) in enumerate(zip(valid_mask_clf, y_test_clf)) 
                                if mask and label != "无" and label is not None]
            
            if len(valid_clf_indices) > 0:
                X_test_clf = X_test.iloc[valid_clf_indices]
                y_test_clf_valid = [y_test_clf[i] for i in valid_clf_indices]
                
                X_test_clf_scaled = scaler_clf.transform(X_test_clf)
                
                try:
                    y_test_clf_encoded = le.transform(y_test_clf_valid)
                    
                    for name, model in trained_models['classification'].items():
                        try:
                            y_pred = model.predict(X_test_clf_scaled)
                            y_pred_proba = model.predict_proba(X_test_clf_scaled) if hasattr(model, 'predict_proba') else None
                            
                            accuracy = accuracy_score(y_test_clf_encoded, y_pred)
                            precision = precision_score(y_test_clf_encoded, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test_clf_encoded, y_pred, average='weighted', zero_division=0)
                            
                            # 计算AUC-ROC
                            auc_roc = None
                            if y_pred_proba is not None and len(np.unique(y_test_clf_encoded)) > 1:
                                try:
                                    auc_roc = roc_auc_score(y_test_clf_encoded, y_pred_proba, 
                                                          average='weighted', multi_class='ovr')
                                except:
                                    auc_roc = None
                            
                            results[name] = {
                                '分类指标': {
                                    "样本数": len(y_test_clf_encoded),
                                    "Accuracy": accuracy,
                                    "Precision": precision,
                                    "F1": f1,
                                    "AUC-ROC": auc_roc
                                }
                            }
                            print(f"  {name}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
                        except Exception as e:
                            print(f"分类模型 {name} 评估失败: {e}")
                except Exception as e:
                    print(f"分类标签编码失败: {e}")
        
        return results
    
    def parse_model_output(self, output_text):
        """解析模型输出，提取概率和部位信息"""
        # 提取概率信息 - 支持多种格式
        prob_patterns = [
            r'骨折风险概率为(\d+\.?\d*)%',
            r'概率为(\d+\.?\d*)%',
            r'风险概率为(\d+\.?\d*)%',
            r'概率：(\d+\.?\d*)%'
        ]
        
        probability = None
        for pattern in prob_patterns:
            prob_match = re.search(pattern, output_text)
            if prob_match:
                try:
                    probability = float(prob_match.group(1))
                    # 确保概率在合理范围内
                    if probability < 0 or probability > 100:
                        probability = None
                    break
                except:
                    continue
        
        # 提取部位信息
        location = None
        # 按长度排序，优先匹配更具体的部位
        sorted_locations = sorted(self.fracture_locations, key=len, reverse=True)
        for loc in sorted_locations:
            if loc in output_text:
                location = loc
                break
        
        return probability, location
    
    def predict_single_sample(self, tokenizer, model, patient_input):
        """对单个样本进行预测"""
        try:
            # 直接使用患者输入作为模型输入，不添加prompt
            inputs = tokenizer(patient_input, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(patient_input, "").strip()
            
            probability, location = self.parse_model_output(response)
            return probability, location, response
        
        except Exception as e:
            print(f"预测失败: {e}")
            return None, None, ""
    
    def evaluate_llm_model(self, model_name, model_path, test_data):
        """评估单个大语言模型"""
        print(f"\n开始评估大模型: {model_name}")
        
        # 加载模型
        tokenizer, model = self.load_model(model_path)
        if tokenizer is None or model is None:
            return None
        
        predictions = {
            'probabilities': [],
            'locations': [],
            'responses': []
        }
        
        # 对测试数据进行预测
        for idx, sample in enumerate(tqdm(test_data, desc=f"评估{model_name}")):
            patient_input = sample.get('input', '')
            
            prob, loc, response = self.predict_single_sample(tokenizer, model, patient_input)
            predictions['probabilities'].append(prob)
            predictions['locations'].append(loc)
            predictions['responses'].append(response)
            
            # 每处理10个样本打印一次进度
            if (idx + 1) % 50 == 0:
                print(f"已处理 {idx + 1}/{len(test_data)} 个样本")
        
        # 清理GPU内存
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return predictions
    
    def calculate_regression_metrics(self, y_true, y_pred, valid_mask=None):
        # 先过滤掉None值和无效值
        if valid_mask is not None:
            # 使用有效掩码，但还要进一步过滤None值
            temp_indices = np.where(valid_mask)[0]
            valid_indices = [i for i in temp_indices 
                            if i < len(y_true) and i < len(y_pred) and 
                            y_true[i] is not None and y_pred[i] is not None and
                            not pd.isna(y_true[i]) and not pd.isna(y_pred[i])]
        else:
            # 过滤掉None值
            valid_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) 
                            if true is not None and pred is not None and 
                            not pd.isna(true) and not pd.isna(pred)]
        
        if len(valid_indices) == 0:
            return {"样本数": 0, "MAE": None, "MSE": None, "RMSE": None}
        
        # 提取有效值并转换为float
        y_true_valid = []
        y_pred_valid = []
        
        for i in valid_indices:
            try:
                true_val = float(y_true[i])
                pred_val = float(y_pred[i])
                
                # 检查是否为有限数值
                if np.isfinite(true_val) and np.isfinite(pred_val):
                    y_true_valid.append(true_val)
                    y_pred_valid.append(pred_val)
            except (ValueError, TypeError):
                continue
        
        if len(y_true_valid) == 0:
            return {"样本数": 0, "MAE": None, "MSE": None, "RMSE": None}
        
        # 转换为numpy数组
        y_true_valid = np.array(y_true_valid)
        y_pred_valid = np.array(y_pred_valid)
        
        # 确保预测值在合理范围内
        y_pred_valid = np.clip(y_pred_valid, 0, 1)
        
        try:
            mae = mean_absolute_error(y_true_valid, y_pred_valid)
            mse = mean_squared_error(y_true_valid, y_pred_valid)
            rmse = np.sqrt(mse)
            
            return {
                "样本数": len(y_true_valid),
                "MAE": mae,
                "MSE": mse, 
                "RMSE": rmse
            }
        except Exception as e:
            print(f"回归指标计算失败: {e}")
            return {"样本数": len(y_true_valid), "MAE": None, "MSE": None, "RMSE": None}
    
    def calculate_classification_metrics(self, y_true, y_pred, valid_mask=None):
        """计算分类任务指标"""
        if valid_mask is not None:
            # 使用有效掩码，并过滤掉"无"标签和None值
            temp_indices = np.where(valid_mask)[0]
            valid_indices = [i for i in temp_indices 
                            if i < len(y_true) and i < len(y_pred) and
                            y_true[i] is not None and y_pred[i] is not None and
                            str(y_true[i]) != "无" and str(y_pred[i]) != "无" and
                            str(y_true[i]) != "nan" and str(y_pred[i]) != "nan"]
        else:
            # 过滤掉None值和"无"标签
            valid_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) 
                            if true is not None and pred is not None and 
                            str(true) != "无" and str(pred) != "无" and
                            str(true) != "nan" and str(pred) != "nan"]
        
        if len(valid_indices) < 2:  # 至少需要2个样本进行分类评估
            return {"样本数": len(valid_indices), "Accuracy": None, "Precision": None, 
                   "F1": None, "AUC-ROC": None}
        
        # 提取有效标签
        y_true_valid = []
        y_pred_valid = []
        
        for i in valid_indices:
            try:
                true_label = str(y_true[i]).strip()
                pred_label = str(y_pred[i]).strip()
                
                if true_label and pred_label and true_label != "无" and pred_label != "无":
                    y_true_valid.append(true_label)
                    y_pred_valid.append(pred_label)
            except:
                continue
        
        if len(y_true_valid) < 2:
            return {"样本数": len(y_true_valid), "Accuracy": None, "Precision": None, 
                   "F1": None, "AUC-ROC": None}
        
        # 标签编码 
        unique_labels = list(set(y_true_valid + y_pred_valid))
        if len(unique_labels) < 2:
            return {"样本数": len(y_true_valid), "Accuracy": None, "Precision": None, 
                   "F1": None, "AUC-ROC": None}
        
        le = LabelEncoder()
        le.fit(unique_labels)
        
        try:
            y_true_encoded = le.transform(y_true_valid)
            y_pred_encoded = le.transform(y_pred_valid)
        except Exception as e:
            print(f"标签编码失败: {e}")
            return {"样本数": len(y_true_valid), "Accuracy": None, "Precision": None, 
                   "F1": None, "AUC-ROC": None}
        
        try:
            accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
            precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
            f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
            
            # 计算AUC-ROC (多分类)
            auc_roc = None
            try:
                if len(unique_labels) == 2:
                    # 二分类
                    auc_roc = roc_auc_score(y_true_encoded, y_pred_encoded)
                else:
                    # 多分类 - 使用one-vs-rest
                    from sklearn.preprocessing import label_binarize
                    y_true_bin = label_binarize(y_true_encoded, classes=range(len(le.classes_)))
                    y_pred_bin = label_binarize(y_pred_encoded, classes=range(len(le.classes_)))
                    auc_roc = roc_auc_score(y_true_bin, y_pred_bin, average='weighted', multi_class='ovr')
            except Exception as e:
                print(f"AUC-ROC计算失败: {e}")
                auc_roc = None
            
            return {
                "样本数": len(y_true_valid),
                "Accuracy": accuracy,
                "Precision": precision,
                "F1": f1,
                "AUC-ROC": auc_roc,
                "分类报告": classification_report(y_true_encoded, y_pred_encoded, 
                                             target_names=le.classes_, zero_division=0)
            }
        except Exception as e:
            print(f"分类指标计算失败: {e}")
            return {"样本数": len(y_true_valid), "Accuracy": None, "Precision": None, 
                   "F1": None, "AUC-ROC": None}
    
    def plot_enhanced_comparison_results(self, llm_results, ml_results):
        """绘制增强的对比结果图表"""
        # 创建更大的子图
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        
        # 合并所有结果
        all_results = {**llm_results, **ml_results}
        
        # 分离回归和分类模型
        regression_models = []
        classification_models = []
        
        for model_name, metrics in all_results.items():
            if '回归指标' in metrics:
                regression_models.append(model_name)
            if '分类指标' in metrics:
                classification_models.append(model_name)
        
        # 1. 回归指标对比 (MAE, MSE, RMSE)
        if regression_models:
            ax1 = axes[0, 0]
            metrics_to_plot = ['MAE', 'MSE', 'RMSE']
            x = np.arange(len(regression_models))
            width = 0.25
            
            for i, metric in enumerate(metrics_to_plot):
                values = []
                for model in regression_models:
                    if model in all_results and '回归指标' in all_results[model]:
                        value = all_results[model]['回归指标'].get(metric)
                        values.append(value if value is not None else 0)
                    else:
                        values.append(0)
                ax1.bar(x + i*width, values, width, label=metric, alpha=0.7)
            
            ax1.set_xlabel('模型')
            ax1.set_ylabel('指标值')
            ax1.set_title('回归任务指标对比 (MAE, MSE, RMSE)')
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(regression_models, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 分类指标对比
        if classification_models:
            ax3 = axes[1, 0]
            classification_metrics = ['Accuracy', 'Precision', 'F1', 'AUC-ROC']
            x = np.arange(len(classification_models))
            width = 0.2
            
            for i, metric in enumerate(classification_metrics):
                values = []
                for model in classification_models:
                    if model in all_results and '分类指标' in all_results[model]:
                        value = all_results[model]['分类指标'].get(metric)
                        values.append(value if value is not None else 0)
                    else:
                        values.append(0)
                ax3.bar(x + i*width, values, width, label=metric, alpha=0.7)
            
            ax3.set_xlabel('模型')
            ax3.set_ylabel('指标值')
            ax3.set_title('分类任务指标对比')
            ax3.set_xticks(x + 1.5*width)
            ax3.set_xticklabels(classification_models, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 3. 回归任务样本数对比
        if regression_models:
            ax4 = axes[1, 1]
            regression_sample_counts = []
            for model in regression_models:
                if model in all_results and '回归指标' in all_results[model]:
                    count = all_results[model]['回归指标'].get('样本数', 0)
                    regression_sample_counts.append(count)
                else:
                    regression_sample_counts.append(0)
            
            colors = ['lightcoral' if 'LLM' in model or any(llm in model for llm in ['DeepSeek', 'Qwen', 'Baichuan']) 
                     else 'skyblue' for model in regression_models]
            bars = ax4.bar(range(len(regression_models)), regression_sample_counts, color=colors, alpha=0.7)
            ax4.set_ylabel('有效样本数')
            ax4.set_title('回归任务有效样本数对比')
            ax4.set_xticks(range(len(regression_models)))
            ax4.set_xticklabels(regression_models, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (bar, count) in enumerate(zip(bars, regression_sample_counts)):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{count}', ha='center', va='bottom', fontsize=8)
        
        # 4. 分类任务样本数对比
        if classification_models:
            ax5 = axes[2, 0]
            classification_sample_counts = []
            for model in classification_models:
                if model in all_results and '分类指标' in all_results[model]:
                    count = all_results[model]['分类指标'].get('样本数', 0)
                    classification_sample_counts.append(count)
                else:
                    classification_sample_counts.append(0)
            
            colors = ['lightcoral' if 'LLM' in model or any(llm in model for llm in ['DeepSeek', 'Qwen', 'Baichuan']) 
                     else 'skyblue' for model in classification_models]
            bars = ax5.bar(range(len(classification_models)), classification_sample_counts, color=colors, alpha=0.7)
            ax5.set_ylabel('有效样本数')
            ax5.set_title('分类任务有效样本数对比')
            ax5.set_xticks(range(len(classification_models)))
            ax5.set_xticklabels(classification_models, rotation=45, ha='right')
            ax5.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (bar, count) in enumerate(zip(bars, classification_sample_counts)):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{count}', ha='center', va='bottom', fontsize=8)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if not os.path.exists('评估结果图表'):
            os.makedirs('评估结果图表')
        plt.savefig('评估结果图表/模型对比综合评估.png', dpi=300, bbox_inches='tight')
        print("图表已保存至 '评估结果图表' 文件夹")
        plt.close()
    
    def save_results_to_json(self, llm_results, ml_results):
        """将评估结果保存为JSON文件"""
        results = {
            "大模型评估结果": llm_results,
            "传统机器学习模型评估结果": ml_results
        }
        
        if not os.path.exists('评估结果'):
            os.makedirs('评估结果')
        
        with open('评估结果/模型评估结果.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("评估结果已保存至 '评估结果/模型评估结果.json'")
    
    def run(self):
        """运行完整的评估流程"""
        print("===== 骨折风险预测模型评估对比系统 =====")
        
        # 1. 加载数据
        train_data, test_data = self.load_data()
        if train_data is None or test_data is None:
            print("数据加载失败，无法继续评估")
            return
        
        # 2. 准备传统机器学习数据
        X_train, y_train_reg, y_train_clf, valid_mask_reg_train = self.prepare_traditional_ml_data(train_data)
        X_test, y_test_reg, y_test_clf, valid_mask_reg_test = self.prepare_traditional_ml_data(test_data)
        
        # 3. 训练传统机器学习模型
        valid_mask_clf_train = [True for _ in y_train_clf]
        valid_mask_clf_test = [True for _ in y_test_clf]
        
        trained_models, scaler, scaler_clf, le = self.train_traditional_models(
            X_train, y_train_reg, y_train_clf, valid_mask_reg_train, valid_mask_clf_train
        )
        
        # 4. 评估传统机器学习模型
        ml_results = self.evaluate_traditional_models(
            trained_models, scaler, scaler_clf, le,
            X_test, y_test_reg, y_test_clf, valid_mask_reg_test, valid_mask_clf_test
        )
        
        # 5. 评估大模型
        llm_results = {}
        for model_id, config in self.model_configs.items():
            predictions = self.evaluate_llm_model(config["name"], config["path"], test_data)
            if predictions is None:
                continue
            
            # 计算回归指标 (骨折风险概率)
            reg_metrics = self.calculate_regression_metrics(
                y_test_reg, predictions['probabilities'], valid_mask_reg_test
            )
            
            # 计算分类指标 (骨折部位)
            clf_metrics = self.calculate_classification_metrics(
                y_test_clf, predictions['locations'], valid_mask_clf_test
            )
            
            llm_results[config["name"]] = {
                '回归指标': reg_metrics,
                '分类指标': clf_metrics
            }
        
        # 6. 可视化结果
        self.plot_enhanced_comparison_results(llm_results, ml_results)
        
        # 7. 保存结果
        self.save_results_to_json(llm_results, ml_results)
        
        print("\n===== 评估完成 =====")
        
        # 恢复原始的input函数
        builtins.input = original_input

if __name__ == "__main__":
    evaluator = EnhancedFractureRiskEvaluator()
    evaluator.run()
