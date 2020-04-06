---
title: TensorFlow之特征处理和feature_column
author: 赵旭山
tags: TensorFlow
typora-root-url: ..

---



一直以来，除数据归一化等偏移操作外，对于特征工程朴素的理解就是特征组合。本次有机会通过`feature_column`深入理解了特征工程的其他方面。

本文主要参考《[TensorFlow 2 中文文档 - 特征工程结构化数据分类 ](https://geektutu.com/post/tf2doc-ml-basic-structured-data.html)》。

#### 1. 数据来源

数据集来自克利夫兰诊所心脏病基金会（Cleveland Clinic Foundation）提供的[303行14列心脏病数据](https://storage.googleapis.com/applied-dl/heart.csv)，每行描述一个患者，每列代表一个属性，详细的列描述参见[`heart.names`](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names)。

以下为本文用到的数据集中各列的说明。

| Column   | Description                                                  | Feature Type   | Data Type |
| -------- | ------------------------------------------------------------ | -------------- | --------- |
| Age      | Age in years                                                 | Numerical      | integer   |
| Sex      | (1 = male; 0 = female)                                       | Categorical    | integer   |
| CP       | Chest pain type (0, 1, 2, 3, 4)                              | Categorical    | integer   |
| Trestbpd | Resting blood pressure (in mm Hg on admission to the hospital) | Numerical      | integer   |
| Chol     | Serum cholesterol in mg/dl                                   | Numerical      | integer   |
| FBS      | (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)      | Categorical    | integer   |
| RestECG  | Resting electrocardiographic results (0, 1, 2)               | Categorical    | integer   |
| Thalach  | Maximum heart rate achieved                                  | Numerical      | integer   |
| Exang    | Exercise induced angina (1 = yes; 0 = no)                    | Categorical    | integer   |
| Oldpeak  | ST depression induced by exercise relative to rest           | Numerical      | integer   |
| Slope    | The slope of the peak exercise ST segment                    | Numerical      | float     |
| CA       | Number of major vessels (0~3) colored by flourosopy          | Numerical      | integer   |
| Thal     | 3 = normal; 6 = fixed defect; 7 = reversible defect          | Categorical    | string    |
| Target   | Diagnosis of heart disease (1 = true; 0 = false)             | Classification | integer   |





#### References:

* [TensorFlow 2 中文文档 - 特征工程结构化数据分类 ](https://geektutu.com/post/tf2doc-ml-basic-structured-data.html)
* [结构化数据分类实战：心脏病预测(tensorflow2.0官方教程翻译)](https://www.jianshu.com/p/2f08f77593e2)

