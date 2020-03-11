---
layout: post
title:  FUNSD数据集介绍
date:   2020-03-11 17:27:00 +0800
categories: NLP学习
tag: 开源数据集
---

* content
{:toc}


### 数据集介绍    
一个可用于FUNSD（噪声很多的扫描文档）上进行表单理解的数据集。

这里的表单理解是指对表单中的文本内容进行抽取，并生成结构化数据。

数据集包含199个真实的、完全注释的、扫描的表单。

文档有很多噪声，而且各种表单的外观差异很大，因此理解表单是一项很有挑战性的任务。

该数据集可用于各种任务，包括文本检测、光学字符识别、空间布局分析和实体标记/链接。

第一个具有完整注释的公共数据集，可用于处理FoUn任务。

### 数据集组成
这个数据集由原始图片（images）和标注结果（annotations）组成。

这些原始图片是e RVL-CDIP数据集的子集。e RVL-CDIP数据集是一个包含各种类型文档的灰度图片，
图片分辨率大约在100像素，共400000张。由于图片质量差且噪声非常多，作者从25000张图片中挑
选出3200张合格的图片（去掉了不可读和类似的），然后随机选择了199张进行标注。

标注结果为JSON格式,如下图：

![json_info](https://elvira521feng.github.io/YanQunJiang.github.io/styles/images/datasets/json_imformation.png)

注：  
1) box位置用左上右下两个点来确定，即box对应的4个值为[x0, y0, x1, y1]。  
2) lable的值有[question, answer, header, other]
3) linking对应的list为其指向的其他实体

### 训练集和测试集的数据分布  
数据分布统计情况
Split | Forms | Words | Entities | Relations  
-|-|-|-|-   
Training | 149 | 22, 512 | 7, 411 | 4, 236  
Testing | 50 | 8, 973 | 2, 332 | 1, 076
  
实体类别分布情况    
Split  | Header | Question | Answer | Other | Total  
-|-|-|-|-|-      
Training | 441 | 3, 266 | 2, 802 | 902 | 7, 411  
Testing | 122 | 1, 077 | 821 | 312 | 2, 332  


论文地址：https://arxiv.org/pdf/1905.13538.pdf
数据下载地址：https://guillaumejaume.github.io/FUNSD/


 
（注：若有错误希望大家指出！）

