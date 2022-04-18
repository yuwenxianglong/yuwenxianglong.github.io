---
title: Markdown 中的Timeline：Mdline
author: 赵旭山
tags: 杂谈 Markdown Mdline
typora-root-url: ..
---

&emsp;&emsp;需要画一个时间轴，梳理一下技术随时间的发展史。突发奇想，Markdown是否有插件能够实现呢？搜了一下也还真找到这个一个工具：mdline，Github发布地址为：[https://github.com/azu/mdline](https://github.com/azu/mdline)。

## 1. mdline的安装与使用

### 1.1 通过npm安装

```bash
npm install --global mdline
```

### 1.2 使用方法

&emsp;&emsp;Markdown无法直接对mdline进行渲染，所以需要转换为`html`格式嵌入Markdown使用。

#### 1.2.1 格式转换

```bash
mdline ./timeline.md -o timeline.html
```

或者：

```bash
npx mdline ./timeline.md -o timeline.html
```

#### 1.2.2 嵌入html标签

```
<iframe id="mdline" style="border:none;" seamless="seamless" src="/resdata/Mdline/timelineOfKnowledgeGraph.html" height="2100" width="100%"></iframe>
```

## 2. mdline语法

通过一个知识图谱发展史的例子来说明mdline的语法。

```
## 1960: Semantic Networks

语义网络作为知识表示的一种方法被提出，主要用于自然语言理解领域

## 1980: Ontology

哲学概念“本体”被引入人工智能领域用来刻画知识

## 1989: Web

Time Berners-Lee在欧洲高能物理研究中心发明了万维网

## 1998: The Semantic Web

Tim Berners-Lee提出了语义互联网的概念

## 1989: Linked Data

Tim Berners-Lee定义了在互联网上链接数据的四条原则

## 2012: Knowledge Graph

谷歌发布了其基于知识图谱的搜索引擎产品
```

***

效果如下。就是字体等比较大，显得与页面不太协调，暂时还不知道怎么完善。

<iframe id="mdline" style="border:none;" seamless="seamless" src="/resdata/Mdline/timelineOfKnowledgeGraph.html" height="2100" width="100%"></iframe>

## References

[markdown时间语法](https://markdown.jianguoyun.com/2945.html)

[mdline: Markdown timeline format and toolkit](https://github.com/azu/mdline)
