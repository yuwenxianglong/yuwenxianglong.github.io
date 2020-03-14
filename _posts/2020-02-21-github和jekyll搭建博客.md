---
title: Github和Jekyll搭建博客
author: 赵旭山
tags: 杂谈
article_header:
  type: cover
  image:
    src: //assets/images/Tunnels_Beach_Kauai_Hawaii_US.jpg
typora-root-url: ..
---



一直有自己搭建一个博客的想法，记录一些日常工作中的技巧和体会，并辅助治疗一下自己莫名袭来的懒惰情绪。但一直惧怕于自己非计算机专业的背景，迟迟懒得付诸实施。还好，今天迈出了第一步。



#### 1. 建立博客用github仓库

> Markdown语法插入相对路径，如`../assets/images/createNewRepository.png`，发布的网站是无法显示图片的。需转换为网站的绝对路径，此时`assets`目录为根目录下的目录，所以`/assets/images/createNewRepository.png`是工作的。

![](/assets/images/createNewRepository202002211200.png)

Repositoy Name分为两部分，用户名部分建议与github账号用户名相同，否则`用户名.github.io`会无法访问。后缀部分采用**github.io**，网上有教程说使用`github.com`也会造成无法访问。

![](/assets/images/repositoryName202002211246.png)

进入Repository页面后，切换到“**Settings**”选项卡，下拉点击“**Choose A Theme**”，可以选择一种Theme。之后，进入`yuwenxianglong.github.io`页面，可见已经完成了一个博客页面设计🦾🦾🦾！

![](/assets/images/githubioSettings202002211522.png)

![](/assets/images/chooseATheme202002211530.png)

![](/assets/images/selectTheme202002211531.png)

#### 2. 结合Jekyll实现jekyll-TeXt-theme风格页面设计

但笔者希望有一个文章的汇总列表，github模版中没找到，所以就想结合Jekyll做一点儿稍微那么高大上一点儿的事情🏋️‍♀️🏋️🏋️‍♂️！

##### 2.1 Jekyll安装和简单实用

安装Jekyll和bundler：

```bash
gem install jekyll
gem install bundler
```

详细信息参考：[https://jekyllrb.com/](https://jekyllrb.com/)

##### 2.2 使用jekyll-TeXt-theme风格页面

> jekyll-TeXt-theme的github代码发布主页：[https://github.com/kitian616/jekyll-TeXt-theme](https://github.com/kitian616/jekyll-TeXt-theme)

拉取*yourname*.github.io，并替换为jekyll-TeXt-theme代码，但`.git`文件夹要保留。

```bash
git clone -b master xxx.yourname.github.io.xxx
git clone -b master https://github.com/kitian616/jekyll-TeXt-theme.git
cd jekyll-TeXt-theme & bundle install & cd ..
cp -rf yourname.github.io backup
rm -rf yourname.github.io/*
rm -rf yourname.github.io/.*
cp -rf jekyll-TeXt-theme/* ./yourname.github.io/
cp -rf jekyll-TeXt-theme/.* ./yourname.github.io/
rm -rf jekyll-TeXt-theme/.git
cp -rf backup/.git ./yourname.github.io/
rm -rf backup
```

##### 2.3 Push更改后的代码

初始化本地`git repository`：

```bash
git init
git config user.name "yuwenxianglong"
git config user.email "yuwenxianglong@hotmail.com"
git add .
git commit -m "First commit."
git push
```

#### 3. That's all ! 👯‍♀️👯‍♀️👯‍♀️👯‍♀️

访问`yourname.github.io`，查看设计好的个人博客吧！