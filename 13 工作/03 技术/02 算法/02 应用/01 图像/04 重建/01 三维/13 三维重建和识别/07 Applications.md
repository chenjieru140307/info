---
title: 07 Applications
toc: true
date: 2018-08-18 16:35:01
---



## Photo tourism


根据一些图，构建一个教堂场景

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/24dbIGIHcF.png?imageslim">
</p>

第二章图是他们拼出来的，第三张图是图库里面的。

http://phototour.cs.washington.edu/

大家如果想从事这个，从上面这个 paper 看起。

他们用到的最重要的就是 Key point detection 和 matching 在论文的 4.1 章有说，是 什么意思呢？一幅图里面首先要检测出关键点 key point，然后有了 matching 之后，求出不同 location 的坐标关系，从而构建出一个物体。

具体是 怎么 construct 的呢？公式他论文写在附录里面了。


## A Projector-Camera Systems

<span style="color:red;">基本没讲</span>


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/7hfLlFDl46.png?imageslim">
</p>


Projector-Camera calibration

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/BbadfkJ2bE.png?imageslim">
</p>

http://www.youtube.com/watch?v=YHhQSglmuqY&feature=channel_page

Our setup

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/759Ag5iGgh.png?imageslim">
</p>

Calibration procedure

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/jFmBIe5K6e.png?imageslim">
</p>


Quadrangle tracking
<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/GCkJbBKC1l.png?imageslim">
</p>

Experiments

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/7jI8ic4dkI.png?imageslim">
</p>

Projection result

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/BFaLg81heC.png?imageslim">
</p>

Results

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/aa2l87jKkH.png?imageslim">
</p>

Hand held direct manipulation 3D Display

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/LEH6j8i78B.png?imageslim">
</p>

http://www.youtube.com/watch?v=vVW9QXuKfoQ&feature=relmfu

Keystone correction

Configuration

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/k7EjfFCLBj.png?imageslim">
</p>


Aim of this work

Desired Results

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/l06cehej4d.png?imageslim">
</p>






Overview

- Three major modules
    - Projector-camera pair calibration
    - Projection region detection and tracking
    - Automatic keystone correction

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/73E7h3iebk.png?imageslim">
</p>


Pre-warp projection image


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/aK75dI575m.png?imageslim">
</p>


http://www.youtube.com/watch?v=y5XYdeh8Bno&list=UUfy2EumiHMeoUorMFR0woZA&index=1&feature=plcp




Keystone correction

Some real correction results


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180817/GCCKEb2gea.png?imageslim">
</p>








# 相关

- 七月在线 opencv计算机视觉
