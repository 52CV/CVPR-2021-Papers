# CVPR2021最新信息及已接收论文/代码(持续更新)


本贴是对 CVPR2021 已接受论文的粗略汇总，后期会有更详细的总结。期待ing......

官网链接：http://cvpr2021.thecvf.com<br>
开会时间：2021年6月19日-6月25日<br>
论文接收公布时间：2021年2月28日<br>

接收论文IDs：<br>

* [CVPR 2021 接收论文列表！27%接受率！](https://zhuanlan.zhihu.com/p/353686917)

### :fireworks::fireworks::fireworks:更新提示：3月23日新增14篇

* 分割
  * [Cross-Dataset Collaborative Learning for Semantic Segmentation](https://arxiv.org/abs/2103.11351)
  * [Temporally-Weighted Hierarchical Clustering for Unsupervised Action Segmentation](https://arxiv.org/abs/2103.11264)<br>:star:[code](https://github.com/ssarfraz/FINCH-Clustering/tree/master/TW-FINCH)
* 医学
  * [Brain Image Synthesis with Unsupervised Multivariate Canonical CSCℓ4Net](https://arxiv.org/abs/2103.11587)<br>:open_mouth:oral
* 3D手部重建
  * [Model-based 3D Hand Reconstruction via Self-Supervised Learning](https://arxiv.org/abs/2103.11703)
* 度量学习（相似度学习）
  * [Dynamic Metric Learning: Towards a Scalable Metric Space to Accommodate Multiple Semantic Scales](https://arxiv.org/abs/2103.11781)<br>:star:[code](https://github.com/SupetZYK/DynamicMetricLearning)
* 视频理解
  * [Context-aware Biaffine Localizing Network for Temporal Sentence Grounding](https://arxiv.org/abs/2103.11555)<br>:star:[code](https://github.com/liudaizong/CBLN)
* 图像生成
  * [Context-Aware Layout to Image Generation with Enhanced Object Appearance](https://arxiv.org/abs/2103.11897)<br>:star:[code](https://github.com/wtliao/layout2img)
* NAS
  * [Prioritized Architecture Sampling with Monto-Carlo Tree Search](https://arxiv.org/abs/2103.11922)<br>:star:[code](https://github.com/xiusu/NAS-Bench-Macro)
* Re-Id
  * [Intra-Inter Camera Similarity for Unsupervised Person Re-Identification](https://arxiv.org/abs/2103.11658)<br>:star:[code](https://github.com/SY-Xuan/IICS)<br>论文公开
  * [Anchor-Free Person Search](https://arxiv.org/abs/2103.11617)<br>:star:[code](https://github.com/daodaofr/AlignPS)
* RGB-D 显著目标检测
  * [Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion](https://arxiv.org/abs/2103.11832)<br>:open_mouth:oral
* Transformer
  * [Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking](https://arxiv.org/abs/2103.11681)<br>:open_mouth:oral:star:[code](https://github.com/594422814/TransformerTrack)
  * [Multimodal Motion Prediction with Stacked Transformers](https://arxiv.org/abs/2103.11624)<br>:star:[code](https://github.com/Mrmoore98/mmTransformer-Multimodal-Motion-Prediction-with-Stacked-Transformers):house:[project](https://decisionforce.github.io/mmTransformer/):tv:[video](https://youtu.be/ytqS8dgVcx0)
* 未分
  * [PGT: A Progressive Method for Training Models on Long Videos](https://arxiv.org/abs/2103.11313)<br>:open_mouth:oral:star:[code](https://github.com/BoPang1996/PGT)


### :fireworks::fireworks::fireworks:更新提示：3月22日新增7篇

* 医学
  * [XProtoNet: Diagnosis in Chest Radiography with Global and Local Explanations](https://arxiv.org/abs/2103.10663)
* 点云
  * [Skeleton Merger: an Unsupervised Aligned Keypoint Detector](https://arxiv.org/abs/2103.10814)<br>:star:[code](https://github.com/eliphatfs/SkeletonMerger)
* 数据集
  * [Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark](https://arxiv.org/abs/2103.10895)<br>:house:[project](https://vap.aau.dk/sewer-ml/)
* 超像素
  * [Learning the Superpixel in a Non-iterative and Lifelong Manner](https://arxiv.org/abs/2103.10681)<br>论文公开
* 域适应
  * [Dynamic Transfer for Multi-Source Domain Adaptation](https://arxiv.org/abs/2103.10583)<br>:star:[code](https://github.com/liyunsheng13/DRT)  
* 模型压缩
  * [CDFI: Compression-Driven Network Design for Frame Interpolation](https://arxiv.org/abs/2103.10559)<br>:star:[code](https://github.com/tding1/CDFI)
* 未分
  * [Generic Perceptual Loss for Modeling Structured Output Dependencies](https://arxiv.org/abs/2103.10571)

# 目录

|:cat:|:dog:|:mouse:|:hamster:|:tiger:|
|------|------|------|------|------|
|:x:|[Workshop征稿](#*)|[58.度量学习](#58)|[57.手语识别](#57)|[56.光学、几何、光场成像](#56)|
|[55.图匹配](#55)|[54.情感预测](#54)|[53.数据集](#53)|[52.图像/视频生成](#52)|[51.对比学习](#51)|
|[50.OCR](#50)|[49.对抗学习](#49)|[48.图像表示](#48)|[47.视觉语言导航VLN](#47)|[46.人物交互HOI](#46)|
|[45.相机定位](#45)|[44.图像/视频字幕](#44)|[43.主动学习](#43)|[42.动作预测](#42)|[41.表示学习（图像+字幕）](#41)|
|[40.超像素](#40)|[39.模型偏见消除](#39)|[38.类增量学习](#38)|[37.持续学习](#37)|[36.动作检测与识别](#36)|
|[35.图像聚类](#35)|[34.图像/细粒度分类](#34)|[33.6D位姿估计](#33)|[32.视图合成](#32)|[31. 开放集识别](#31)|
|[30.新视角合成](#30)|[29.姿态估计](#29)|[28.密集预测](#28)|[27.活体检测](#27)|[26.视频相关技术](#26)|
|[25.三维视觉](#25)|[24.强化学习](#24)|[23.自动驾驶](#23)|[22.医学影像](#22)|[21.Transformer](#21)|
|[20.人员重识别/人群计数](#20)|[19.量化、剪枝、蒸馏、模型压缩与优化](#19)|[18.航空影像](#18)|[17.超分辨率](#17)|[16.视觉问答](#16)|
|[15.GAN](#15)|[14.小/零样本学习，域适应，域泛化](#14)|[13.图像检索](#13)|[12.图像增广](#12)|[11.人脸技术](#11)|
|[10.神经架构搜索](#10)|[9.目标跟踪](#9)|[8.图像分割](#8)|[7.目标检测](#7)|[6.数据增强](#6)|
|[5.异常检测](#5)|[4.自/半/弱监督学习](#4)|[3.点云](#3)|[2.图卷积网络GNN](#2)|[1.未分类](#1)|

<a name="58"/>

## 58.度量学习（相似度学习）
  * [Dynamic Metric Learning: Towards a Scalable Metric Space to Accommodate Multiple Semantic Scales](https://arxiv.org/abs/2103.11781)<br>:star:[code](https://github.com/SupetZYK/DynamicMetricLearning)

<a name="57"/>

## 57.手语识别
  * [Skeleton Based Sign Language Recognition Using Whole-body Keypoints](https://arxiv.org/abs/2103.08833)<br>:star:[code](https://github.com/jackyjsy/CVPR21Chal-SLR)

<a name="56"/>

## 56.光学、几何、光场成像
  * [Deep Gaussian Scale Mixture Prior for Spectral Compressive Imaging](https://arxiv.org/abs/2103.07152)<br>:star:[code](https://github.com/TaoHuang95/DGSMP):house:[project](https://see.xidian.edu.cn/faculty/wsdong/Projects/DGSM-SCI.htm)


<a name="55"/>

## 55.图匹配
  * [Deep Graph Matching under Quadratic Constraint](https://arxiv.org/abs/2103.06643)

<a name="54"/>

## 54.情感预测
  * [Affect2MM: Affective Analysis of Multimedia Content Using Emotion Causality](https://arxiv.org/abs/2103.06541)<br>:house:[project](https://gamma.umd.edu/researchdirections/affectivecomputing/affect2mm/)

<a name="53"/>

## 53.数据集
  * [Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts](https://arxiv.org/abs/2102.08981)<br>:sunflower:[dataset](https://github.com/google-research-datasets/conceptual-12m)
  * [Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark](https://arxiv.org/abs/2103.10895)<br>:house:[project](https://vap.aau.dk/sewer-ml/)

<a name="52"/>

## 52.图像/视频生成Image Generation

- [Spatially-Adaptive Pixelwise Networks for Fast Image Translation](https://arxiv.org/abs/2012.02992)<br>:house:[project](https://tamarott.github.io/ASAPNet_web/)<br>采用超网络和隐式函数，极快的图像到图像翻译速度（比基线快18倍）
- [Image Generators with Conditionally-Independent Pixel Synthesis](https://arxiv.org/abs/2011.13775)<br>:open_mouth:oral:star:[code](https://github.com/saic-mdal/CIPS)
* [Im2Vec: Synthesizing Vector Graphics without Vector Supervision](https://arxiv.org/abs/2102.02798)<br>:open_mouth:oral:star:[code](https://github.com/preddy5/Im2Vec):house:[project](http://geometry.cs.ucl.ac.uk/projects/2021/im2vec/)
* 图像生成
  * [Context-Aware Layout to Image Generation with Enhanced Object Appearance](https://arxiv.org/abs/2103.11897)<br>:star:[code](https://github.com/wtliao/layout2img) 
* 视频生成
  * [Playable Video Generation](https://arxiv.org/abs/2101.12195)<br>:open_mouth:oral:star:[code](https://github.com/willi-menapace/PlayableVideoGeneration):house:[project](https://willi-menapace.github.io/playable-video-generation-website/):tv:[video](https://www.youtube.com/watch?v=QtDjSyZERpg)

<a name="51"/>

## 51.对比学习
* [AdCo: Adversarial Contrast for Efficient Learning of Unsupervised Representations from Self-Trained Negative Adversaries](https://arxiv.org/abs/2011.08435)<br>:star:[code](https://arxiv.org/abs/2011.08435)<br>解读:[CVPR 2021接收论文：AdCo基于对抗的对比学习](https://mp.weixin.qq.com/s/u7Lhzh8uYEEHfWiM32-4yQ)

<a name="50"/>

## 50.OCR

* 场景文本检测
  * [What If We Only Use Real Datasets for Scene Text Recognition? Toward Scene Text Recognition With Fewer Labels](https://arxiv.org/abs/2103.04400)
  * [Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition](https://arxiv.org/abs/2103.06495)<br>:open_mouth:oral:star:[code](https://github.com/FangShancheng/ABINet)

<a name="49"/>

## 49.对抗学习

- [Simulating Unknown Target Models for Query-Efficient Black-box Attacks](https://arxiv.org/abs/2009.00960)<br>:star:[code](https://github.com/machanic/MetaSimulator)<br>黑盒对抗攻击
- Delving into Data: Effectively Substitute Training for Black-box Attack<br>基于高效训练替代模型的黑盒攻击方法<br>解读：[8](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)


<a name="48"/>

## 48.图像表示Image Representation

- [Learning Continuous Image Representation with Local Implicit Image Function](https://arxiv.org/abs/2012.09161)<br>:open_mouth:oral:star:[code](https://github.com/yinboc/liif):house:[project](https://yinboc.github.io/liif/):tv:[video](https://youtu.be/6f2roieSY_8)

<a name="47"/>

## 47.视觉语言导航vision-language navigation

- [Structured Scene Memory for Vision-Language Navigation](https://arxiv.org/abs/2103.03454)

<a name="46"/>

## 46.人物交互（human-object interaction）

- [Learning Asynchronous and Sparse Human-Object Interaction in Videos](https://arxiv.org/abs/2103.02758)
- [QPIC: Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information](https://arxiv.org/abs/2103.05399)<br>:star:[code](https://github.com/hitachi-rd-cv/qpic)
- [Reformulating HOI Detection as Adaptive Set Prediction](https://arxiv.org/abs/2103.05983)
* [Detecting Human-Object Interaction via Fabricated Compositional Learning](https://arxiv.org/abs/2103.08214)<br>:star:[code](https://github.com/zhihou7/FCL)

<a name="45"/>

## 45.相机定位(Camera Localization)

- [Robust Neural Routing Through Space Partitions for Camera Relocalization in Dynamic Indoor Environments](https://arxiv.org/abs/2012.04746)<br>:open_mouth:oral
- [Back to the Feature: Learning Robust Camera Localization from Pixels to Pose](https://arxiv.org/abs/2103.09213)<br>:star:[code](https://github.com/cvg/pixloc)

<a name="44"/>

## 44.图像/视频字幕

- [Scan2Cap: Context-aware Dense Captioning in RGB-D Scans](https://arxiv.org/abs/2012.02206)<br>:star:[code](https://github.com/daveredrum/Scan2Cap):house:[project](https://daveredrum.github.io/Scan2Cap/):tv:[video](https://youtu.be/AgmIpDbwTCY)
- [VX2TEXT: End-to-End Learning of Video-Based Text Generation From Multimodal Inputs](https://arxiv.org/abs/2101.12059)<br>视频字幕、视频问答和视频对话任务的多模式框架
- [Open-book Video Captioning with Retrieve-Copy-Generate Network](https://arxiv.org/abs/2103.05284)

<a name="43"/>

## 43.主动学习

- [Vab-AL: Incorporating Class Imbalance and Difficulty with Variational Bayes for Active Learning](https://arxiv.org/abs/2003.11249)

<a name="42"/>

## 42.动作预测

- [Learning the Predictability of the Future](https://arxiv.org/abs/2101.01600)<br>预测未来<br>:star:[code](https://github.com/cvlab-columbia/hyperfuture):house:[project](https://hyperfuture.cs.columbia.edu/):tv:[video](https://www.youtube.com/watch?v=-Uy92jvT_90)<br>



<a name="41"/>

## 41.表示学习（图像+字幕）

- [VirTex: Learning Visual Representations from Textual Annotations](https://arxiv.org/abs/2006.06666)<br>:star:[code](https://github.com/kdexd/virtex)

<a name="40"/>

## 40.超像素

- [Learning the Superpixel in a Non-iterative and Lifelong Manner](https://arxiv.org/abs/2103.10681)<br>论文公开

<a name="39"/>

## 39.模型偏见消除

- [Fair Attribute Classification through Latent Space De-biasing](https://arxiv.org/abs/2012.01469)<br>:star:[code](https://github.com/princetonvisualai/gan-debiasing):house:[project](https://princetonvisualai.github.io/gan-debiasing/)<br>

<a name="38"/>

## 38.类增量学习（class-incremental learning）

- [IIRC: Incremental Implicitly-Refined Classification](https://arxiv.org/abs/2012.12477)<br>:house:[project](https://chandar-lab.github.io/IIRC/)<br>
- [Semantic-aware Knowledge Distillation for Few-Shot Class-Incremental Learning](https://arxiv.org/abs/2103.04059)

<a name="37"/>

## 37.持续学习

- Rainbow Memory: Continual Learning with a Memory of Diverse Samples<br>
- [Training Networks in Null Space for Continual Learning]()<br>:open_mouth:oral:star:[code](https://github.com/ShipengWang/Adam-NSCL)


<a name="36"/>

## 36.动作检测与识别

- [Coarse-Fine Networks for Temporal Activity Detection in Videos](https://arxiv.org/abs/2103.01302)<br>
- [3D CNNs with Adaptive Temporal Feature Resolutions](https://arxiv.org/abs/2011.08652)<br> 
- [Understanding the Robustness of Skeleton-based Action Recognition under Adversarial Attack](https://arxiv.org/abs/2103.05347)
- [BASAR:Black-box Attack on Skeletal Action Recognition](https://arxiv.org/abs/2103.05266)<br>:tv:[video](https://www.youtube.com/watch?v=PjWgwnAkV8g)
- [TDN: Temporal Difference Networks for Efficient Action Recognition]( https://arxiv.org/abs/2012.10071)<br>:star:[code](https://github.com/MCG-NJU/TDN)
- [ACTION-Net: Multipath Excitation for Action Recognition](https://arxiv.org/abs/2103.07372)<br>:star:[code](https://github.com/V-Sense/ACTION-Net)<br>解读：[CVPR 2021 | 用于动作识别，即插即用、混合注意力机制的 ACTION 模块](https://mp.weixin.qq.com/s/L2_lkhKbVhW8fjAaDdsyWQ)
* 时序动作定位
  * [Modeling Multi-Label Action Dependencies for Temporal Action Localization](https://arxiv.org/abs/2103.03027)<br>:open_mouth:oral<br>提出基于注意力的网络架构来学习视频中的动作依赖性，用于解决多标签时间动作定位任务。
  * Learning Salient Boundary Feature for Anchor-free Temporal Action Localization<br>基于显著边界特征学习的无锚框时序动作定位<br>解读：[10](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)


<a name="35"/>

## 35.图像聚类

- [Improving Unsupervised Image Clustering With Robust Learning](https://arxiv.org/abs/2012.11150)<br>:star:[code](https://github.com/deu30303/RUC)<br>利用鲁棒学习改进无监督图像聚类技术<br>

<a name="34"/>

## 34.图像分类

- [PML: Progressive Margin Loss for Long-tailed Age Classification](https://arxiv.org/abs/2103.02140)<br>
- [Re-labeling ImageNet: from Single to Multi-Labels, from Global to Localized Labels](https://arxiv.org/abs/2101.05022)<br>:star:[code](https://github.com/naver-ai/relabel_imagenet)<br>
- [Fine-grained Angular Contrastive Learning with Coarse Labels](https://arxiv.org/abs/2012.03515)<br>:open_mouth:oral<br>使用自监督进行 Coarse Labels（粗标签）的细粒度分类方面的工作。粗标签与细粒度标签相比，更容易和更便宜，因为细粒度标签通常需要域专家。
- Graph-based High-Order Relation Discovery for Fine-grained Recognition<br>基于特征间高阶关系挖掘的细粒度识别方法<br>解读：[20](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)

<a name="33"/>

## 33.6D位姿估计

- [FFB6D: A Full Flow Bidirectional Fusion Network for 6D Pose Estimation](https://arxiv.org/abs/2103.02242)<br>:star:[code](https://github.com/ethnhe/FFB6D)<br>
- [GDR-Net: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation](http://arxiv.org/abs/2102.12145)<br>:star:[code](https://github.com/THU-DA-6D-Pose-Group/GDR-Net)
- [FS-Net: Fast Shape-based Network for Category-Level 6D Object Pose Estimation with Decoupled Rotation Mechanism](https://arxiv.org/abs/2103.07054)<br>:open_mouth:oral:star:[code](https://github.com/DC1991/FS-Net)

<a name="32"/>

## 32.视图合成

- [ID-Unet: Iterative Soft and Hard Deformation for View Synthesis](https://arxiv.org/abs/2103.02264)
- [NeX: Real-time View Synthesis with Neural Basis Expansion](https://arxiv.org/abs/2103.05606)<br>:open_mouth:oral:house:[project](https://nex-mpi.github.io/):tv:[video](https://www.youtube.com/watch?v=HyfkF7Z-ddA)<br>利用神经基础扩展的实时视图合成技术

<a name="31"/>

## 31.开放集识别

- [Counterfactual Zero-Shot and Open-Set Visual Recognition](https://arxiv.org/abs/2103.00887)<br>:star:[code](https://github.com/yue-zhongqi/gcm-cf)<br>
- [Few-shot Open-set Recognition by Transformation Consistency](https://arxiv.org/abs/2103.01537)<br>

<a name="30"/>

## 30.新视角合成

- [DeRF: Decomposed Radiance Fields](https://arxiv.org/abs/2011.12490)<br>:house:[project](https://ubc-vision.github.io/derf/)<br>
- [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961)<br>:house:[project](https://www.albertpumarola.com/research/D-NeRF/index.html)<br>

<a name="29"/>

## 29.姿态估计

- [PCLs: Geometry-aware Neural Reconstruction of 3D Pose with Perspective Crop Layers](https://arxiv.org/abs/2011.13607)<br>:tv:[video](https://twitter.com/i/status/1334395954644930560)<br>通过消除 location-dependent 透视效果来改进3D人体姿势估计技术工作。<br>
- [CanonPose: Self-supervised Monocular 3D Human Pose Estimation in the Wild](https://arxiv.org/abs/2011.14679)
- [Camera-Space Hand Mesh Recovery via Semantic Aggregation and Adaptive 2D-1D Registration](https://arxiv.org/abs/2103.02845)<br>:star:[code](https://github.com/SeanChenxy/HandMesh)<br>
- [Monocular Real-time Full Body Capture with Inter-part Correlations](https://arxiv.org/abs/2012.06087)<br>:tv:[video](https://www.youtube.com/watch?v=pAcywTUTv-E)
- [Behavior-Driven Synthesis of Human Dynamics](https://arxiv.org/abs/2103.04677)<br>:star:[code](https://github.com/CompVis/behavior-driven-video-synthesis):house:[project](https://compvis.github.io/behavior-driven-video-synthesis/)
- Context Modeling in 3D Human Pose Estimation: A Unified Perspective
- [Learning Compositional Representation for 4D Captures with Neural ODE](https://arxiv.org/abs/2103.08271)
* 3D手部重建
  * [Model-based 3D Hand Reconstruction via Self-Supervised Learning](https://arxiv.org/abs/2103.11703)

<a name="28"/>

## 28.密集预测

- [Densely connected multidilated convolutional networks for dense prediction tasks](https://arxiv.org/abs/2011.11844)<br>提出的D3Net在语义分割&音乐源分离任务上的表现优于SOTA网络<br>
- [Dense Contrastive Learning for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2011.09157)<br>:star:[code](https://github.com/WXinlong/DenseCL)
* [Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning](https://arxiv.org/abs/2011.10043)<br>:star:[code](https://github.com/zdaxie/PixPro)

<a name="27"/>

## 27.活体检测

- [Cross Modal Focal Loss for RGBD Face Anti-Spoofing](https://arxiv.org/abs/2103.00948)<br>

<a name="26"/>

## 26.视频相关技术

* [VideoMoCo: Contrastive Video Representation Learning with Temporally Adversarial Examples](https://arxiv.org/abs/2103.05905)<br>:star:[code](https://github.com/tinapan-pt/VideoMoCo)
* 视频摘要
  * [Learning Discriminative Prototypes with Dynamic Time Warping](https://arxiv.org/abs/2103.09458)<br>:star:[code](https://github.com/BorealisAI/TSC-Disc-Proto)
* 视频编解码
  * [MetaSCI: Scalable and Adaptive Reconstruction for Video Compressive Sensing](https://arxiv.org/abs/2103.01786)<br>:star:[code](https://github.com/xyvirtualgroup/MetaSCI-CVPR2021)
* 视频插帧
  * [FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation](https://arxiv.org/pdf/2012.08512.pdf)<br>:star:[code](https://tarun005.github.io/FLAVR/Code):house:[project](https://tarun005.github.io/FLAVR/)<br>
* 视频语言学习（video-and-language learning）
  * [Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling](https://arxiv.org/pdf/2102.06183.pdf)<br>:open_mouth:oral:star:[code](https://github.com/jayleicn/ClipBERT)<br>
* 视频预测
  * [Greedy Hierarchical Variational Autoencoders for Large-Scale Video Prediction](https://arxiv.org/abs/2103.04174)<br>:house:[project](https://sites.google.com/view/ghvae):tv:[video](https://youtu.be/C8_-z8SEGOU)
* 视频理解
  * [Context-aware Biaffine Localizing Network for Temporal Sentence Grounding](https://arxiv.org/abs/2103.11555)<br>:star:[code](https://github.com/liudaizong/CBLN)

<a name="25"/>

## 25.三维视觉

- [A Deep Emulator for Secondary Motion of 3D Characters](https://arxiv.org/abs/2103.01261)<br>
- [Neural Deformation Graphs for Globally-consistent Non-rigid Reconstruction](https://arxiv.org/abs/2012.01451)<br>:open_mouth:oral:house:[project](https://aljazbozic.github.io/neural_deformation_graphs/):tv:[video](https://www.youtube.com/watch?v=vyq36eFkdWo)<br>
- [Deep Implicit Templates for 3D Shape Representation](https://arxiv.org/abs/2011.14565)<br>:open_mouth:oral:star:[code](https://github.com/ZhengZerong/DeepImplicitTemplates):house:[project](http://www.liuyebin.com/dit/dit.html):tv:[video](http://www.liuyebin.com/dit/assets/supp_vid.mp4)<br>[CVPR 2021 Oral，清华学者提出Deep Implicit Templates，极大扩展DIF能力](https://zhuanlan.zhihu.com/p/354737798)<br>
- [SMPLicit: Topology-aware Generative Model for Clothed People](https://arxiv.org/abs/2103.06871)<br>:house:[project](http://www.iri.upc.edu/people/ecorona/smplicit/)
* 深度估计
  * [PLADE-Net: Towards Pixel-Level Accuracy for Self-Supervised Single-View Depth Estimation with Neural Positional Encoding and Distilled Matting Loss](https://arxiv.org/abs/2103.07362)
  * [Beyond Image to Depth: Improving Depth Prediction using Echoes](https://arxiv.org/abs/2103.08468)<br>:star:[code](https://github.com/krantiparida/beyond-image-to-depth):house:[project](https://krantiparida.github.io/projects/bimgdepth.html)
  * [Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos](https://arxiv.org/abs/2103.03319)<br>:open_mouth:oral:star:[code](https://github.com/yasaminjafarian/HDNet_TikTok):house:[project](https://www.yasamin.page/hdnet_tiktok):tv:[video](https://youtu.be/EFJ8WXdKghs)


<a name="24"/> 

## 24.强化学习
- [Hierarchical and Partially Observable Goal-driven Policy Learning with Goals Relational Graph](https://arxiv.org/abs/2103.01350)<br>
- [Unsupervised Learning for Robust Fitting:A Reinforcement Learning Approach](https://arxiv.org/abs/2103.03501)

<a name="23"/> 

## 23.自动驾驶

- [Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition](https://arxiv.org/abs/2103.01486)<br>:star:[code](https://github.com/QVPR/Patch-NetVLAD)<br>ECCV 2020 Facebook Mapillary Visual Place Recognition Challenge 冠军方案

<a name="22"/> 

## 22.医学影像

- [3D Graph Anatomy Geometry-Integrated Network for Pancreatic Mass Segmentation, Diagnosis, and Quantitative Patient Management](https://arxiv.org/abs/2012.04701)<br>用纯多模态 CT 影像可替代目前 JHMI 的需要做肿瘤化学检测和 DNA 测序+医学影像的综合多模态诊断流程，从诊断准确度上有可比较性，定量诊断精度更优<br>
- [Deep Lesion Tracker: Monitoring Lesions in 4D Longitudinal Imaging Studies](https://arxiv.org/abs/2012.04872)<br>肿瘤影像里面智能 PACS 辅助医生读片的重要功能<br>
- [Automatic Vertebra Localization and Identification in CT by Spine Rectification and Anatomically-constrained Optimization](https://arxiv.org/abs/2012.07947)<br>基于CT 影像的骨折/骨质疏松系统<br>
- [Multi-institutional Collaborations for Improving Deep Learning-based Magnetic Resonance Image Reconstruction Using Federated Learning](https://arxiv.org/abs/2103.02148)<br>:star:[code](https://github.com/guopengf/FL-MRCM)<br>多机构合作，利用联合学习改进基于深度学习的磁共振图像重建技术<br>
- [DeepTag: An Unsupervised Deep Learning Method for Motion Tracking on Cardiac Tagging Magnetic Resonance Images]()<br>:open_mouth:oral:star:[code](https://github.com/DeepTag/cardiac_tagging_motion_estimation)<br>DeepTag: 一种无监督的深度学习方法，用于心脏标记磁共振图像的运动跟踪<br>
- [Multiple Instance Captioning: Learning Representations from Histopathology Textbooks and Articles](https://arxiv.org/abs/2103.05121)
* [XProtoNet: Diagnosis in Chest Radiography with Global and Local Explanations](https://arxiv.org/abs/2103.10663)
* 医学图像分割
  * [FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space](https://arxiv.org/abs/2103.06030)
  * [DoDNet: Learning to segment multi-organ and tumors from multiple partially labeled datasets](https://arxiv.org/abs/2011.10217)<br>:star:[code](https://github.com/jianpengz/DoDNet)
* 医学图像合成
  * [Brain Image Synthesis with Unsupervised Multivariate Canonical CSCℓ4Net](https://arxiv.org/abs/2103.11587)<br>:open_mouth:oral


<a name="21"/> 

## 21.Transformer 

- [Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/pdf/2012.09838.pdf)<br>:star:[code](https://github.com/hila-chefer/Transformer-Explainability)<br> 
- [MIST: Multiple Instance Spatial Transformer Network](https://arxiv.org/abs/1811.10725)<br>试图从热图中进行可微的top-K选择(MIST)（目前在自然图像上也有了一些结果；) 用它可以在没有任何定位监督的情况下进行检测和分类（并不是它唯一能做的事情!）
* 动作识别检测
  * 3D Vision Transformers for Action Recognition<br>用于动作识别的3D视觉Transformer
* 目标检测
  * [UP-DETR: Unsupervised Pre-training for Object Detection with Transformers](https://arxiv.org/pdf/2011.09094.pdf)<br>:open_mouth:oral:star:[code](https://github.com/dddzg/up-detr)
* 图像处理
  * [Pre-Trained Image Processing Transformer](https://arxiv.org/pdf/2012.00364.pdf)<br>
* 人机交互
  * [End-to-End Human Object Interaction Detection with HOI Transformer](https://arxiv.org/abs/2103.04503)<br>:star:[code](https://github.com/bbepoch/HoiTransformer)
* 图像分割
  * [Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers](https://arxiv.org/abs/2012.15840)<br>:star:[code](https://github.com/fudan-zvg/SETR):house:[project](https://fudan-zvg.github.io/SETR/)<br>基于Transformers从序列到序列的角度重新思考语义分割<br>解读：[16](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)<br>解读：[Transformer 在语义分割中的应用，曾位ADE20K 榜首（44.42% mIoU）](https://zhuanlan.zhihu.com/p/341768446)
  * [VisTR: End-to-End Video Instance Segmentation with Transformers](https://arxiv.org/abs/2011.14503)<br>:open_mouth:oral:star:[code](https://github.com/Epiphqny/VisTR)
* 跟踪
  * [Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking](https://arxiv.org/abs/2103.11681)<br>:open_mouth:oral:star:[code](https://github.com/594422814/TransformerTrack)
* 动作预测
  * [Multimodal Motion Prediction with Stacked Transformers](https://arxiv.org/abs/2103.11624)<br>:star:[code](https://github.com/Mrmoore98/mmTransformer-Multimodal-Motion-Prediction-with-Stacked-Transformers):house:[project](https://decisionforce.github.io/mmTransformer/):tv:[video](https://youtu.be/ytqS8dgVcx0)  

<a name="20"/> 

## 20.人员重识别

- [Meta Batch-Instance Normalization for Generalizable Person Re-Identification](https://arxiv.org/abs/2011.14670)<br>
- [Watching You: Global-guided Reciprocal Learning for Video-based Person Re-identification](https://arxiv.org/abs/2103.04337)
- [Joint Noise-Tolerant Learning and Meta Camera Shift Adaptation for Unsupervised Person Re-Identification](https://arxiv.org/abs/2103.04618)<br>:star:[code](https://github.com/FlyingRoastDuck/MetaCam_DSCE)
- [Self-supervised 3D Reconstruction and Re-Projection for Texture Insensitive Person Re-identification]<br>基于自监督三维重建和重投影的纹理不敏感行人重识别<br>解读：[12](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
- [Intra-Inter Camera Similarity for Unsupervised Person Re-Identification](https://arxiv.org/abs/2103.11658)<br>:star:[code](https://github.com/SY-Xuan/IICS)<br>论文公开
- [Anchor-Free Person Search](https://arxiv.org/abs/2103.11617)<br>:star:[code](https://github.com/daodaofr/AlignPS)
* 拥挤人群计数
  * [Cross-Modal Collaborative Representation Learning and a Large-Scale RGBT Benchmark for Crowd Counting](https://arxiv.org/abs/2012.04529)

<a name="19"/> 


## 19.量化、剪枝、蒸馏、模型压缩/扩展与优化

- Learning Student Networks in the Wild<br>
- [ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network](https://arxiv.org/abs/2007.00992)<br>:star:[code](https://github.com/clovaai/rexnet)<br>
- [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)<br>:star:[code](https://github.com/megvii-model/RepVGG)<br>
- [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907)<br>:star:[code](https://github.com/Andrew-Qibin/CoordAttention)
* 剪枝
  * [Manifold Regularized Dynamic Network Pruning](https://arxiv.org/abs/2103.05861)
* 模型扩展
  * [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877)<br>:star:[code](https://github.com/facebookresearch/pycls)
* 量化  
  * [Learnable Companding Quantization for Accurate Low-bit Neural Networks](https://arxiv.org/abs/2103.07156)
  * [Diversifying Sample Generation for Accurate Data-Free Quantization](https://arxiv.org/abs/2103.01049)
* 知识蒸馏
  * [Refine Myself by Teaching Myself: Feature Refinement via Self-Knowledge Distillation](https://arxiv.org/abs/2103.08273)<br>:star:[code](https://github.com/MingiJi/FRSKD)
* 可逆神经网络
  * [Neural Parts: Learning Expressive 3D Shape Abstractions with Invertible Neural Networks](https://arxiv.org/abs/2103.10429)<br>:house:[project](https://paschalidoud.github.io/neural_parts)
* 模型压缩
  * [CDFI: Compression-Driven Network Design for Frame Interpolation](https://arxiv.org/abs/2103.10559)<br>:star:[code](https://github.com/tding1/CDFI)

<a name="18"/> 

## 18.航空影像

- Dogfight: Detecting Drones from Drone Videos<br>
* 航空影像分割
  * [PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation](https://arxiv.org/abs/2103.06564)<br>:star:[code](https://github.com/lxtGH/PFSegNets)
* 航空影像检测
  * [ReDet: A Rotation-equivariant Detector for Aerial Object Detection](https://arxiv.org/abs/2103.07733)<br>:star:[code](https://github.com/csuhan/ReDet)

<a name="17"/> 

## 17.超分辨率

- Data-Free Knowledge Distillation For Image Super-Resolution<br>
- [AdderSR: Towards Energy Efficient Image Super-Resolution](https://arxiv.org/pdf/2009.08891.pdf)<br>:star:[code](https://github.com/huawei-noah/AdderNet)<br>
- [Cross-MPI: Cross-scale Stereo for Image Super-Resolution using Multiplane Images](https://arxiv.org/abs/2011.14631)<br>:house:[project](http://www.liuyebin.com/crossMPI/crossMPI.html):tv:[video](http://www.liuyebin.com/crossMPI/assets/supp_vid.mp4)<br>[CVPR 2021，Cross-MPI以底层场景结构为线索的端到端网络，在大分辨率（x8）差距下也可完成高保真的超分辨率](https://zhuanlan.zhihu.com/p/354752197)
- [ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic](https://arxiv.org/abs/2103.04039)<br>:star:[code](https://github.com/Xiangtaokong/ClassSR)
* Robust Reference-based Super-Resolution via C²-Matching
* [GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution](https://arxiv.org/abs/2012.00739)<br>:open_mouth:oral:house:[project](https://ckkelvinchan.github.io/projects/GLEAN/)
* [BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond](https://arxiv.org/abs/2012.02181)<br>:star:[code](https://github.com/ckkelvinchan/BasicVSR-IconVSR):house:[project](https://ckkelvinchan.github.io/projects/BasicVSR/)
* [Temporal Modulation Network for Controllable Space-Time Video Super-Resolution]<br>[作者主页](https://csjunxu.github.io/)<br>基于时空特征可控插值的视频超分辨率网络<br>解读：[18](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)

<a name="16"/> 


## 16.视觉问答

- Weakly-supervised Grounded Visual Question Answering using Capsules<br>
* [Counterfactual VQA: A Cause-Effect Look at Language Bias](https://arxiv.org/abs/2006.04315)<br>:star:[code](https://github.com/yuleiniu/cfvqa)

<a name="15"/> 

## 15.GAN
- Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing<br>
- [Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation](https://arxiv.org/abs/2008.00951)<br>:star:[code](https://github.com/eladrich/pixel2style2pixel):house:[project](https://eladrich.github.io/pixel2style2pixel/)<br>
- [Hijack-GAN: Unintended-Use of Pretrained, Black-Box GANs](https://arxiv.org/pdf/2011.14107.pdf)<br>
- [Image-to-image Translation via Hierarchical Style Disentanglement](https://arxiv.org/abs/2103.01456)<br>:star:[code](https://github.com/imlixinyang/HiSD)
- [Efficient Conditional GAN Transfer with Knowledge Propagation across Classes](https://arxiv.org/abs/2102.06696)<br>:star:[code](https://github.com/mshahbazi72/cGANTransfer)
- [Anycost GANs for Interactive Image Synthesis and Editing](https://arxiv.org/abs/2103.03243)<br>:star:[code](https://github.com/mit-han-lab/anycost-gan):house:[project](https://hanlab.mit.edu/projects/anycost-gan/):tv:[video](https://www.youtube.com/watch?v=_yEziPl9AkM&t=90s)<br>Anycost GAN，可适应广泛的硬件和延迟要求，以及实现交互式图像编辑
- [TediGAN: Text-Guided Diverse Image Generation and Manipulation](https://arxiv.org/abs/2012.03308)<br>:star:[code](https://github.com/weihaox/TediGAN):house:[project](https://xiaweihao.com/projects/tedigan/):tv:[video](https://www.youtube.com/watch?v=L8Na2f5viAM)
- [Generative Hierarchical Features from Synthesizing Images](https://arxiv.org/abs/2007.10379)<br>:open_mouth:oral:star:[code](https://github.com/genforce/ghfeat):house:[project](https://genforce.github.io/ghfeat/)<br>作者称预训练 GAN 生成器可以当作是一种学习的多尺度损失。用它进行训练可以带来高度竞争的层次化和分离的视觉特征，称之为生成层次化特征（GH-Feat）。并进一步表明，GH-Feat不仅有利于生成性任务，更重要的是有利于分辨性任务，包括人脸验证、关键点检测、layout prediction、迁移学习、style mixing、图像编辑等。
- [Teachers Do More Than Teach: Compressing Image-to-Image Models](https://arxiv.org/abs/2103.03467)
- [PISE: Person Image Synthesis and Editing with Decoupled GAN](https://arxiv.org/abs/2103.04023)<br>:star:[code](https://github.com/Zhangjinso/PISE)
- [LOHO: Latent Optimization of Hairstyles via Orthogonalization](https://arxiv.org/abs/2103.03891)
- [Image-to-image Translation via Hierarchical Style Disentanglement](https://arxiv.org/abs/2103.01456)<br>:open_mouth:oral:star:[code](https://github.com/imlixinyang/HiSD)<br>在图像到图像翻译上实现层次风格解耦
- [CoMoGAN: continuous model-guided image-to-image translation](https://arxiv.org/abs/2103.06879)<br>:open_mouth:oral:star:[code](https://github.com/cv-rits/CoMoGAN)
- [HumanGAN: A Generative Model of Humans Images](https://arxiv.org/abs/2103.06902)
- [HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms](https://arxiv.org/abs/2011.11731)<br>:star:[code](https://github.com/mahmoudnafifi/HistoGAN)
- [DivCo: Diverse Conditional Image Synthesis via Contrastive Generative Adversarial Network](https://arxiv.org/abs/2103.07893)<br>:star:[code](https://github.com/ruiliu-ai/DivCo)

<a name="14"/> 


## 14.小/零样本学习，域适应，域泛化

* 小样本学习
  * Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning<br>
  * [Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning](https://arxiv.org/abs/2103.01315)<br>
  * [Learning Dynamic Alignment via Meta-filter for Few-shot Learning]<br>[作者主页](https://yanweifu.github.io/page3.html)<br>通过元卷积核实现基于动态对齐的小样本学习<br>解读：[17](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
* 域泛化
  * [FSDR: Frequency Space Domain Randomization for Domain Generalization](https://arxiv.org/abs/2103.02370)<br>受 JPEG 将空间图像转换为多个频率分量(FCs)的启发，提出频率空间域随机化(FSDR)，通过保留域变量FCs(DIFs)和只随机化域变量FCs(DVFs)来随机化频率空间的图像。
  * [Domain Generalization via Inference-time Label-Preserving Target Projections](https://arxiv.org/abs/2103.01134)<br>:open_mouth: Oral
* 零样本学习
  * [Goal-Oriented Gaze Estimation for Zero-Shot Learning](https://arxiv.org/abs/2103.03433):star:[code](https://github.com/osierboy/GEM-ZSL)
* 域适应
  * [Dynamic Transfer for Multi-Source Domain Adaptation](https://arxiv.org/abs/2103.10583)<br>:star:[code](https://github.com/liyunsheng13/DRT)  


<a name="13"/> 


## 13.图像/视频检索
* 图像检索
  * [Probabilistic Embeddings for Cross-Modal Retrieval](https://arxiv.org/abs/2101.05068)<br>
  * [QAIR: Practical Query-efficient Black-Box Attacks for Image Retrieval](https://arxiv.org/abs/2103.02927)
* 视频检索
  * [On Semantic Similarity in Video Retrieval](https://arxiv.org/abs/2103.10095)<br>:star:[code](https://github.com/mwray/Semantic-Video-Retrieval):house:[project](https://mwray.github.io/SSVR/):tv:[video](https://youtu.be/pS9qa_B771I)



<a name="12"/> 

## 12.图像增强

* 图像恢复Image Restoration
  * [Multi-Stage Progressive Image Restoration](https://arxiv.org/abs/2102.02808)<br>:star:[code](https://github.com/swz30/MPRNet)<br>
* 去阴影Shadow Removal
  * [Auto-Exposure Fusion for Single-Image Shadow Removal](https://arxiv.org/abs/2103.01255)<br>:star:[code](https://github.com/tsingqguo/exposure-fusion-shadow-removal)<br>
* 去模糊Deblurring
  * [DeFMO: Deblurring and Shape Recovery of Fast Moving Objects](https://arxiv.org/abs/2012.00595)<br>:star:[code](https://github.com/rozumden/DeFMO):tv:[video](https://www.youtube.com/watch?v=pmAynZvaaQ4)<br>
  * [ARVo: Learning All-Range Volumetric Correspondence for Video Deblurring](https://arxiv.org/abs/2103.04260)
* 去反射Reflection Removal
  * [Robust Reflection Removal with Reflection-free Flash-only Cues](https://arxiv.org/abs/2103.04273)<br>:star:[ccode](https://github.com/ChenyangLEI/flash-reflection-removal)
* 去雾
  * Learning to Restore Hazy Video: A New Real-World Dataset and A New Method<br>学习复原有雾视频：一种新的真实数据集及算法<br>解读：[9](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
  * Contrastive Learning for Compact Single Image Dehazing<br>基于对比学习的紧凑图像去雾方法<br>解读：[5](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
* 去噪Denoising
  * [Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images](https://arxiv.org/abs/2101.02824)
* 去雨Deraining
  * [Semi-Supervised Video Deraining with Dynamic Rain Generator](https://arxiv.org/abs/2103.07939)
* 曝光校正
  * [Learning Multi-Scale Photo Exposure Correction](https://arxiv.org/abs/2003.11596)<br>:star:[code](https://github.com/mahmoudnafifi/Exposure_Correction)
* 图像修复Image Inpainting
  * [Generating Diverse Structure for Image Inpainting With Hierarchical VQ-VAE](https://arxiv.org/abs/2103.10022)<br>:star:[code](https://github.com/USTC-JialunPeng/Diverse-Structure-Inpainting)

<a name="11"/> 

## 11. 人脸技术

* 人脸识别
  * [A 3D GAN for Improved Large-pose Facial Recognition](https://arxiv.org/pdf/2012.10545.pdf)<br>
  * [When Age-Invariant Face Recognition Meets Face Age Synthesis: A Multi-Task Learning Framework](https://arxiv.org/abs/2103.01520)<br>:star:[github](https://github.com/Hzzone/MTLFace)<br>
  * [MagFace: A Universal Representation for Face Recognition and Quality Assessment](https://arxiv.org/abs/2103.06627)<br>:open_mouth:oral:star:[code](https://github.com/IrvingMeng/MagFace)<br>人脸识别+质量，今年的Oral presentation。 代码待整理
  * [WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition](https://arxiv.org/abs/2103.04098)<br>:house:[project](https://www.face-benchmark.org/)
  * [ForgeryNet: A Versatile Benchmark for Comprehensive Forgery Analysis](https://arxiv.org/abs/2103.05630)<br>:open_mouth:oral:house:[project](https://yinanhe.github.io/projects/forgerynet.html):tv:[video](https://youtu.be/e8XIL3Di2Y8) 
  * Spherical Confidence Learning for Face Recognition<br>:open_mouth:oral<br>基于超球流形置信度学习的人脸识别
  * Consistent Instance False Positive Improves Fairness in Face Recognition<br>基于实例误报一致性的人脸识别公平性提升方法<br>解读：[7](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
  * [CRFace: Confidence Ranker for Model-Agnostic Face Detection Refinement](https://arxiv.org/abs/2103.07017)
  * [Cross-Domain Similarity Learning for Face Recognition in Unseen Domains](https://arxiv.org/abs/2103.07503)
* 合成人脸（Deepfake/Face Forgery）检测
  * [Multi-attentional Deepfake Detection](https://arxiv.org/abs/2103.02406)<br>
  * [Frequency-aware Discriminative Feature Learning Supervised by Single-Center Loss for Face Forgery Detection](https://arxiv.org/abs/2103.09096)
* 人脸质量评估
  * [SDD-FIQA: Unsupervised Face Image Quality Assessment with Similarity Distribution Distance](https://arxiv.org/abs/2103.05977)<br>基于相似度分布距离的无监督人脸质量评估<br>解读：[6](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
* 3D人脸重建
  * Learning to Aggregate and Personalize 3D Face from In-the-Wild Photo Collection<br>:open_mouth:oral<br>在开放的人像集合中学习3D人脸的聚合与特异化重建
  * [3DCaricShop: A Dataset and A Baseline Method for Single-view 3D Caricature Face Reconstruction](https://arxiv.org/abs/2103.08204)<br>:star:[code](https://github.com/qiuyuda/3DCaricShop):house:[project](https://qiuyuda.github.io/3DCaricShop/)
  
<a name="10"/> 

## 10.神经架构搜索

- [AttentiveNAS: Improving Neural Architecture Search via Attentive](https://arxiv.org/pdf/2011.09011.pdf)<br>
- [HourNAS: Extremely Fast Neural Architecture Search Through an Hourglass Lens](https://arxiv.org/pdf/2005.14446.pdf)<br>
- [ReNAS: Relativistic Evaluation of Neural Architecture Search](https://arxiv.org/pdf/1910.01523.pdf)<br>
- [OPANAS: One-Shot Path Aggregation Network Architecture Search for Object](https://arxiv.org/abs/2103.04507)
- Towards Improving the Consistency, Efficiency, and Flexibility of Differentiable Neural Architecture Search<br>北京大学人工智能研究院机器学习研究中心
- [Contrastive Neural Architecture Search with Neural Architecture Comparators](https://arxiv.org/abs/2103.05471)<br>:star:[code](https://arxiv.org/abs/2103.05471)
- [Searching by Generating: Flexible and Efficient One-Shot NAS with Architecture Generator](https://arxiv.org/abs/2103.07289)<br>:star:[code](https://github.com/eric8607242/SGNAS)
- [Prioritized Architecture Sampling with Monto-Carlo Tree Search](https://arxiv.org/abs/2103.11922)<br>:star:[code](https://github.com/xiusu/NAS-Bench-Macro)

<a name="9"/> 

## 9.目标跟踪

- [Rotation Equivariant Siamese Networks for Tracking](https://arxiv.org/abs/2012.13078)<br>
- [Graph Attention Tracking](https://arxiv.org/abs/2011.11204)<br>:star:[code](https://github.com/ohhhyeahhh/SiamGAT)
* 多目标跟踪
  * [Probabilistic Tracklet Scoring and Inpainting for Multiple Object Tracking](https://arxiv.org/abs/2012.02337)<br>
  * [Track to Detect and Segment: An Online Multi-Object Tracker](https://arxiv.org/abs/2103.08808)<br>:star:[code](https://github.com/JialianW/TraDeS):house:[project](https://jialianwu.com/projects/TraDeS.html):tv:[video](https://youtu.be/oGNtSFHRZJA)<br>TraDeS ：CVPR 2021多目标跟踪算法，改进了目前联合检测与跟踪的在线方法，使用跟踪线索辅助检测，在多个数据集实现了大幅精度提升，作者来自纽约州立大学。代码已开源。
  * Multiple Object Tracking with Correlation Learning
  * [Learning a Proposal Classifier for Multiple Object Tracking](https://arxiv.org/abs/2103.07889)<br>:star:[code](https://github.com/daip13/LPC_MOT)

<a name="8"/> 

## 8.图像分割

- [Information-Theoretic Segmentation by Inpainting Error Maximization](https://arxiv.org/abs/2012.07287)<br>
- [Simultaneously Localize, Segment and Rank the Camouflaged Objects](https://arxiv.org/abs/2103.04011)<br>:star:[code](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment)
- [Capturing Omni-Range Context for Omnidirectional Segmentation](https://arxiv.org/abs/2103.05687)<br>:star:[code](https://github.com/elnino9ykl/WildPASS)
* 实例分割
  * [Zero-Shot Instance Segmentation]<br>创新奇智首次提出零样本实例分割，助力解决工业场景数据瓶颈难题
* 全景分割
  * [4D Panoptic LiDAR Segmentation](https://arxiv.org/abs/2102.12472)<br>
  * [Cross-View Regularization for Domain Adaptive Panoptic Segmentation](https://arxiv.org/abs/2103.02584)<br>:open_mouth:oral<br>用于域自适应全景分割的跨视图正则化方法<br>
  * Part-aware Panoptic Segmentation<br>
  * Toward Joint Thing-and-Stuff Mining for Weakly Supervised Panoptic Segmentation<br>联合物体和物质挖掘的弱监督全景分割<br>解读：[15](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
* 语义分割
  * [PLOP: Learning without Forgetting for Continual Semantic Segmentation](https://arxiv.org/abs/2011.11390)<br>:star:[code](https://github.com/arthurdouillard/CVPR2021_PLOP)
  * [Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges](https://arxiv.org/abs/2009.03137)<br>:star:[dataset](https://github.com/QingyongHu/SensatUrban):tv:[video](https://www.youtube.com/watch?v=IG0tTdqB3L8)<br>
  * [Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation](https://arxiv.org/abs/2103.04717)
  * [Semi-supervised Domain Adaptation based on Dual-level Domain Mixing for Semantic Segmentation](https://arxiv.org/abs/2103.04705)
  * [Differentiable Multi-Granularity Human Representation Learning for Instance-Aware Human Semantic Parsing](https://arxiv.org/abs/2103.04570)<br>:open_mouth:oral:star:[code](https://github.com/tfzhou/MG-HumanParsing)
  * [Learning Statistical Texture for Semantic Segmentation](https://arxiv.org/abs/2103.04133)
  * [MetaCorrection: Domain-aware Meta Loss Correction for Unsupervised Domain Adaptation in Semantic Segmentation](https://arxiv.org/abs/2103.05254)<br>:star:[code](https://github.com/cyang-cityu/MetaCorrection)<br>语义分割中的无监督域适应的域感知元损失校正
  * [Continual Semantic Segmentation via Repulsion-Attraction of Sparse and Disentangled Latent Representations](https://arxiv.org/abs/2103.06342)
  * [Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion](https://arxiv.org/abs/2103.07074)<br>:star:[code](https://github.com/ShiQiu0419/BAAF-Net)
  * [Rethinking BiSeNet For Real-time Semantic Segmentation]<br>:star:[code](https://github.com/MichaelFan01/STDC-Seg)
  * [BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation](https://arxiv.org/abs/2103.08907)<br>:star:[code](https://github.com/jbeomlee93/BBAM)
  * [Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2103.08896)<br>:star:[code](https://github.com/jbeomlee93/AdvCAM)
  * [Cross-Dataset Collaborative Learning for Semantic Segmentation](https://arxiv.org/abs/2103.11351)
* 场景理解/场景解析
  * [Exploring Data Efficient 3D Scene Understanding with Contrastive Scene Contexts](https://arxiv.org/abs/2012.09165)<br>:open_mouth:oral:house:[project](https://sekunde.github.io/project_efficient/):tv:[video](https://youtu.be/E70xToZLgs4)
  * [Exploiting Edge-Oriented Reasoning for 3D Point-based Scene Graph Analysis](https://arxiv.org/abs/2103.05558)<br>:house:[project](https://sggpoint.github.io/)<br>利用面向边缘的推理进行基于3D点的场景图分析---场景理解
  * [Probabilistic Modeling of Semantic Ambiguity for Scene Graph Generation](https://arxiv.org/abs/2103.05271)<br>场景图生成---场景解析
  * [Monte Carlo Scene Search for 3D Scene Understanding](https://arxiv.org/abs/2103.07969)
* 抠图
  * [Real-Time High Resolution Background Matting](https://arxiv.org/abs/2012.07810)<br>:open_mouth:oral:star:[code](https://github.com/PeterL1n/BackgroundMattingV2):house:[project](https://grail.cs.washington.edu/projects/background-matting-v2/):tv:[video](https://youtu.be/oMfPTeYDF9g)<br>最新开源抠图技术，实时快速高分辨率，4k(30fps)、现代GPU（60fps）<br>解读：[单块GPU实现4K分辨率每秒30帧，华盛顿大学实时视频抠图再升级，毛发细节到位](https://mp.weixin.qq.com/s/0OJR3Y5cPfeHhdTdI3BgEA)<br>[最新开源抠图技术，实时快速高分辨率，4k(30fps)、现代GPU（60fps）](https://zhuanlan.zhihu.com/p/337028483)
* 视频动作分割
  * [Global2Local: Efficient Structure Search for Video Action Segmentation](https://arxiv.org/abs/2101.00910)<br>从全局到局部：面向视频动作分割的高效网络结构搜索<br>解读：[19](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
* 时序动作分割
  * [Temporal Action Segmentation from Timestamp Supervision](https://arxiv.org/abs/2103.06669)<br>:star:[code](https://github.com/ZheLi2020/TimestampActionSeg)
  * [Temporally-Weighted Hierarchical Clustering for Unsupervised Action Segmentation](https://arxiv.org/abs/2103.11264)<br>:star:[code](https://github.com/ssarfraz/FINCH-Clustering/tree/master/TW-FINCH)
* 雷达分割
  * [Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation](https://arxiv.org/abs/2011.10033)<br>:open_mouth:oral:star:[code](https://github.com/xinge008/Cylinder3D)<br>在 SemanticKITTI 榜单排名第一（until CVPR DDL），在 nuScenes 中获得 SOTA，并对其他基于激光雷达的任务保持了良好的泛化能力，包括激光雷达全景分割和激光雷达三维检测，其中就基于此工作，在 SemanticKITTI 全景分割榜单也排名第一。
* 视频目标分割
  * [Modular Interactive Video Object Segmentation:Interaction-to-Mask, Propagation and Difference-Aware Fusion]()<br>:open_mouth:oral:star:[code](https://github.com/hkchengrex/MiVOS):house:[project](https://hkchengrex.github.io/MiVOS/):tv:[video](https://hkchengrex.github.io/MiVOS/video.html)
  * [Learning to Recommend Frame for Interactive Video Object Segmentation in the Wild](https://arxiv.org/abs/2103.10391)<br>:star:[code](https://github.com/svip-lab/IVOS-W)

<a name="7"/> 

## 7.目标检测

- [Multiple Instance Active Learning for Object Detection](https://github.com/yuantn/MIAL/raw/master/paper.pdf)<br>:star:[code](https://github.com/yuantn/MIAL)<br>
- Positive-Unlabeled Data Purification in the Wild for Object Detection<br>
- [Depth from Camera Motion and Object Detection](https://arxiv.org/abs/2103.01468)<br>:star:[github](https://github.com/griffbr/ODMD):tv:[video](https://www.youtube.com/watch?v=GruhbdJ2l7k)<br>
- [Towards Open World Object Detection](https://arxiv.org/abs/2103.02603)<br>:open_mouth:oral:star:[code](https://github.com/JosephKJ/OWOD)<br>
- [General Instance Distillation for Object Detection](https://arxiv.org/abs/2103.02340)<br>
- Distilling Object Detectors via Decoupled Features<br>
- [MeGA-CDA: Memory Guided Attention for Category-Aware Unsupervised Domain Adaptive Object Detection](https://arxiv.org/abs/2103.04224)<br>
- Informative and Consistent Correspondence Mining for Cross-Domain Weakly Supervised Object Detection<br>:open_mouth:oral
* [You Only Look One-level Feature](https://arxiv.org/abs/2103.09460)<br>:star:[code](https://github.com/megvii-model/YOLOF)<br>论文公开<br>[开源 YOLOF，无需 FPN，速度比 YOLOv4 快13%](https://zhuanlan.zhihu.com/p/357986047)<br>解读：[目标检测算法YOLOF：You Only Look One-level Feature](https://mp.weixin.qq.com/s/0ChHOljrqrXk8yWi_S6ITg)
- [Sparse R-CNN: End-to-End Object Detection with Learnable Proposals](https://arxiv.org/abs/2011.12450)<br>:star:[code](https://github.com/PeizeSun/SparseR-CNN)
* 小样本目标检测
  * [Semantic Relation Reasoning for Shot-Stable Few-Shot Object Detection](https://arxiv.org/abs/2103.01903)<br>首个研究少样本检测任务的语义关系推理，并证明它可提升强基线的潜。 <br> 
  * Dense Relation Distillation with Context-aware Aggregation for Few-Shot Object Detection<br>北京大学人工智能研究院机器学习研究中心<br>
  * [FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding](https://arxiv.org/abs/2103.05950)<br>:star:[code](https://github.com/bsun0802/FSCE)
* 多目标检测
  * [There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge](https://arxiv.org/abs/2103.01353)<br>:house:[project](https://rl.uni-freiburg.de/)<br>
* 3D目标检测
  * [Categorical Depth Distribution Network for Monocular 3D Object Detection](https://arxiv.org/abs/2103.01100)<br>:open_mouth:oral<br>
  * [3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection](https://arxiv.org/abs/2012.04355)<br>:star:[code](https://github.com/thu17cyz/3DIoUMatch):house:[project](https://thu17cyz.github.io/3DIoUMatch/):tv:[video](https://youtu.be/nuARjhkQN2U)<br>更多：[CVPR 2021|利用IoU预测进行半监督式3D目标检测](https://zhuanlan.zhihu.com/p/354618636)
  * [ST3D: Self-training for Unsupervised Domain Adaptation on 3D ObjectDetection](https://arxiv.org/abs/2103.05346)<br>:star:[code](https://github.com/CVMI-Lab/ST3D)
  * Depth-conditioned Dynamic Message Propagation for Monocular 3D Object Detection
* 旋转目标检测
  * [Dense Label Encoding for Boundary Discontinuity Free Rotation Detection](https://arxiv.org/abs/2011.09670)<br>:star:[code](https://github.com/Thinklab-SJTU/DCL_RetinaNet_Tensorflow)
* 目标定位
  * [Unveiling the Potential of Structure-Preserving for Weakly Supervised Object Localization](https://arxiv.org/abs/2103.04523v1)<br>基于结构信息保持的弱监督目标定位<br>解读：[13](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
* 密集目标检测
  * [Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection]()<br>:star:[code](https://github.com/implus/GFocalV2)<br>解读：[目标检测无痛涨点之 Generalized Focal Loss V2](https://mp.weixin.qq.com/s/H3LuCuqKCUNFldzqiPWQXg)
* RGB-D 显著目标检测
  * [Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion](https://arxiv.org/abs/2103.11832)<br>:open_mouth:oral
  
<a name="6"/> 

## 6.数据增广

- [KeepAugment: A Simple Information-Preserving Data Augmentation](https://arxiv.org/pdf/2011.11778.pdf)<br>

<a name="5"/> 

## 5.异常检测

- [Multiresolution Knowledge Distillation for Anomaly Detection](https://arxiv.org/abs/2011.11108)<br>

<a name="4"/> 

## 4.自/半/弱监督学习

* 弱监督
  * [Weakly Supervised Learning of Rigid 3D Scene Flow](https://arxiv.org/pdf/2102.08945.pdf)<br>:star:[code](https://arxiv.org/pdf/2102.08945.pdf):house:[project](https://3dsceneflow.github.io/)<br>
* 半监督
  * [Adaptive Consistency Regularization for Semi-Supervised Transfer Learning](https://arxiv.org/abs/2103.02193)<br>:star:[code](https://github.com/SHI-Labs/Semi-Supervised-Transfer-Learning)<br>
* 自监督
  * [Self-supervised Geometric Perception](https://arxiv.org/abs/2103.03114)<br>:open_mouth:oral:star:[code](https://github.com/theNded/SGP)<br>作者称 SGP 是第一个在几何感知中进行特征学习的通用框架，不需要任何来自 ground-truth 几何标签的监督。SGP以EM方式运行，它迭代执行几何模型的鲁棒估计以生成伪标签，并在噪声伪标签的监督下进行特征学习。将 SGP 应用于相机姿势估计和点云配准，并证明在大规模真实数据集中，SGP 的性能等同于甚至优于监督的权威。

<a name="3"/> 

## 3.点云

- [Diffusion Probabilistic Models for 3D Point Cloud Generation](https://arxiv.org/abs/2103.01458)<br>:open_mouth:oral:star:[code](https://github.com/luost26/diffusion-point-cloud)<br>
- [Style-based Point Generator with Adversarial Rendering for Point Cloud Completion](https://arxiv.org/abs/2103.02535)<br>
- [MultiBodySync: Multi-Body Segmentation and Motion Estimation via 3D Scan Synchronization](https://arxiv.org/abs/2101.06605)<br>:open_mouth:oral:star:[code](https://github.com/huangjh-pub/multibody-sync)
- [TPCN: Temporal Point Cloud Networks for Motion Forecasting](https://arxiv.org/abs/2103.03067)<br>用于运动预测的时空点云网络<br>
- [PointGuard: Provably Robust 3D Point Cloud Classification](https://arxiv.org/abs/2103.03046)
- [How Privacy-Preserving are Line Clouds? Recovering Scene Details from 3D Lines](https://arxiv.org/abs/2103.05086)<br>:star:[code](https://github.com/kunalchelani/Line2Point)
* 点云配准
  * [PREDATOR: Registration of 3D Point Clouds with Low Overlap](https://arxiv.org/pdf/2011.13005.pdf)<br>:open_mouth:oral:star:[code](https://github.com/ShengyuH/OverlapPredator):house:[project](https://overlappredator.github.io/)<br>
  * [SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration](https://arxiv.org/abs/2011.12149)<br>:star:[code](https://github.com/QingyongHu/SpinNet)
  * [Robust Point Cloud Registration Framework Based on Deep Graph Matching](https://arxiv.org/abs/2103.04256)<br>:star:[code](https://github.com/fukexue/RGM)
  * [PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency](https://arxiv.org/abs/2103.05465)<br>:star:[code](https://github.com/XuyangBai/PointDSC)
* 点云补全
  * [Cycle4Completion: Unpaired Point Cloud Completion using Cycle Transformation with Missing Region Coding](https://arxiv.org/abs/2103.07838)
* 点云关键点检测
  * [Skeleton Merger: an Unsupervised Aligned Keypoint Detector](https://arxiv.org/abs/2103.10814)<br>:star:[code](https://github.com/eliphatfs/SkeletonMerger)


<a name="2"/> 

## 2.图卷积网络GNN

- [Sequential Graph Convolutional Network for Active Learning](https://arxiv.org/pdf/2006.10219.pdf)<br>
- [Quantifying Explainers of Graph Neural Networks in Computational Pathology](https://arxiv.org/abs/2011.12646)<br>
- [Binary Graph Neural Networks](https://arxiv.org/abs/2012.15823)

<a name="1"/> 

## 1.未分类

- Inverting the Inherence of Convolution for Visual Recognition<br>
- Representative Batch Normalization with Feature Calibration<br>
- UC2: Universal Cross-lingual Cross-modal Vision-and-Language Pretraining<br>
- Reconsidering Representation Alignment for Multi-view Clustering<br>
- Self-supervised Simultaneous Multi-Step Prediction of Road Dynamics and Cost Map<br>
- [Instance Localization for Self-supervised Detection Pretraining](https://arxiv.org/pdf/2102.08318.pdf)<br>:star:[code](https://github.com/limbo0000/InstanceLoc)<br>
- Model-Contrastive Federated Learning<br>提出模型对比学习来解决联合学习中的非IID数据问题<br>
- [Neural Geometric Level of Detail:Real-time Rendering with Implicit 3D Surfaces](https://arxiv.org/abs/2101.10994)<br>:open_mouth:Oral:star:[code](https://github.com/nv-tlabs/nglod):house:[project](https://nv-tlabs.github.io/nglod/)<br>
- [Data-Free Model Extraction](https://arxiv.org/abs/2011.14779)<br>:star:[code](https://github.com/cake-lab/datafree-model-extraction)<br>
- Single-Stage Instance Shadow Detection with Bidirectional Relation Learning<br>:open_mouth:oral<br>:star:[code](https://github.com/stevewongv/SSIS)
- [Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning](https://arxiv.org/abs/2012.04324)<br>:open_mouth:oral
- [PatchmatchNet: Learned Multi-View Patchmatch Stereo](https://arxiv.org/abs/2012.01411)<br>:open_mouth:oral:star:[code](https://github.com/FangjinhuaWang/PatchmatchNet)
- [Online Bag-of-Visual-Words Generation for Unsupervised Representation Learning]
- [Semantic Palette: Guiding Scene Generation with Class Proportions]
- Function4D: Real-time Human Volumetric Capture from Very Sparse Consumer RGBD Sensors<br>:open_mouth:oral
- POSEFusion:Pose-guided Selective Fusion for Single-view Human Volumetric Capture<br>:open_mouth:oral
- [Multi-Objective Interpolation Training for Robustness to Label Noise](https://arxiv.org/abs/2012.04462)<br>:star:[code](https://github.com/DiegoOrtego/LabelNoiseMOIT)
- [Right for the Right Concept: Revising Neuro-Symbolic Concepts by Interacting with their Explanations](https://arxiv.org/abs/2011.12854)<br>:star:[code](https://github.com/ml-research/NeSyXIL)
- Simpler Certified Radius Maximization by Propagating Covariances<br>:open_mouth:oral:tv:[video](https://www.youtube.com/watch?v=1V9sBzlfuwY)
- [Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food](https://arxiv.org/abs/2103.03375)
- [Discovering Hidden Physics Behind Transport Dynamics](https://arxiv.org/abs/2011.12222)<br>:open_mouth:oral
- [Soft-IntroVAE: Analyzing and Improving the Introspective Variational Autoencoder](https://arxiv.org/abs/2012.13253)<br>:open_mouth:oral:star:[code](https://github.com/taldatech/soft-intro-vae-pytorch):house:[project](https://taldatech.github.io/soft-intro-vae-web/)
- [Deep Gradient Projection Networks for Pan-sharpening](https://arxiv.org/abs/2103.04584)<br>:star:[code](https://github.com/xsxjtu/GPPNN)
- [Consensus Maximisation Using Influences of Monotone Boolean Functions](https://arxiv.org/abs/2103.04200)<br>:open_mouth:oral
* Forecasting Irreversible Disease via Progression Learning
* Causal Hidden Markov Model for Time Series Disease Forecasting
* Towards Unified Surgical Skill Assessment
- [Knowledge Evolution in Neural Networks](https://arxiv.org/abs/2103.05152)<br>:open_mouth:oral:star:[code](https://github.com/ahmdtaha/knowledge_evolution)
* RSTNet: Captioning with Adaptive Attention on Visual and Non-Visual Words<br>RSTNet: 基于可区分视觉词和非视觉词的自适应注意力机制的图像描述生成模型<br>解读：[14](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
* [Removing the Background by Adding the Background: Towards a Background Robust Self-supervised Video Representation Learning](https://arxiv.org/abs/2009.05769)<br>通过添加背景来去除背景影响：背景鲁棒的自监督视频表征学习<br>解读：[11](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
* Representative Batch Normalization with Feature Calibration<br>:open_mouth:oral<br>[作者主页](https://duoli.org/)<br>基于特征校准的表征批规范化方法解读：[4](https://mp.weixin.qq.com/s/yNDkHMhOIb76b4KcEhx4XQ)
* Learning Compositional Representation for 4D Captures with Neural ODE
* [Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/abs/2103.06255)<br>:star:[code](https://github.com/d-li14/involution)<br>解读：[CVPR'21 | involution：超越convolution和self-attention的神经网络新算子](https://mp.weixin.qq.com/s/Kn7QJdldLhyBfYS1KiCdcA)
* [Spatially Consistent Representation Learning](https://arxiv.org/abs/2103.06122)
* [Limitations of Post-Hoc Feature Alignment for Robustness](https://arxiv.org/abs/2103.05898)
* [AutoDO: Robust AutoAugment for Biased Data with Label Noise via Scalable Probabilistic Implicit Differentiation](https://arxiv.org/abs/2103.05863)<br>:star:[code](https://github.com/gudovskiy/autodo)
* [Abstract Spatial-Temporal Reasoning via Probabilistic Abduction and Execution](http://wellyzhang.github.io/attach/cvpr21zhang_prae.pdf)<br>:star:[code](https://github.com/WellyZhang/PrAE)
* CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching<br>:star:[code](https://github.com/gallenszl/CFNet)
* [Augmentation Strategies for Learning with Noisy Labels](https://arxiv.org/abs/2103.02130)<br>:star:[code](https://github.com/KentoNishi/Augmentation-for-LNL)
* [Abstract Spatial-Temporal Reasoning via Probabilistic Abduction and Execution](http://wellyzhang.github.io/attach/cvpr21zhang_prae.pdf)<br>:star:[code](https://github.com/WellyZhang/PrAE)
* CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching<br>:star:[code](https://github.com/gallenszl/CFNet)
* [Augmentation Strategies for Learning with Noisy Labels](https://arxiv.org/abs/2103.02130)<br>:star:[code](https://github.com/KentoNishi/Augmentation-for-LNL)
* [PGT: A Progressive Method for Training Models on Long Videos](https://arxiv.org/abs/2103.11313)<br>:open_mouth:oral:star:[code](https://github.com/BoPang1996/PGT)
* [Generic Perceptual Loss for Modeling Structured Output Dependencies](https://arxiv.org/abs/2103.10571)

<a name="*"/>

## Workshop 征稿ing

- [Visual Perception for Navigation in Human Environments](https://jrdb.stanford.edu/workshops/jrdb-cvpr21)<br>第二届人类环境导航视觉感知征稿 :warning:4月15截止
- [UG 2 + Challenge](http://cvpr2021.ug2challenge.org/index.html)<br>旨在通过应用图像恢复和增强算法提高分析性能，推动对 "difficult"图像的分析。参与者任务是开发新的算法，以改进对在问题条件下拍摄的图像分析。<br>:crown:10K美元奖金<br>
   * [低能见度环境下的目标检测](https://www.deepl.com/translator#en/zh/OBJECT%20DETECTION%20IN%20POOR%20VISIBILITY%20ENVIRONMENTS)
      * 雾霾条件下的(半)监督目标检测
      * (半)低光条件下的人脸检测
   * [黑暗视频中的动作识别](http://cvpr2021.ug2challenge.org/track2.html)
      * 黑暗中进行完全监督动作识别
      * 黑暗中进行半监督动作识别

- [Continual Learning in Computer Vision 征稿中](https://sites.google.com/view/clvision2021/overview?authuser=0)<br>旨在聚集学术界和工业界的研究人员和工程师，讨论持续学习的最新进展。<br>
  * Best paper award: 					500 USD + 500 USD worth of Huawei cloud credits (HUAWEI)
  * Overall Challenge winner: 				1,000 USD  + 500 USD worth of Huawei cloud credits (HUAWEI)
  * Supervised-Learning track winner: 		500 USD (HUAWEI)
  * Reinforcement-Learning track winner: 	500 USD (ServiceNow)

- [第四届UG2研讨会和竞赛：弥合计算成像与视觉识别之间的鸿沟](https://mp.weixin.qq.com/s/u7YDi7tAsDn77P0hb-dmFg)

- [10万美元奖金！CVPR 2021 重磅赛事，安全AI挑战者计划](https://mp.weixin.qq.com/s/1d8RL9_YqKp9stVf2M1N-Q)
  * [CVPR 2021大赛， 安全AI 之防御模型的「白盒对抗攻击」解析](https://mp.weixin.qq.com/s/OJ9_xaQ6GyuCUrph8boxYA)
  * [还在刷榜ImageNet？找出模型的脆弱之处更有价值！](https://mp.weixin.qq.com/s/nbAudiGJX_l69zaSEiWJFA)

- [Responsible Computer Vision](https://sites.google.com/view/rcv-cvpr2021/home)<br>:warning:3月25日截止<br>本次研讨会将广泛讨论计算机视觉背景下负责任的人工智能的三个主要方面：公平性；可解释性和透明度；以及隐私。
- [Holistic Video Understanding](https://holistic-video-understanding.github.io/workshops/cvpr2021.html)<br>目的是建立一个整合所有语义概念联合识别的视频基准，因为每个任务的单一类标签往往不足以描述视频的整体内容。
- [ThreeDWorld Transport Challenge](http://tdw-transport.csail.mit.edu/)<br>:warning:6月1截止<br>:tv:[video](https://www.youtube.com/watch?v=WcJTpiRC4Zo)
- [FGVC 8](https://sites.google.com/view/fgvc8)<br>第八届细粒度视觉分类研讨会（FGVC8）将通过细粒度视觉理解的视角，探讨细粒度学习、自监督学习、半监督学习、matching(匹配)、localization(定位)、域适应、迁移学习、小样本学习、机器教学、多模态学习（如音频和视频）、众包和分类学预测等相关话题。<br>:warning:[论文截稿日期](https://sites.google.com/view/fgvc8/submission)为4月2日<br>[征稿主题包含以下几个方面](https://sites.google.com/view/fgvc8/submission)
  * Fine-grained categorization细粒度分类
    * Novel datasets and data collection strategies for fine-grained categorization用于细粒度分类的新型数据集和数据收集策略
    * Appropriate error metrics for fine-grained categorization细粒度分类的适当误差指标
    * Low/few shot learning少/小样本学习
    * Self-supervised learning自监督学习
    * Semi-supervised learning半监督学习
    * Transfer-learning from known to novel subcategories
    * Attribute and part based approaches
    * Taxonomic predictions
    * Addressing long-tailed distributions
  * Human-in-the-loop
    * Fine-grained categorization with humans in the loop
    * Embedding human experts’ knowledge into computational models
    * Machine teaching
    * Interpretable fine-grained models
  * Multi-modal learning
    * Using audio and video data
    * Using geographical priors
    * Learning shape
  * Fine-grained applications
    * Product recognition 
    * Animal biometrics and camera traps 
    * Museum collections
    * Agricultural 
    * Medical 
    * Fashion
  * 相关挑战赛如下（部分已在Kaggle网站开始）
    * [GeoLifeCLEF2021](https://www.kaggle.com/c/geolifeclef-2021)<br>利用观测结果与航空图像和环境特征配对，预测物种的存在
    * [Semi-iNat2021](https://www.kaggle.com/c/semi-inat-2021)<br>由iNaturalist的数据组成的半监督细粒度图像分类
    * [iNatChallenge2021](https://www.kaggle.com/c/inaturalist-2021)<br>对1万类动植物进行图像分类挑战赛
    * [iMet2021](https://www.kaggle.com/c/imet-2021-fgvc8)<br>对艺术品进行细粒度属性分类
    * [iMat-Fashion2021](未开始)未开始<br>服装实例分割和细粒度属性分类
    * [Hotel-ID 2021](https://www.kaggle.com/c/hotel-id-2021-fgvc8)<br>从图像中识别酒店房间
    * [HerbariumChallenge2021](https://www.kaggle.com/c/herbarium-2021-fgvc8)<br>从数据集中识别标本，该数据集包含来自美洲、大洋洲和太平洋地区的近66,000种 vascular plant species（维管束植物）的 2.5M 图像
    * [iWildCam2021](https://www.kaggle.com/c/iwildcam2021-fgvc8)<br>对图像序列中每个物种的动物数量计数
    * [PlantPathologyChallenge2021](未开始)未开始<br>对病害植物的图像进行分类
    


## 扫码CV君微信（注明：CVPR）入微信交流群：

![image](https://user-images.githubusercontent.com/62801906/109789529-655e4380-7c4b-11eb-9f1a-58c5cb097428.png)
