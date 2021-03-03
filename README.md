# CVPR2021最新信息及已接受论文/代码(持续更新)


本贴是对 CVPR2021 已接受论文的粗略汇总，后期会有更详细的总结。期待ing......

官网链接：http://cvpr2021.thecvf.com<br>
时间：2021年6月19日-6月25日<br>
论文接收公布时间：2021年2月28日<br>

相关问题：<br>

* [CVPR 2021 接收论文列表！27%接受率！](https://zhuanlan.zhihu.com/p/353686917)


# 目录

|:cat:|:dog:|:mouse:|:hamster:|:tiger:|
|----|-----|----|-----|-----|
|----|-----|----|-----|[* Workshop征稿](#*)|
|[30.新视角合成](#30)|[29.姿态估计](#29)|[28.密集预测](#28)|[27.活体检测](#27)|[26.视频编解码](#26)|
|[25.三维视觉](#25)|[24.强化学习](#24)|[23.自动驾驶](#23)|[22.医学影像](#22)|[21.Transformer](#21)|
|[20.人员重识别](#20)|[19.模型压缩](#19)|[18.航空影像](#18)|[17.超分辨率](#17)|[16.视觉问答](#16)|
|[15.GAN](#15)|[14.少样本学习](#14)|[13.图像检索](#13)|[12.图像增强](#12)|[11.人脸技术](#11)|
|[10.神经架构搜索](#10)|[9.目标跟踪](#9)|[8.图像分割](#8)|[7.目标检测](#7)|[6.数据增强](#6)|
|[5.异常检测](#5)|[4.弱监督学习](#4)|[3.点云](#3)|[2.图卷积网络GNN](#2)|[1.未分类](#1)|

<a name="30"/>

## 30.新视角合成

- [DeRF: Decomposed Radiance Fields](https://arxiv.org/abs/2011.12490)<br>:house:[project](https://ubc-vision.github.io/derf/)<br>
- [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961)<br>:house:[project](https://www.albertpumarola.com/research/D-NeRF/index.html)<br>

<a name="29"/>

## 29.姿态估计

- [PCLs: Geometry-aware Neural Reconstruction of 3D Pose with Perspective Crop Layers](https://arxiv.org/abs/2011.13607)<br>:tv:[video](https://twitter.com/i/status/1334395954644930560)<br>通过消除 location-dependent 透视效果来改进3D人体姿势估计技术工作。<br>
- [CanonPose: Self-supervised Monocular 3D Human Pose Estimation in the Wild](https://arxiv.org/abs/2011.14679)

<a name="28"/>

## 28.密集预测

- [Densely connected multidilated convolutional networks for dense prediction tasks](https://arxiv.org/abs/2011.11844)<br>提出的D3Net在语义分割&音乐源分离任务上的表现优于SOTA网络<br>

<a name="27"/>

## 27.活体检测

- [Cross Modal Focal Loss for RGBD Face Anti-Spoofing](https://arxiv.org/abs/2103.00948)<br>

<a name="26"/>

## 26.视频编解码

- [MetaSCI: Scalable and Adaptive Reconstruction for Video Compressive Sensing](https://arxiv.org/abs/2103.01786)<br>:star:[code](https://github.com/xyvirtualgroup/MetaSCI-CVPR2021)

<a name="25"/>

## 25.三维视觉

- [A Deep Emulator for Secondary Motion of 3D Characters](https://arxiv.org/abs/2103.01261)<br>

<a name="24"/> 

## 24.强化学习
- [Hierarchical and Partially Observable Goal-driven Policy Learning with Goals Relational Graph](https://arxiv.org/abs/2103.01350)<br>
<a name="23"/> 

## 23.自动驾驶

- [Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition](https://arxiv.org/abs/2103.01486)<br>:star:[github](https://github.com/QVPR/Patch-NetVLAD)

<a name="22"/> 

## 22.医学影像

- [3D Graph Anatomy Geometry-Integrated Network for Pancreatic Mass Segmentation, Diagnosis, and Quantitative Patient Management](https://arxiv.org/abs/2012.04701)<br>用纯多模态 CT 影像可替代目前 JHMI 的需要做肿瘤化学检测和 DNA 测序+医学影像的综合多模态诊断流程，从诊断准确度上有可比较性，定量诊断精度更优<br>
- [Deep Lesion Tracker: Monitoring Lesions in 4D Longitudinal Imaging Studies](https://arxiv.org/abs/2012.04872)<br>肿瘤影像里面智能 PACS 辅助医生读片的重要功能<br>
- [Automatic Vertebra Localization and Identification in CT by Spine Rectification and Anatomically-constrained Optimization](https://arxiv.org/abs/2012.07947)<br>基于CT 影像的骨折/骨质疏松系统<br>

<a name="21"/> 

## 21.Transformer 

- [Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/pdf/2012.09838.pdf)<br>:star:[code](https://github.com/hila-chefer/Transformer-Explainability)<br>
- [UP-DETR: Unsupervised Pre-training for Object Detection with Transformers](https://arxiv.org/pdf/2011.09094.pdf)<br>
- [Pre-Trained Image Processing Transformer](https://arxiv.org/pdf/2012.00364.pdf)<br>
- 3D Vision Transformers for Action Recognition<br>用于动作识别的3D视觉Transformer
- [MIST: Multiple Instance Spatial Transformer Network](https://arxiv.org/abs/1811.10725)<br>试图从热图中进行可微的top-K选择(MIST)（目前在自然图像上也有了一些结果；) 用它可以在没有任何定位监督的情况下进行检测和分类（并不是它唯一能做的事情!）



<a name="20"/> 

## 20.人员重识别

- [39.Meta Batch-Instance Normalization for Generalizable Person Re-Identification](https://arxiv.org/abs/2011.14670)<br>

<a name="19"/> 


## 19.模型压缩

- [Learning Student Networks in the Wild](https://arxiv.org/pdf/1904.01186.pdf)<br>:star:[code](https://github.com/huawei-noah/DAFL)<br>
- [Rethinking Channel Dimensions for Efficient Model Design](https://arxiv.org/abs/2007.00992)<br>:star:[code](https://github.com/clovaai/rexnet)<br>
- Manifold Regularized Dynamic Network Pruning（动态剪枝的过程中考虑样本复杂度与网络复杂度的约束）<br>

<a name="18"/> 

## 18.航空影像

- Dogfight: Detecting Drones from Drone Videos（从无人机视频中检测无人机）<br>
- PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation(语义流经点以进行航空图像分割)<br>

<a name="17"/> 

## 17.超分辨率

- Data-Free Knowledge Distillation For Image Super-Resolution(DAFL算法的SR版本)<br>
- [AdderSR: Towards Energy Efficient Image Super-Resolution](https://arxiv.org/pdf/2009.08891.pdf)<br>:star:[code](https://github.com/huawei-noah/AdderNet)<br>

<a name="16"/> 


## 16.视觉问答

- Weakly-supervised Grounded Visual Question Answering using Capsules<br>


<a name="14"/> 

## 15.GAN
- Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing<br>
- [Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation](https://arxiv.org/abs/2008.00951)<br>:star:[code](https://github.com/eladrich/pixel2style2pixel):house:[project](https://eladrich.github.io/pixel2style2pixel/)<br>
- [Hijack-GAN: Unintended-Use of Pretrained, Black-Box GANs](https://arxiv.org/pdf/2011.14107.pdf)<br>
- [Image-to-image Translation via Hierarchical Style Disentanglement](https://arxiv.org/abs/2103.01456)<br>:star:[code](https://github.com/imlixinyang/HiSD)

<a name="14"/> 


## 14.少样本学习

- Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning(探索少量学习的不变表示形式和等变表示形式的互补强度)<br>
- [Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning](https://arxiv.org/abs/2103.01315)<br>

<a name="11"/> 


## 13.图像检索

- [Probabilistic Embeddings for Cross-Modal Retrieval](https://arxiv.org/abs/2101.05068)<br>

<a name="12"/> 

## 12.图像增强

- [Multi-Stage Progressive Image Restoration](https://arxiv.org/abs/2102.02808)<br>:star:[code](https://github.com/swz30/MPRNet)<br>
- [Auto-Exposure Fusion for Single-Image Shadow Removal](https://arxiv.org/abs/2103.01255)<br>:star:[code](https://github.com/tsingqguo/exposure-fusion-shadow-removal)<br>

<a name="11"/> 

## 11. 人脸技术

- [A 3D GAN for Improved Large-pose Facial Recognition](https://arxiv.org/pdf/2012.10545.pdf)<br>
- [When Age-Invariant Face Recognition Meets Face Age Synthesis: A Multi-Task Learning Framework](https://arxiv.org/abs/2103.01520)<br>:star:[github](https://github.com/Hzzone/MTLFace)<br>


<a name="10"/> 

## 10.神经架构搜索

- [AttentiveNAS: Improving Neural Architecture Search via Attentive](https://arxiv.org/pdf/2011.09011.pdf)<br>
- [HourNAS: Extremely Fast Neural Architecture Search Through an Hourglass Lens](https://arxiv.org/pdf/2005.14446.pdf)<br>
- [ReNAS: Relativistic Evaluation of Neural Architecture Search](https://arxiv.org/pdf/1910.01523.pdf)<br>


<a name="9"/> 

## 9.目标跟踪
- [Probabilistic Tracklet Scoring and Inpainting for Multiple Object Tracking](https://arxiv.org/abs/2012.02337)<br>
- [Rotation Equivariant Siamese Networks for Tracking](https://arxiv.org/abs/2012.13078)<br>


<a name="8"/> 

## 8.图像分割

- [4D Panoptic LiDAR Segmentation](https://arxiv.org/abs/2102.12472)<br>
- [PLOP: Learning without Forgetting for Continual Semantic Segmentation](https://arxiv.org/abs/2011.11390)<br>



<a name="7"/> 

## 7.目标检测

- [Multiple Instance Active Learning for Object Detection](https://github.com/yuantn/MIAL/raw/master/paper.pdf)<br>:star:[code](https://github.com/yuantn/MIAL)<br>
- Positive-Unlabeled Data Purification in the Wild for Object Detection<br>
- Open-world object detection(开放世界中的目标检测)<br>:star:[code](https://github.com/JosephKJ/OWOD)<br>
- [Depth from Camera Motion and Object Detection](https://arxiv.org/abs/2103.01468)<br>:star:[github](https://github.com/griffbr/ODMD):tv:[video](https://www.youtube.com/watch?v=GruhbdJ2l7k)
- [There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge](https://arxiv.org/abs/2103.01353)<br>:house:[project](https://rl.uni-freiburg.de/)<br>
- [Categorical Depth Distribution Network for Monocular 3D Object Detection](https://arxiv.org/abs/2103.01100)<br>

<a name="6"/> 

## 6.数据增强

- [KeepAugment: A Simple Information-Preserving Data Augmentation](https://arxiv.org/pdf/2011.11778.pdf)<br>

<a name="5"/> 

## 5.异常检测

- [Multiresolution Knowledge Distillation for Anomaly Detection](https://arxiv.org/abs/2011.11108)<br>

<a name="4"/> 

## 4.弱监督学习

- [Weakly Supervised Learning of Rigid 3D Scene Flow](https://arxiv.org/pdf/2102.08945.pdf)<br>:star:[code](https://arxiv.org/pdf/2102.08945.pdf):house:[project](https://3dsceneflow.github.io/)<br>

<a name="3"/> 

## 3.点云

- [PREDATOR: Registration of 3D Point Clouds with Low Overlap](https://arxiv.org/pdf/2011.13005.pdf)<br>:star:[code](https://github.com/ShengyuH/OverlapPredator):house:[project](https://overlappredator.github.io/)<br>
- [Diffusion Probabilistic Models for 3D Point Cloud Generation](https://arxiv.org/abs/2103.01458)<br>:star:[code](https://github.com/luost26/diffusion-point-cloud)<br>

<a name="2"/> 

## 2.图卷积网络GNN

- [Sequential Graph Convolutional Network for Active Learning](https://arxiv.org/pdf/2006.10219.pdf)<br>

<a name="1"/> 

## 1.未分类

- [Improving Unsupervised Image Clustering With Robust Learning](https://arxiv.org/abs/2012.11150)<br>:star:[code](https://github.com/deu30303/RUC)<br>利用鲁棒学习改进无监督图像聚类技术<br>
- [Coarse-Fine Networks for Temporal Activity Detection in Videos](https://arxiv.org/abs/2103.01302)(用于视频中的时间活动检测的粗细网络)<br>
- [Instance Localization for Self-supervised Detection Pretraining](https://arxiv.org/pdf/2102.08318.pdf)<br>:star:[code](https://github.com/limbo0000/InstanceLoc)<br>
- [FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation](https://arxiv.org/pdf/2012.08512.pdf)<br>:star:[code](https://tarun005.github.io/FLAVR/Code):house:[project](https://tarun005.github.io/FLAVR/)<br>
- [Re-labeling ImageNet: from Single to Multi-Labels, from Global to Localized Labels](https://arxiv.org/abs/2101.05022)<br>:star:[code](https://github.com/naver-ai/relabel_imagenet)<br>
- Rainbow Memory: Continual Learning with a Memory of Diverse Samples（不断学习与多样本的记忆）<br>
- Reconsidering Representation Alignment for Multi-view Clustering(重新考虑多视图聚类的表示对齐方式)<br>
- Self-supervised Simultaneous Multi-Step Prediction of Road Dynamics and Cost Map(道路动力学和成本图的自监督式多步同时预测)<br>
- [IIRC: Incremental Implicitly-Refined Classification](https://arxiv.org/abs/2012.12477)<br>:house:[project](https://chandar-lab.github.io/IIRC/)<br>
- [Fair Attribute Classification through Latent Space De-biasing](https://arxiv.org/abs/2012.01469)<br>:star:[code](https://github.com/princetonvisualai/gan-debiasing):house:[project](https://princetonvisualai.github.io/gan-debiasing/)<br>
- [Information-Theoretic Segmentation by Inpainting Error Maximization](https://arxiv.org/abs/2012.07287)<br>
- Few-shot Open-set Recognition by Transformation Consistency(转换一致性很少的开放集识别)<br>
- UC2: Universal Cross-lingual Cross-modal Vision-and-Language Pretraining(UC2：通用跨语言跨模态视觉和语言预培训)<br>
- [Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling](https://arxiv.org/pdf/2102.06183.pdf)<br>:star:[code](https://github.com/jayleicn/ClipBERT)<br>
- [3D CNNs with Adaptive Temporal Feature Resolutions](https://arxiv.org/abs/2011.08652)<br>
- Distilling Object Detectors via Decoupled Features（前景背景分离的蒸馏技术） <br>
- Inverting the Inherence of Convolution for Visual Recognition（颠倒卷积的固有性以进行视觉识别）<br>
- Representative Batch Normalization with Feature Calibration（具有特征校准功能的代表性批量归一化）<br>
- Learning the Superpixel in a Non-iterative and Lifelong Manner(以非迭代和终身的方式学习超像素)<br>
- [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)<br>:star:[code](https://github.com/megvii-model/RepVGG)<br>
- [Counterfactual Zero-Shot and Open-Set Visual Recognition](https://arxiv.org/abs/2103.00887)<br>:star:[code](https://github.com/yue-zhongqi/gcm-cf)<br>
- [VirTex: Learning Visual Representations from Textual Annotations](https://arxiv.org/abs/2006.06666)<br>:star:[code](https://github.com/kdexd/virtex)
- [Learning the Predictability of the Future](https://arxiv.org/abs/2101.01600)<br>预测未来<br>:star:[code](https://github.com/cvlab-columbia/hyperfuture):house:[project](https://hyperfuture.cs.columbia.edu/):tv:[video](https://www.youtube.com/watch?v=-Uy92jvT_90)<br>
- [Vab-AL: Incorporating Class Imbalance and Difficulty with Variational Bayes for Active Learning]
- [Domain Generalization via Inference-time Label-Preserving Target Projections](https://arxiv.org/abs/2103.01134)<br>

<a name="*"/>

## Workshop 征稿

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
