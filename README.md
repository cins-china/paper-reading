# Paper-Reading

项目地址：https://github.com/cins-china/paper-reading

团队介绍：该项目主要是组内在研读顶会论文过程中，所见、所思、所想、所闻，希望组内思想碰撞以出现新的Idea。

## Notes:

CCF会议的截止日期: https://ccfddl.github.io/

## Group Member

- 张巍 	weizhang-git-zw(https://github.com/weizhang-git-zw)
- 杨洋	yeung2y(https://github.com/yeung2y)
- 顾天一
- 黄金城	huangJC0429(https://github.com/huangJC0429)
- 唐国根	Teigenm(https://github.com/Teigenm)
- 李丽	Lily029106(https://github.com/Lily029106)
- 陈娜	nachen-na(https://github.com/nachen-na)
- 黄恺文
- 李国伟 	lgw9527111(https://github.com/lgw9527111)
- 胡诗琪 	Nancy206-hu(https://github.com/Nancy206-hu)
- 李攀	bert-ply(https://github.com/bert-ply)
- 张阿聪	zac-hub(https://github.com/zac-hub)



## Details
[:arrow_up:](#table-of-contents)

#### 2021.08.30
- 【关于IMP-GCN】研讨

  - 论文: Interest-aware Message-Passing GCN for Recommendation

  - 论文地址: https://arxiv.org/pdf/2102.10044.pdf

  - 动机: 旨在解决GCN过平滑问题

  - 论文方法:

    - (1) 增加一个子图生成模块，对用户节点进行分组，使兴趣相似的用户为一组；

    - (2) 一阶传播作用于整个图，高阶传播作用于子图内部，最后将每一层加权求和得到嵌入。

      

- 【关于医学图像分类 ChestX-ray8】研讨
  - 论文: ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases
  - 论文地址: https://arxiv.org/pdf/1705.02315.pdf
  - 动机: 旨在更好地进行医学图像诊断分类
  - 论文方法:
    - (1) 将NLP和DCNN结合起来对病理进行诊断和定位；
    - (2) 移除了CNN中的全连接层，加入了过渡层、全局池化层、预测层和损失层。

#### 2021.09.06

- 【关于BERT模型基于E2E-ABSA】研讨

  -  论文：Exploiting BERT for End-to-End Aspect-based Sentiment Analysis
  -  论文地址：https://arxiv.org/pdf/1910.00883.pdf
  -  动机：旨在BERT-based的模型在很大程度上减了上下文独立性的问题
  - 论文方法：
    -  (1) BERT作为嵌入层为每个token提供一个上下文无关的表示形式；
    -  (2) 探讨了E2E-ABSA的几种不同设计：Linear Layer、RNN、SAN、CRF的性能。

- 【关于分型网络】研讨

  - 论文：https://github.com/cins-china/paper-reading
  - 论文地址：https://arxiv.org/abs/1605.07648v1
  - 动机：旨在更好的构建网络
  - 论文方法：
    - （1）利用新的学习机制drop-path随机抛弃路径。采取局部和全局的两种方式，即Local：在每一个联合层以固定几率舍弃每个输入，但保证至少保留一个输入。 Global：在整个网络中随机只保留一条完整的路径进行学习。 实现更好的网路模型。

  
#### 2021.09.27

- 【关于自动获取注意监督信息】研讨
   
   - 论文：Progressive Self-Supervised Attention Learning for Aspect-Level Sentiment Analysis
   - 论文地址：https://arxiv.org/abs/1906.01213
   - 动机：旨在自动从训练语料库中挖掘有用的注意监督信息，以细化注意机制。
   - 论文方法：
     - （1）基本事实：具有最大值注意力权重的上下文词对输入句的情感预测有最大的影响。
     - （2）以一种渐进的自监督注意学习方法，自动从训练语料库中挖掘有用的注意监督信息，从而在模型训练过程中约束注意力机制的学习。
     - （3）通过迭代训练MN、TNet神经模型获得监督信息。

#### 2021.10.11

- 【关于时空注意力机制】研讨
  
  - 论文：CBAM: Convolutional Block Attention Module
   - 论文地址：https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
   - 动机：利用时空注意力机制增强CNN模型的表征能力。
   - 论文方法：
     - （1）提出卷积注意模块(CBAM)，从通道和空间两个维度计算feature map的attention map。
     - （2）将attention map与输入的feature map相乘来进行特征的自适应学习。

-  【关于图推荐综述】研讨
    
    - 论文：Graph Neural Networks for Recommender Systems:Challenges, Methods, and Directions
    - 论文地址：https://arxiv.org/abs/2109.12843
    - 动机：提供一个系统和全面的研究工作。
    - 论文方法：
      - （1）从阶段、场景、目标和应用四个方面对推荐系统的研究进行了分类。
      - （2) 特别是如何改进图神经网络的推荐和解决相应的挑战。

#### 2021.10.25

- 【关于自适应多通道图机制】研讨
    
    - 论文：AM-GCN: Adaptive Multi-channel Graph Convolutional
    - 论文地址：https://dl.acm.org/doi/abs/10.1145/3394486.3403177
    - 动机：GCN从拓扑结构和节点特征中学习并融合了哪些信息。
    - 论文方法：
      - （1）通过节点特征利用余弦相似性构建k最近邻居拓扑图。
      - （2) 将特征拓扑图和结构拓扑图分别卷积和共享卷积得到节点特征。
      - （3）对得出来的特征学习权重。

-  【关于多标签图像分类问题】研讨
    
    - 论文：Residual Attention A Simple but Effective Method for Multi-Label Recognition
    - 论文地址：https://arxiv.org/pdf/2108.02456.pdf
    - 动机：利用特征的空间注意力得分提升多标签图像分类效果。
    - 论文方法：
      - （1）提出残差注意力模块(CSRA)，结合spatial pooling和average pooling来计算每个类别在不同空间位置上出现的概率；
      - （2) 引入了multi-head attention的方式来避免公式中参数调参。

#### 2021.11.01

- 【关于分型网络】研讨
   
   - 论文：A Hierarchical Graph Network for 3D Object Detection on Point Clouds
   - 论文地址：https://www.aminer.cn/pub/5eccb534e06a4c1b26a834c7?conf=cvpr2020
   - 动机：更好的去做3D目标检测
   - 论文方法：
     - （1）开发了一种新的层级图网络（HGNet），用于在点云上进行 3D 对象检测，其表现好于已有方法。
     - （2）提出了一种新颖的 SA-（De）GConv，它可以有效地聚合特征并捕获点云中对象的形状信息。
     - （3）构建了一个新的 GU-net，用于生成多级特征，这对于 3D 对象检测至关重要。
     - （4）构建了一个新的 GU-net，用于生成多级特征，这对于 3D 对象检测至关重要。

- 【关于方面级情感三元组提取】研讨
   
   - 论文：Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction
   - 论文地址：https://arxiv.org/pdf/2107.12214.pdf
   - 动机：提出一个基于跨度的ASTE模型来解决三元组抽取这个任务
   - 论文方法：
     - （1）这篇论文提出一种span-level方法Span-ASTE来学习target spans 和opinion spans之间的交互作用。
     - （2）提出了一种双通道跨度剪枝策略，结合了来自ATE和OTE任务的显式监督，以减轻跨度枚举带来的高计算成本。
