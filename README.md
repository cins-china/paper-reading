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

#### 2021.11.08

- 【关于分离句内和句间文档级关系推理】研讨
   
   - 论文：SIRE Separate Intra- and Inter-sentential Reasoning for Document-level Relation Extraction
   - 论文地址：https://arxiv.org/pdf/2106.01709.pdf
   - 分享汇报：唐国根
   - 动机：现有的图方法视该问题为一个表征学习问题，使用实体作为节点构造文档图，并不是所有的实体对都有正确的逻辑推理路径 
   - 论文方法：
     - （1）分别表示句子内和句子间的关系  
     - （2）一种新的逻辑推理形式，尽可能涵盖了逻辑推理链的所有案例  

#### 2021.11.15

- 【关于方面级情感分析中语法依赖】研讨
   
   - 论文：Relational Graph Attention Network for Aspect-based Sentiment Analysis
   - 论文地址：https://arxiv.org/pdf/2004.12362.pdf
   - 分享汇报：李丽
   - 动机：构建一种基于方面的依存树并通过Relational -GCN，来解决依赖关系不同和多个方面之间互相影响的问题。
   - 论文方法：
     - （1）提出一种新的基于方面的依存树，给每一个方面一个唯一的依存树，解决多个方面之间相互影响的问题；
     - （2）提出一种Relational -GCN模型，对不同的依赖关系进行嵌入并基于不同的注意权重，解决依赖关系不同对任务贡献不同的问题。


#### 2021.11.22

- 【关于置信度的GCN】研讨
   
   - 论文：Confidence-based Graph Convolutional Networks for Semi-Supervised Learning
   - 论文地址：https://arxiv.org/abs/1901.08255
   - 分享汇报：黄睿
   - 动机：弥补GCN背景下的置信度估计这一空白。
   - 论文方法：
     - （1）联合估计标签分数及其置信度；
     - （2）使用估计的置信度来确定一个节点在邻域聚合期间对另一个节点的影响，使得模型邻居聚合时各向异性(anisotropic)。


#### 2021.12.01

- 【关于GCN中过平滑和异质性问题】研讨
   
   - 论文：Two Sides of the Same Coin Heterophily and Oversmoothing in Graph Convolutional Neural Networks
   - 论文地址：https://arxiv.org/abs/2102.06462v5
   - 分享汇报：张阿聪
   - 动机：从统一的角度分析过平滑和异质性问题， 找到并解决造成的原因。
   - 论文方法：
     - （1）通过理论推导和实验分析，在同质性和低节点度下会造成伪异质性来解释过平滑；
     - （2）提出信号消息分别对有效信息和无效信息进行传递，提出放缩度的方法，对低度节点的度修正。
   
 - 【关于动态社交网络神经建模】研讨
   
   - 论文：Neural Modeling of Behavioral Patterns in Dynamic Social Interaction Networks
   - 论文地址：https://www-cs.stanford.edu/people/jure/pubs/tedic-www21.pdf
   - 分享汇报：李攀
   - 动机：从动态社会交互网络中挖掘指示性特征用来预测人的社交行为。
   - 论文方法：
     - （1）通过网络扩散过程来学习人们对各种社会任务的交互作用；
     - （2）通过集合池收集随机分布在整个长时间空间中的局部模式。
  
#### 2021.12.10

- 【关于GCN中过平滑和异质性问题】研讨
   
   - 论文： Graph Attention Collaborative Similarity Embedding for Recommender System
   - 论文地址：https://arxiv.org/abs/2102.03135
   - 分享汇报：赵晋松
   - 动机：从邻居权重以及损失函数上进行改进。
   - 论文方法：
     - （1）采用注意力机制对每个节点的不同邻居进行打分作为权重；
     - （2）提出一种自适应裕度的BPR损失再加上用户之间物品之间的相似性作为辅助损失函数。
 
- 【关于通道注意力和组卷积】研讨
   
   - 论文： ResNeSt: Split-Attention Networks
   - 论文地址：https://hangzhang.org/files/resnest.pdf
   - 分享汇报：胡诗琪
   - 动机：结合了通道注意力和分组卷积来提升网络性能。
   - 论文方法：
     - （1）ResNeSt借鉴了SE-Net和SK-Net的思想；
     - （2）提出ResNeSt Block，结合了通道注意力和分组卷积。
#### 2021.12.17

- 【关于分型网络】研讨
   
   - 论文： Local Migration Model of lmages Based onDeep Learning against Adversarial Attacks
   - 论文地址：https://www.ieee.org/publications/rights/index.html
   - 分享汇报：李国伟
   - 动机：通过利用风格迁移攻击图像优化模型。
   - 论文方法：
     - （1）在风格迁移的基础上，使用了单区域和多区域攻击方式，通过攻击前景、后景和像素点达到不同的攻击效果。
 
- 【】研讨
   
   - 论文： 
   - 论文地址：
   - 分享汇报：唐国根
   - 动机：
   - 论文方法：
     - （1）
     - （2）

#### 2021.12.22

- 【】研讨
   
   - 论文： 
   - 论文地址：
   - 分享汇报：李丽
   - 动机：
   - 论文方法：
     - （1）
     - （2）
 
- 【图随机网络GRAND】研讨
   
   - 论文： Graph Random Neural Networks for Semi-Supervised Learning on Graphs
   - 论文地址：https://arxiv.org/pdf/2005.11079.pdf
   - 分享汇报：黄金诚
   - 动机：增强节点表示，并且约束一致性。
   - 论文方法：
     - （1）提出了随机传播S次的DropNode节点特征增强方法，并且约束每一次的confidence都和最后该节点confidence保持一致。
