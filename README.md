# Paper-Reading

项目地址：https://github.com/cins-china/paper-reading

团队介绍：该项目主要是组内在研读顶会论文过程中，所见、所思、所想、所闻，希望组内思想碰撞以出现新的Idea。

## Notes:
数据库：https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

CCF会议的截止日期: https://ccfddl.github.io/

https://www.aminer.cn/conf
### Accepted papers list（2021.11.2）
2022年顶会 Accepted papers list - 持续更新 : https://blog.csdn.net/lijinde07/article/details/122651155?spm=1001.2014.3001.5502

AAAI 2021 全文：https://dblp.uni-trier.de/db/conf/aaai/aaai2021.html

ICLR 2021 ：https://openreview.net/group?id=ICLR.cc/2021/Conference

CVPR 2021 ：全文 https://openaccess.thecvf.com/CVPR2021?day=all

ACL-IJCNLP 2021 ：接收列表https://2021.aclweb.org/registration/accept/

ICML 2021 ：全文: http://proceedings.mlr.press/v139/

IJCAI 2021 ：全文: https://www.ijcai.org/proceedings/2021/

ICCV 2021 ：全文: https://openaccess.thecvf.com/ICCV2021?day=all

ACM MM 2021 ：接收列表: https://2021.acmmm.org/main-track-list

NeurIPS 2021 ：接收列表: https://neurips.cc/Conferences/2021/AcceptedPapersInitial

### 论文下载地址
NeurIPS Proceedings(1987-2021)https://proceedings.neurips.cc/

ICML(2013-2021)http://proceedings.mlr.press/

CVPR(2013-2021)/ICCV(2013-2021)https://openaccess.thecvf.com/menu

ICLR(2013-2021)https://openreview.net/group?id=ICLR.cc

ECCV(2018/2020)http://www.ecva.net/papers.php


## Group Member

- 张巍  weizhang-git-zw (https://github.com/weizhang-git-zw)
- 杨洋 yeung2y (https://github.com/yeung2y)
- 顾天一
- 黄金城 huangJC0429 (https://github.com/huangJC0429)
- 唐国根 Teigenm (https://github.com/Teigenm)
- 李丽	Lily029106 (https://github.com/Lily029106)
- 陈娜	nachen-na (https://github.com/nachen-na)
- 黄恺文
- 李国伟 lgw9527111 (https://github.com/lgw9527111)
- 胡诗琪 Nancy206-hu (https://github.com/Nancy206-hu)
- 李攀 bert-ply (https://github.com/bert-ply)
- 张阿聪 zac-hub (https://github.com/zac-hub)
- 王泓淏 Assassinswhh (https://github.com/Assassinswhh)
- 朱鑫鹏 CharlieZhuZhu (https://github.com/CharlieZhuZhu)
- 沈雅文 Cheesepackage (https://github.com/Cheesepackage)
- 蒋玉洁 Thealuv (https://github.com/Thealuv)
- 李雅杰 Echo-yajie (https://github.com/Echo-yajie)
- NomanChowdhury (https://github.com/NomanChowdhury)
- 周泽生 Jasen-Zhou (https://github.com/Jasen-Zhou)
- 张洪瑞 zhr11021820 https://github.com/zhr11021820)
- 夏皓凡 FINEort (https://github.com/FINEort)
- 赵珂 zkzqbycg  (https://github.com/zkzqbycg)
- 张术洵 zfss1234 (https://github.com/zfss1234)
- 付鑫 (https://github.com/1760886997)



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
   - 动机：分析过平滑和异质性问题， 找到并解决造成的原因。
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

- 【关于邻居权重和损失函数】研讨
   
   - 论文： Graph Attention Collaborative Similarity Embedding for Recommender System
   - 论文地址：https://arxiv.org/abs/2102.03135
   - 分享汇报：赵晋松
   - 动机：从不同邻居的重要性不同以及BPR损失的缺陷进行改进。
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
   
   - 论文： Document-level Relation Extraction as Semantic Segmentation
   - 论文地址：https://arxiv.org/pdf/2106.03618.pdf
   - 分享汇报：唐国根
   - 动机：局部和全局依赖关系推理
   - 论文方法：
     - （1）第一种将文档级RE作为语义分割任务的方法。
     - （2）引入语义分割Unet模型，能捕捉局部上下文信息，又能捕捉三元组之间的全局依赖关系。
     - （3）实体-实体关系矩阵上的卷积可以隐式地进行实体之间的关系推理。
 
 - 【关于语义分割应用在文档级关系抽取】研讨
   
   - 论文： Local Migration Model of lmages Based onDeep Learning against Adversarial Attacks
   - 论文地址：https://www.ieee.org/publications/rights/index.html
   - 分享汇报：李国伟
   - 动机：通过利用风格迁移攻击图像优化模型
   - 论文方法：
     - （1）在风格迁移的基础上，使用了单区域和多区域攻击方式，通过攻击前景、后景和像素点达到不同的攻击效果。

#### 2021.12.22

- 【关于双向表示增强问题】研讨
   
   - 论文： METNet A Mutual Enhanced Transformation Network for Aspect-based Sentiment Analysis
   - 论文地址：https://arxiv.org/pdf/2106.03618.pdf
   - 分享汇报：李丽
   - 动机：通过双向的方式增强文本和方面的表征，特征的质量直接影响分类的精度。
   - 论文方法：
     - （1）METNet中的方面增强模块利用上下文语义特征改进了方面的表示学习，为方面提供了更丰富的信息;
     - （2）METNet设计并实现了一个层次结构，该结构迭代地增强了方面和上下文的表示。
 
 - 【图随机网络GRAND】研讨
   
   - 论文：Graph Random Neural Networks for Semi-Supervised Learning on Graphs
   - 论文地址：https://arxiv.org/pdf/2005.11079.pdf
   - 分享汇报：黄金诚
   - 动机：增强节点表示，并且约束一致性。
   - 论文方法：
     - （1）提出了随机传播S次的DropNode节点特征增强方法，并且约束每一次的confidence都和最后该节点confidence保持一致。

#### 2021.12.27

- 【基于方面情感分类的依赖图增加双向-transformer】研讨
   
   - 论文： Dependency Graph Enhanced Dual-transformer Structure for Aspect-based Sentiment Classification
   - 论文地址：https://www.aclweb.org/anthology/2020.acl-main.588.pdf
   - 分享汇报：李攀
   - 动机：传统的transformer和基于依赖树的BiGCN能够实现适当平衡。
   - 论文方法：
     - （1）基于BiLSTM/Bert编码获得句子的上下文表示;
     - （2）然后将表示送入一种双向的transformer结构来支持“水平表示学习”与“基于图神经网络的表示学习”之间交互地学习；
     - （3）最后通过MaxPooling得到Aspect表示，再根据Attention机制来识别与Aspect相关的上下文词。
 
 - 【】研讨
   
   - 论文：Unified Robust Training for Graph Neural Networks against Label Noise
   - 论文地址：https://arxiv.org/pdf/2103.03414.pdf
   - 分享汇报：陈娜
   - 动机：针对标签噪声攻击，提出一种利用标签聚合的方式得到节点类别的概率分布，根据类别概率分布引导样本重新加权和标签校准的标签攻击防御模型
   - 论文方法：
     - （1）设计标签聚合器：从其上下文节点聚合每个标记节点的标签信息，来估计其类别概率分布
     - （2）根据训练集节点的类别概率分布设计样本重新加权策略
     - （3）根据类别概率分布对训练集节点进行标签校准

#### 2022.01.10

- 【关于深层GCN的过平滑问题】研讨
   
   - 论文： Dirichlet Energy Constrained Learning for Deep Graph Neural Networks
   - 论文地址：https://www.zhuanzhi.ai/paper/fa888970498d4ccb46392d5c7c0ab15b
   - 分享汇报：张阿聪
   - 动机：通过Dirichlet Energy约束学习原则指导训练深层模型。
   - 论文方法：
     - （1）根据理论推导得出能量值的上下界的范围，值越小，表现过平滑，值大，同类节点过度分离;
     - （2）通过权重参数正交初始化和正则化，残差连接来约束能量。

- 【基于四元数的图卷积网络】研讨
   
   - 论文： Quaternion-Based Graph Convolution Network for Recommendation
   - 论文地址：https://arxiv.org/pdf/2111.10536.pdf
   - 分享汇报：赵晋松
   - 动机：欧式空间建模的不足以及四元数空间更好的表示性。
   - 论文方法：
     - （1）将用户物品嵌入到四元数空间；
     - （2）并通过四元数特征变换进行消息传播；
     - （3）最后将每一层采用均值池化得到最终的嵌入并进行推荐。


#### 2022.02.14

- 【证据增强的文档级关系抽取框架】研讨
   
   - 论文： EIDER: Evidence-enhanced Document-level Relation Extraction
   - 论文地址：https://arxiv.org/pdf/2106.08657.pdf
   - 分享汇报：唐国根
   - 动机：文档级关系提取中需要多句推断关系，但是句子对每对实体并不同等重要，并且有些句子可能与关系预测无关。
   - 论文方法：
     - （1）证据增强的关系抽取框架，自动提取证据并有效地利用提取的证据（用于推测关系的最小的句子集合）以提高DocRE的性能。
     - （2）通过使用混合层来优化模型推理阶段，将原始文档的预测结果与提取的证据融合在一起，使模型更关注重要的句子而不会丢失信息。

#### 2022.02.19

- 【无监督学习上的对抗训练】研讨
   
   - 论文： Unsupervised Adversarially Robust Representation Learning on Graphs
   - 论文地址：https://arxiv.org/pdf/2012.02486.pdf
   - 分享汇报：陈娜
   - 动机：在无监督学习上（预训练模型）通过对抗训练得到鲁棒的表示，并将该表示应用于下游任务达到防御对抗攻击的效果
   - 论文方法：
     - （1）利用互信息构建评估图表示的脆弱性和无监督学习的优化目标函数
#### 2022.02.28

- 【关于改进目标检测算法】研讨
   
   - 论文： R-FCN: Object Detection via Region-based Fully Convolutional Networks
   - 论文地址：https://arxiv.org/pdf/1605.06409v2.pdf
   - 分享汇报：胡诗琪
   - 动机：从分类和检测两方面来提高目标检测的速度。
   - 论文方法：
     - （1）仅使用ResNet卷积层（去掉全连接层）来提取图像特征；
     - （2）提出位置敏感图（Position-sensitive score maps）和对应的Position-sensitive ROI Pooling，有效的编码了目标的空间位置信息，增强了位置敏感性。
#### 2022.03.12

- 【深层图卷积的过平滑问题】研讨
   
   - 论文： SkipNode: On Alleviating Over-smoothing for Deep Graph Convolutional Networks
   - 论文地址：https://arxiv.org/abs/2112.11628
   - 分享汇报：张阿聪
   - 动机：用一种即插即用的方法缓解过平滑。
   - 论文方法：
     - （1）通过随机或者基于节点度采样一部分节点只进行传递自身信息，不聚合邻居信息；
     - （2）这种采样节点进行跳跃卷积的方法可以应用于大部分基线模型。

- 【关于对比学习在推荐上应用】研讨
   
   - 论文： Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning
   - 论文地址：https://arxiv.org/pdf/2202.06200.pdf
   - 分享汇报：赵晋松
   - 动机：来缓解数据的稀疏性，充分利用对比学习的潜力
   - 论文方法： 
     - （1）提出了一种新的对比学习范式，称为邻域丰富的对比学习（NCL），以明确地捕捉潜在的节点关联性，用于图形协同过滤的对比学习。
     - （2）从图形结构和语义空间两个方面考虑用户（或项目）的邻居。
     - （3）首先，为了利用交互图上的结构邻居，开发了一种新的结构对比目标，可以与基于GNN的协同过滤方法相结合。其次，为了利用语义邻域，通过聚类嵌入并将语义邻域合并到原型对比目标中，来获得用户项的原型。
### 2022.10.14
- 【半监督层次图分类】
   - 论文： Semi-Supervised Hierarchical Graph Classification
   - 论文地址：https://arxiv.org/pdf/2206.05416
   - 分享汇报：王泓淏
   - 动机：使用半监督的方法对层次图进行分类，可以提高分类准确性。
   - 论文方法：
      - （1）以半监督的方式来对数据进行监督训练，使用了为标签技术。
      - （2）从两个视角来看层次图分类，设计了两个分类器IC和HC。前者主要是对图实例进行分类，后者主要是对层次图进行分类。使用马尔科夫链证明了输入信息和最后输出新的一致性。
      - （3）提出了HGMI层次图互信息这个新的指标，用来判断输入与输出是否一致。
- 【谱图鲁棒性提升】
   - 论文： GRANET：Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks
   - 论文地址： https://arxiv.org/pdf/2201.12741v2.pdf
   - 分享汇报：沈雅文
   - 动机：使用谱图嵌入和概率图模型来提高GNN模型的鲁棒性
   - 论文方法：
     - （1） 通过加权谱嵌入和K近邻算法构建一个基础图。
     - （2） 通过概率图模型修建不重要的边进一步定义干净基础图G_base。
     -  (3)  在阶段二的G_base基础上训练GNN模型以此来提高鲁棒性。
### 2022.10.21

- 【对抗生成网络】研讨
   - 论文:  Protecting Facial Privacy：Generating Adversarial Identity Masks via Style-robust Makeup Transfer
   - 论文地址：https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Protecting_Facial_Privacy_Generating_Adversarial_Identity_Masks_via_Style-Robust_Makeup_CVPR_2022_paper.pdf
   - 分享汇报： 李国伟
   - 动机：利用对抗样本对人脸图像的隐私保护，利用GAN的循环一致性产生具有风格的对抗样本
   - 论文方法： 
      - （1） 在GAN原有循环一致性基础上加入正则项，使得加入对抗后达成新的循环一致性。
      - （2） 利用攻击集成模型，实现黑盒攻击。
### 2022.11.22

- 【骨折CT图像检测】研讨
   - 论文：ORF-Net: Deep Omni-supervised Rib Fracture Detection from Chest CT Scans
   - 论文地址：https://arxiv.org/pdf/2207.01842
   - 分享汇报：高美琳
   - 动机：骨折标注费时费力且需要专业知识，而且在实际应用中会面临多种不同的监督标签，为了减少对昂贵标注数据的依赖以及更好的利用有标签和无标签的数据，提出全方位监督目标检测网络。
   - 论文方法：
     - （1） 全方位监督检测头，以利用不同形式的标注数据
     - （2） 动态标签分配策略（IGM图）
     - （3） 置信度感知的分类损失，进一步强调置信度更高的样本
### 2022.11.29

- 【异构图神经网络鲁棒性】研讨
   - 论文： Robust Heterogeneous Graph Neural Networks against Adversarial Attacks
   - 论文地址：https://www.aaai.org/AAAI22Papers/AAAI-11130.ZhangM.pdf
   - 分享汇报： 沈雅文
   - 动机：由于异构图的脆弱性，在节点分类中难以获得好的效果，在2019年发表的HAN模型基础上增加了概率先验准则和可区分的净化器，使其模型效果得到提升
   - 论文方法：
      - （1） 在HAN模型上增加了概率矩阵先验准则，相当于左边的随机游走
      - （2） 之后增加一个可区分的净化器，选取T个可靠的邻居，屏蔽其他邻居
- 【基于推荐信息瓶颈的对比图结构学习】研讨
   - 论文： Contrastive Graph Structure Learning via Information Bottleneck for Recommendation
   - 论文地址：https://openreview.net/pdf?id=lhl_rYNdiH6
   - 分享汇报：赵晋松
   - 动机：构建对推荐任务更好的图扩充，并减少流行偏见和噪声交互的能力
   - 论文方法：
     - （1） 通过自适应地丢弃节点和边来构建优化的图结构，以用于用户和项目的多视图表示学习，这为缓解流行偏见提供了理论依据
     - （2） 将信息瓶颈集成到推荐的多视图对比学习过程中，有效地丢弃与下游推荐无关的信息
### 2022.12.5

- 【基于残差网络对分子性质预测】研讨
   - 论文：Equivariant Message Passing for the Prediction of Tensorial Properties and Molecular Spectra
   - 论文地址：https://arxiv.org/pdf/2102.03150
   - 分享汇报：王泓淏
   - 动机：不变表示的局限性是限制MPNN在分子化学性质预测的一个主要原因，不变的消息传递机制无法区别分子中的某些原子
   - 论文方法：
     -（1） 作者通过对2017年自己的模型Schnet进行改进，在原有的连续滤波卷积的基础上，将消息传递机制改为了等变的消息传递机制
     -（2） 文中使用门控机制，将矢量和标量的化学性质，转化为张量，通过实验证明在大部分性质预测上能精确
- 【基于情感分析中方面-观点抽取】研讨
   - 论文：Open-Domain Aspect-Opinion Co-Mining with Double Layer Span Extraction
   - 论文地址：https://dl.acm.org/doi/pdf/10.1145/3534678.3539386
   - 分享汇报：李攀
   - 动机：由于缺乏训练数据，它们受限于开放域任务，缺乏挖掘方面-观点对应关系
   - 论文方法：
     - （1） 作者通过使用基于通用依赖解析的规则为未标注的语料库生成弱标签
     - （2） 作者通过弱监督标签进行自训练来训练双层跨度提取框架
### 2022.12.22

- 【基于图像的对抗攻击】研讨
   - 论文：Adv-Attribute: Inconspicuous and Transferable Adversarial Attack on Face Recognition
   - 论文地址：https://arxiv.org/abs/2210.06871
   - 分享汇报：李国伟
   - 动机：不提出通过对高级语义的扰动来生成攻击，而不是对低级像素进行扰动，以提高攻击的可移植性。
   - 论文方法：
     - （1） 利用目标图像和原图像的特征差异生成对抗噪声，结合属性选择策略，选择属性与对抗噪声融合输入styleGAN生成对抗样本
     - （2） 同时提出多目标优化方式，达到攻击与隐蔽的平衡
### 2023.2.21

- 【推荐系统综述】研讨
   - 分享汇报：赵晋松
   - 动机：对现有研究工作进行系统和全面的回顾，以及未来改进方向。
   - 难点：1.数据稀疏性 2.噪声 3.长尾分布/偏差 4.用户/物品之间显式关系 5.计算效率
   - 方法：1.图构造 2.消息传递机制 3.自监督 4.采样
   - 未来方向：1.多模态/跨域推荐 2.大规模图 3.自监督 4.鲁棒性
- 【图神经网络鲁棒性综述】研讨
   - 分享汇报：沈雅文
   - 动机：图神经网络不够鲁棒产生的安全问题
   - 难点：如何纯化图结构，如何在聚合函数及消息传递机制方面将扰动达到最低值
   - 方法总结：图结构纯化，对抗训练，高低中频滤波器
   - 对自己方向的认识：如何设计卷积核和滤波函数，使得模型有效过滤掉异常的结构信息；如何设计鲁棒的消息传递机制和聚合函数使得异常信息影响不大。
### 2023.02.27

- 【方面级情感分析综述】研讨
   - 分享汇报：李攀
   - 动机：细粒度方面级情感分析任务杂乱，缺乏统一的分类，以及总结统一的建模范式
   - 难点：如何学习各情感要素的表征
   - 方法总结：通常采用四种建模范式（Token-level、Sequence-level、MRC、Sequence-to-Sequence）和管道方法来解决该类任务
   - 认识：对ABSA任务的建模方法总结，以及更好表征ABSA任务，对ASTE任务建模有新的启发
- 【隐式情感分析综述】研讨
	 - 分享汇报：李雅杰
	 - 动机：提高隐式情感句子分类的准确性
	 - 难点：句子中没有明显的情感词，使用显式情感词的模型效果不佳；无法界定一些本身具有歧义或者涉及宗教伦理信仰等的词语；倘若借助外部知识库，一些知识库的更新赶不上热词。
	 - 方法总结：有监督的对比学习；借助外部知识增强句子表征；层次张量组合。
- 【基于图像的对抗攻击】研讨
   - 分享汇报：朱鑫鹏
   - 动机:跟进图像对抗攻击最热门方法。
   - 论文方法总结:基于梯度的方法：在FGSM的基础上，变换不同的梯度迭代方式，或者使用方差等数学方式去添加约束。
- 【基于图像的对抗攻击】研讨
   - 分享汇报：李国伟
   - 动机:跟进图像对抗攻击最热门方法。
   - 论文方法总结:在热门的对抗研究中除开梯度的相关研究，有两个方向：1.基于GAN的，通常考虑如何去在向量空间中，做不可察觉的扰动。2.基于风格迁移的对抗工作，通过编辑、添加噪声属性，在图像上攻击。
### 2023.3.13

- 【分子属性预测综述】研讨
   - 分享汇报： 王泓淏
   - 动机：帮助医疗和化学领域
   - 难点：如何获取一个良好的图级别表征
   - 方法总结：2D分子表征， 3D分子表征， 对比学习相互补充
   - 对自己方向的认识：2D和3D分子相互补充，如何编码3D图里的信息，消息传递机制设计与改良

- 【图神经网络节点分类综述】研讨
   - 分享汇报： 张阿聪
   - 动机：解决图神经网络的异质性问题
   - 难点：如何寻找更优的高阶邻居节点，如何聚合高阶邻居节点信息
   - 方法总结：通过目标节点和高阶邻居的表征的相似性选取高阶邻居节点；利用拼接或者筛选表征维度聚合信息
### 2023.3.20

- 【基于超复数的协同过滤】研讨
   - 论文：Hypercomplex Graph Collaborative Filtering
   - 论文地址：https://dl.acm.org/doi/abs/10.1145/3485447.3512065
   - 分享汇报：王泓淏
   - 动机：为了弥补现有方法只针对特定的低维超复代数（如复代数和四元数代数）设计，而忽略了对高维代数的探索和利用，以及大多数推荐者将每一次用户-项目交互都视为一个孤立的数据实例，而不考虑高阶协作关系这两个问题
   - 论文方法：
     - （1） 首先，作者引入Cayley–Dickson构造来探索高维超复形空间。Cayley-Dickson构造利用递归过程产生一系列超复数代数，称为Cayley-Dickson代数。
     - （2） 其次，基于Cayley-Dickson代数，作者设计了一个超复数图卷积算子，通过聚集超复数空间中的邻居信息来学习节点表示。
     - （3） 最后，一个超复数线性聚合器和一个超复杂交互聚合器。线性聚合器使用超复数和池来学习邻域的摘要。交互聚合器在中心节点和邻居嵌入之间应用超复数乘法来捕获邻居特征交互。
- 【半监督任务】研讨
   - 论文：SOFTMATCH: ADDRESSING THE QUANTITY-QUALITY TRADE-OFF IN SEMI-SUPERVISED LEARNING 
   - 论文地址：https://arxiv.org/pdf/2301.10921.pdf
   - 分享汇报：沈雅文
   - 动机：由于现在大多数现实数据集处于无标签状态，需要合适的伪标签技术使得模型中引入大量的伪标签。
   - 论文方法：
     - （1） 在无监督损失中加入权重函数，并且权重函数服从截断高斯分布
     - （2） 使用UA，让伪标签分布平均
### 2023.3.27

- 【基于统一图神经网络中的随机删除】研讨
   - 论文：DropMessage: Unifying Random Dropping for Graph Neural Networks
   - 论文地址：https://arxiv.org/pdf/2204.10037.pdf
   - 分享汇报：赵晋松
   - 动机：首先，现有随机丢弃的方法由于不同数据集和模型的差异，很难找到一种适用于所有情况的通用方法。其次，引入GNN的增广数据导致参数覆盖不完整和训练过程不稳定。第三，没有关于随机丢弃方法对GNN的有效性的理论分析。
   - 论文方法： 提出了一种新的随机丢弃方法DropMessage，它在消息传递过程中直接对传播的消息执行丢弃操作。并且DropMessage为大多数现有的随机丢弃方法提供了一个统一的框架，并在此基础上对其有效性进行了理论分析。
### 2023.4.3

- 【图对比学习】研讨
   - 论文：Neighbor Contrastive Learning on Learnable Graph Augmentation
   - 论文地址：https://arxiv.org/ftp/arxiv/papers/2301/2301.01404.pdf
   - 分享汇报：李攀
   - 动机：现有图对比学习视图增强策略会破坏图拓扑，现有的对比学习会不满足图同质性假设
   - 论文方法：
     - （1）利用多头图注意力函数生成自适应增强视图，它与节点嵌入都是端到端的
     - （2）构建邻居对比损失，根据同质性假设，利用视图内和视图间的直接邻居作额外的正例样本，非一阶邻居作为负例
- 【图自动编码器】研讨
   - 论文：GraphMAE: Self-Supervised Masked Graph Autoencoders
   - 论文地址：https://arxiv.org/pdf/2205.10803.pdf
   - 分享汇报：李雅杰
   - 动机：缓解生成式自监督图预训练的四个问题：1）过度强调结构信息；2）无损坏的特征重建可能不够健壮；3）均方误差可能是敏感和不稳定的；4）MLP作为decoder表达性差。
   - 论文方法：
     - （1）随机地[MASK]图中的某一节点，然后使用GNN encoder编码图，生成code，在decoder阶段，对code再次进行[DMASK]，之后使用GNN decoder进行解码，最后对节点特征进行重建。
     - （2）使用缩放余弦误差作为特征重建的评价标准。
### 2023.4.10

- 【图注意力网络】研讨
   - 论文：HOW ATTENTIVE ARE GRAPH ATTENTION NETWORKS?
   - 论文地址：https://openreview.net/forum?id=F72ximsx7C1
   - 分享汇报：朱鑫鹏
   - 动机：作者发现GAT的attention计算比较有局限性的，attention评分的排序不受query node的限制，不论query node怎么变，得到的attention weights的ranking结果是一样的，这样会导致有一些简单的图问题是GAT无法处理的。
   - 论文方法：
     - （1）证明了GAT的局限性，并将GAT注意力计算方式称为static attention。
     - （2）提出了dynamic attention，给出了相应的证明，并且构建了dynamic attention，称为GATv2。
### 2023.5.9

- 【分子属性预测】研讨
   - 题目： UNI-MOL: A UNIVERSAL 3D MOLECULARREPRESENTATION LEARNING FRAMEWORK
   - 作者： Gengmo Zhou1, Zhifeng Gao2, Qiankun Ding, Hang Zheng, Hongteng Xu, Zhewei Wei, Linfeng Zhang, Guolin Ke
   - 单位： 中国人民大学， 深势科技， 北京科学智能研究院
   - 解决问题： 在大多数MRL方法中，分子被视为1D顺序标记或2D拓扑图，限制了它们将3D信息用于下游任务的能力，特别是使得模型几乎不可能进行3D几何预测/生成。
   - 关键思路： 通过预训练方式，在Graphormer中添加对表示，通过对遮盖原子和原子坐标预测两个代理任务，使得模型拥有学习3D结构信息的能力，使得模型学到的表征能包含丰富的3D信息，能够在下游任务中拥有更好的表达能力。
   - 论文摘要： Molecular representation learning (MRL) has gained tremendous attention due to its critical role in learning from limited supervised data for applications like drug design. In most MRL methods, molecules are treated as 1D sequential tokens or 2D topology graphs, limiting their ability to incorporate 3D information for downstream tasks and, in particular, making it almost impossible for 3D geometry prediction/generation. In this paper, we propose a universal 3D MRL framework, called Uni-Mol, that significantly enlarges the representation ability and application scope of MRL schemes. Uni-Mol contains two pretrained models with the same SE(3) Transformer architecture: a molecular model pretrained by 209M molecular conformations; a pocket model pretrained by 3M candidate protein pocket data. Besides, Uni-Mol contains several finetuning strategies to apply the pretrained models to various downstream tasks. By properly incorporating 3D information, Uni-Mol outperforms SOTA in 14/15 molecular property prediction tasks. Moreover, Uni-Mol achieves superior performance in 3D spatial tasks, including protein-ligand binding pose prediction, molecular conformation generation, etc.
### 2023.5.16

- 【对抗样本】研讨
   - 题目： Adversarial Attack with Raindrops
   - 作者： Jiyuan Liu， Bingyi Lu，Mingkang Xiong，Taobao Zhang，Huilin Xiong
   - 单位： 上海交通大学
   - 解决问题：
     - （1）可以用过AdvRD生成的雨滴来提高攻击性。
     - （2）将AdvRD生成样本传入进行对抗训练，从而提高模型对雨水的鲁棒性。
   - 关键思路：利用GAN模型对于现实生活中的雨滴样式进行学习，通过生成类似于雨滴的图像来欺骗判别器，从而使生成的雨滴图像更加具有对抗攻击的能力，其中的判别器是来判定雨滴是来自自然世界还是来自GAN。
   - 论文摘要：Deep neural networks (DNNs) are known to be vulnerable to adversarial examples, which are usually designed artificially to fool DNNs, but rarely exist in real-world scenarios. In this paper, we study the adversarial examples caused by raindrops, to demonstrate that there exist plenty ofnatural phenomena being able to work as adversarial attackers to DNNs. Moreover, we present a new approach to generate adversarial raindrops, denoted as AdvRD, using the generative adversarial network (GAN) technique to simulate natural raindrops. The images crafted by our AdvRD look very similar to the real-world raindrop images, statistically close to the distribution of true raindrop images,and more importantly, can perform strong adversarial attack to the state-of-the-art DNN models. On the other side,we show that the adversarial training using our AdvRD images can significantly improve the robustness of DNNs to the real-world raindrop attacks. Extensive experiments arecarried out to demonstrate that the images crafted by AdvRD are visually and statistically close to the natural raindrop images, can work as strong attackers to DNN models,and also help improve the robustness ofDNNs to raindrop attacks.
### 2023.5.23

- 【协同过滤】研讨
   - 题目：Graph Collaborative Signals Denoising and Augmentation for Recommendation
   - 作者：Ziwei Fan, Ke Xu, Zhang Dong, Hao Peng, Jiawei Zhang, Philip S. Yu
   - 单位：伊利诺大学，亚马逊广告组，北京航空航天大学，加州大学
   - 解决问题：GCF的邻接矩阵对于具有丰富交互的用户和物品可能存在噪声，稀少交互的用户和物品也可能存在不足。邻接矩阵忽略了user-user和item-item之间的相关性。
   - 关键思路：提出了一个新的图形邻接矩阵，结合用户-用户和项目-项目的相关性，以及一个适当设计的用户-项目的交互矩阵，平衡所有用户的交互数量。预训练一个基于图的推荐方法，以获得用户/项目嵌入，然后通过      top-K采样增强用户-项目交互矩阵。我们还增加了对称的用户-用户和项目-项目的相关性组件的邻接矩阵。
   - 论文摘要：Graph collaborative filtering (GCF) is a popular technique for capturing high-order collaborative signals in recommendation systems. However, GCF’s bipartite adjacency matrix, which defines the neighbors being aggregated based on user-item interactions, can be noisy for users/items with abundant interactions and insufficient for users/items with scarce interactions. Additionally, the adjacency matrix ignores user-user and item-item correlations, which can limit the scope of beneficial neighbors being aggregated.
In this work, we propose a new graph adjacency matrix that incorporates user-user and item-item correlations, as well as a properly designed user-item interaction matrix that balances the number of interactions across all users. To achieve this, we pre-train a graph-based recommendation method to obtain users/items embeddings, and then enhance the user-item interaction matrix via top-K sampling. We also augment the symmetric user-user and item-item correlation components to the adjacency matrix. Our experiments demonstrate that the enhanced user-item interaction matrix with improved neighbors and lower density leads to significant benefits in graph-based recommendation. Moreover, we show that
the inclusion of user-user and item-item correlations can improve recommendations for users with both abundant and insufficient interactions.
### 2023.5.30

- 【图神经网络鲁棒性】研讨
   - 题目：Robust Tensor Graph Convolutional Networks via T-SVD based Graph Augmentation
   - 作者：陈川团队
   - 单位：中山大学
   - 解决问题：图神经网络的防御模型
   - 关键思路：通过T-SVD方法对高阶张量进行卷积操作
   - 相关研究：SVD分解二阶矩阵
   - 论文摘要：图神经网络（Graph Neural Networks，GNNs）展示了它们在处理图上复杂问题方面的强大能力。然而，作为深度学习模型在图上的扩展，GNNs对噪声或对抗性攻击非常脆弱，这是由于底层扰动在消息传递过程中的传播，会对最终性能产生显著影响。因此，研究一个鲁棒的GNN框架来抵御各种扰动非常重要。本文提出了一种鲁棒的张量图卷积网络（Robust Tensor Graph Convolutional Network，RT-GCN）模型来提高鲁棒性。一方面，我们利用多视角增强来减少增强的方差，并将它们组织为三阶张量，然后使用截断的T-SVD来捕捉多视角增强图的低秩性，从图预处理的角度提高鲁棒性。另一方面，为了有效捕捉多视角增强图上的视角间和视角内的信息，我们提出了张量图卷积网络（Tensor GCN，TGCN）框架，并分析了TGCN与传统GCN之间的数学关系，从模型架构的角度提高了鲁棒性。广泛的实验结果验证了RT-GCN在各种数据集上的有效性，并证明了在不同的图对抗攻击上，相对于现有模型的优越性。
### 2023.6.6

- 【分子属性预测】研讨
   - 题目：Hierarchical Molecular Graph Self-Supervised Learning for property prediction
   - 作者：Xuan Zang, Xianbing Zhao & Buzhou Tang
   - 单位：哈尔滨工业大学
   - 解决问题：目前分子属性预测问题很难获得属性标签，并且传统的GNN很难学习到分子的结构信息。
   - 关键思路：采用自监督预训练任务，设计了丰富的代理任务能从多个角度学习到更多信息；使用层次图的建模形式使得模型能学到分子的结构信息。
   - 论文摘要：Molecular graph representation learning has shown considerable strength in molecular
analysis and drug discovery. Due to the difficulty of obtaining molecular property labels, pretraining models based on self-supervised learning has become increasingly popular in molecular representation learning. Notably, Graph Neural Networks (GNN) are employed as the backbones to encode implicit representations of molecules in most existing works.However, vanilla GNN encoders ignore chemical structural information and functions implied in molecular motifs, and obtaining the graph-level representation via the READOUT function hinders the interaction of graph and node representations. In this paper, we propose Hierarchical Molecular Graph Self-supervised Learning (HiMol), which introduces a pre-training framework to learn molecule representation for property prediction. First, we present a Hierarchical Molecular Graph Neural Network (HMGNN), which encodes motif structure and extracts node-motif-graph hierarchical molecular representations. Then, we introduce Multilevel Self-supervised Pre-training (MSP), in which corresponding multi-level generative and
predictive tasks are designed as self-supervised signals of HiMol model. Finally, superior molecular property prediction results on both classification and regression tasks demonstrate the effectiveness of HiMol. Moreover, the visualization performance in the downstream dataset shows that the molecule representations learned by HiMol can capture chemical semantic information and properties.
### 2023.6.13

- 【图像对抗】研讨
   - 題目：ColorFool: Semantic Adversarial Colorization
   - 作者：Ali Shahin Shamsabadi, Ricardo Sanchez-Matilla, Andrea Cavallaro
   - 单位：Queen Mary University of London
   - 解決问题：首先提出了基于内容的无限制黑盒攻击，在以往的工作之中，大部分的对抗攻击都是在 Lp 范数的规定之下，在这种限制下，就会导致攻击效果不是特别好，该论文使用无限制攻击，很大程度提升了攻击成功率，并且使图片看起来并不违和。
   - 关键思路：首先对图片进行语义分割，并且标注出敏感部位和不敏感部位，然后对于敏感部位进行小程度的颜色变换，而对于不敏感部分则进行大范围的颜色修改，颜色修改主要集中于 Lab 颜色通道。
   - 论文摘要：Adversarial attacks that generate small Lp-norm perturbations to mislead classifiers have limited success in black-box settings and with unseen classifiers. These attacks are also not robust to defenses that use denoising filters and to adversarial training procedures. Instead, adversarial attacks that generate unrestricted perturbations are more robust to defenses, are generally more successful in black-box settings and are more transferable to unseen classifiers. However, unrestricted perturbations may be noticeable to humans. In this paper, we propose a content-based black-box adversarial attack that generates unrestricted perturbations by exploiting image semantics to selectively modify colors within chosen ranges that are perceived as natural by humans. We show that the proposed approach, ColorFool, outperforms in terms of success rate, robustness to defense frameworks and transferability, five state-of-the-art adversarial attacks on two different tasks, scene and object classification, when attacking three state-of-the-art deep neural networks using three standard datasets. The source code is available at https://github.com/smartcameras/ColorFool.
### 2023.6.20

- 【推荐系统中负反馈】研讨
   - 题目：PANE-GNN: Unifying Positive and Negative Edges in Graph Neural Networks for Recommendation
   - 作者：Ziyang Liu, Chaokun Wang, Jingcao Xu, Cheng Wu, Kai Zheng, Yang Song, Na Mou, Kun Gai
   - 单位：清华大学
   - 解决问题：基于GNN的模型主要专注于分析用户的积极反馈，而忽略了用户的负面反馈提供的有价值的见解，作者利用负反馈，改善推荐系统。
   - 关键思路：提出一个新的推荐模型PANE-GNN，聚合和更新正面反馈图和负面反馈图上的消息，并且在负面反馈图上采用对比学习来减少噪声并过滤具有高不感兴趣分数的物品。
   - 论文摘要：Recommender systems play a crucial role in addressing the issue of information overload by delivering personalized recommendations to users. In recent years, there has been a growing interest in leveraging graph neural networks (GNNs) for recommender systems, capitalizing on advancements in graph representation learning. These GNN-based models primarily focus on analyzing users’ positive feedback while overlooking the valuable insights provided by their negative feedback. In this paper, we propose PANE-GNN, an innovative recommendation model that unifies Positive And Negative Edges in Graph Neural Networks for recommendation. By incorporating user preferences and dispreferences, our approach enhances the capability of recommender systems to offer personalized suggestions. PANE-GNN first partitions the raw rating graph into two distinct bipartite graphs based on positive and negative
feedback. Subsequently, we employ two separate embeddings, the interest embedding and the disinterest embedding, to capture users’ likes and dislikes, respectively. To facilitate effective information propagation, we design distinct message-passing mechanisms for positive and negative feedback. Furthermore, we introduce a distortion to the negative graph, which exclusively consists of negative feedback edges, for contrastive training. This distortion plays a crucial role in effectively denoising the negative feedback. The experimental results provide compelling evidence that PANE-GNN surpasses the existing state-of-the-art benchmark methods across four real-world datasets. These datasets include three commonly used recommender system datasets and one open-source short video recommendation dataset.
### 2023.7.4

- 【蛋白质结构预测】研讨
   - 题目：Highly accurate protein structure prediction with AlphaFold**
   - 作者：John Jumper
   - 单位：DeepMind**
   - 解决问题：使用图神经网络进行蛋白质结构预测，首次提出将预测精度提升到原子级别
   - 关键思路：
     - （1）：使用基因数据库和结构数据库比对，对输入的蛋白质序列进行处理，得到Evoformer的输入
     - （2）：Evoformer模块使用了横向注意力和纵向注意力机制，对pairing做了三角限制，通过多次重复的Evoformer不断精化embedding的结构。最后使用structure module生成3维结构
     - （3）：用带标签数据（氨基酸序列与三维坐标的对应）先训练一遍网络，然后用训练完的网络在无标签数据（仅有氨基酸序列）上预测一遍生成新的数据集，只保留预测得好的部分，然后把这两者混合拿来再进行训练，效果更好。
   - 论文摘要：Proteins are essential to life, and understanding their structure can facilitate a mechanistic understanding of their function. Through an enormous experimental effort1–4, the structures of around 100,000 unique proteins have been determined5, but this represents a small fraction of the billions of known protein sequences6,7. Structural coverage is bottlenecked by the months to years of painstaking effort required to determine a single protein structure. Accurate computational approaches are needed to address this gap and to enable large-scale structural bioinformatics. Predicting the three-dimensional structure that a protein will adopt based solely on its amino acid sequence—the structure prediction component of the ‘protein folding problem’8—has been an important open research problem for more than 50 years9. Despite recent progress10–14, existing methods fall far short of atomic accuracy, especially when no homologous structure is available. Here we provide the first computational method that can regularly predict protein structures with atomic accuracy even in cases in which no similar structure is known. We validated an entirely redesigned version of our neural network-based model, AlphaFold, in the challenging 14th Critical Assessment of protein Structure Prediction (CASP14)15, demonstrating accuracy competitive with experimental structures in a majority of cases and greatly outperforming other methods. Underpinning the latest version of AlphaFold is a novel machine learning approach that incorporates physical and biological knowledge about protein structure, leveraging multi-sequence alignments, into the design of the deep learning algorithm. AlphaFold predicts protein structures with an accuracy competitive with experimental structures in the majority of cases using a novel deep learning architecture.

- 【局部同质性水平差异】研讨
   - 题目：On Performance Discrepancies Across Local Homophily Levels in Graph Neural Networks
   - 作者：Donald Loveland,Jiong Zhu,Mark Heimann,Benjamin Fish,Michael T. Schaub,Danai Koutra
   - 单位：University of Michigan
   - 解决问题：提出了局部同质性水平对性能的影响，在以往的工作中所做出的假设，
数据集通常被视为跨节点具有恒定的同质性水平，过度依赖于全局同质性，GNN可能无法泛化到偏离图的全局同质性的测试节点，该论文提出通过调控局部同质性水平，对局部同质性如何影响性能进行系统的实证研究。
   - 关键思路：在同质性分析中常用的优先附着模型中引入了一个新参数，以能够控制生成的图中的局部同质性水平，进而对局部同质性如何影响性能进行系统的研究。
   - 摘要：Research on GNNs has highlighted a relationship between high homophily (i.e., the tendency for nodes of a similar class to connect) and strong predictive performance in node classification. However, recent research has found the relationship to be more nuanced, demonstrating that even simple GNNs can learn in certain heterophilous settings. To bridge the gap between these findings, we revisit the assumptions made in previous works and identify that datasets are often treated as having a constant homophily level across nodes. To align closer to real-world datasets, we theoretically and empirically study the performance of GNNs when the local homophily level of a node deviates at test-time from the global homophily level of its graph. To aid our theoretical analysis, we introduce a new parameter to the preferential attachment model commonly used in homophily analysis to enable the control of local homophily levels in generated graphs, enabling a systematic empirical study on how local homophily can impact performance
### 2023.7.11

- 【蛋白质结构预测】研讨
   - 题目：MSA Transformer
   - 作者：Roshan Rao，Jason Liu，Robert Verkuil，Joshua Meier，John F. Canny，Pieter Abbeel，Tom Sercu，Alexander Rives
   - 单位：Alexander River
   - 解决问题：引入了一种多序列对齐的形式作为输入，利用了蛋白质序列之间的共进化信息，降低了模型中的参数量，提高了模型预测性能。
   - 关键思路：将注意力交叉在对齐的行和列上，并提供一个一维序列位置嵌入，独立添加到msa每一行，最后根据序列之间的相似性捆绑行注意力。
   - 摘要：Unsupervised protein language models trained
across millions of diverse sequences learn structure and function of proteins. Protein language
models studied to date have been trained to perform inference from individual sequences. The longstanding approach in computational biology has been to make inferences from a family of evolutionarily related sequences by fitting a model to each family independently. In this work we combine the two paradigms. We introduce a protein language model which takes as input a set of sequences in the form of a multiple sequence alignment. The model interleaves row and column attention across the input sequences and is trained with a variant of the masked language modeling objective across many protein families. The performance of the model surpasses current state-ofthe-art unsupervised structure learning methods
by a wide margin, with far greater parameter effi- ciency than prior state-of-the-art protein language models.
### 2023.9.28

- 【子图分类】研讨
   - 题目：Position-Aware Subgraph Neural Networks with Data-Efficient Learning
   - 作者：Chang Liu，Yuwen Yang，Zhe Xie，Hongtao Lu，Yue Ding
   - 单位：Shanghai Jiao Tong University Shanghai, China
   - 解决问题：
     - （1）现有数据严重稀疏，少量节点主导了子图的表示学习。
     - （2）现有方法很麻烦，需要很多计算资源。
   - 关键思路：
     - （1）提出了一种基于余弦相位编码的方案，无需选取锚点和设计模型，而是直接通过节点间的距离进行编码。以此省去了大量的训练开销。
     - （2）针对子图的特性设计了生成式的增广方案以及对比学习方法，以此实现了子图上的数据高效学习，并缓解了样本不足带来的 bias 问题。
   - 摘要：Data-efficient learning on graphs (GEL) is essential in real-world applications. Existing GEL methods focus on learning useful representations for nodes, edges, or entire graphs with “small” labeled data. But the problem of data-efficient learning for subgraph prediction has not been explored. The challenges of this problem lie in the following aspects: 1) It is crucial for subgraphs to learn positional features to acquire structural information in the base graph in which they exist. Although the existing subgraph neural network method is capable of learning disentangled position encodings, the overall computational complexity is very high. 2) Prevailing graph augmentation methods for GEL, including rule-based, sample-based, adaptive, and automated methods, are not suitable for augmenting subgraphs because a subgraph contains fewer nodes but richer information such as position, neighbor, and structure. Subgraph augmentation is more susceptible to undesirable perturbations. 3)Only a small number of nodes in the base graph are contained in subgraphs, which leads to a potential “bias” problem that the subgraph representation learning is dominated by these“hot”nodes. By contrast, the remaining nodes fail to be fully learned, which reduces the generalization ability of subgraph representation learning. In this paper, we aim to address the challenges above and propose a Position-Aware Data-Efficient Learning framework for subgraph neural networks called PADEL. Specifically, we propose a novel node position encoding method that is anchor-free, and design a new generative subgraph augmentation method based on a diffused variational subgraph autoencoder, and we propose exploratory and exploitable views for subgraph contrastive learning. Extensive experiment results on three real-world datasets show the superiority of our proposed method over state-of-the-art baselines.

- 【图神经网络同质性、异质性问题】研讨
   - 题目：LSGNN: Towards General Graph Neural Network in Node Classification by Local Similarity
   - 作者：Yuhan Chen，Yihong Luo， Jing Tang，Liang Yang， Siya Qiu，Chuan Wang，Xiaochun Cao
   - 单位：arXiv
   - 解决问题：文章将节点间的拓扑信息纳入考虑范畴，提出局部相似度作为指示条件，为了解决图神经网络的性能在不同类型图中能够不受损害这一问题。
   - 关键思路：提出局部相似度来学习节点级加权融合，以及初始残差连接模型进行特征提取。
   - 摘要：- Heterophily has been considered as an issue that hurts the performance of Graph Neural Networks (GNNs). To  address this issue, some existing work uses a graph-level weighted fusion of the information of multi-hop neighbors to include more  nodes with homophily. However, the heterophily might differ among nodes, which requires to consider the local topology. Motivated by it, we propose to use the local similarity (LocalSim) to learn node level weighted fusion, which can also serve as a plug-and-play module. For better fusion, we propose a novel and efficient Initial Residual Difference Connection (IRDC) to extract more informative multi-hop information. Moreover, we provide theoretical analysis on the effectiveness of LocalSim representing node homophily on synthetic graphs. Extensive evaluations over real bench mark datasets show th at our proposed method, namely Local Similarity Graph Neural Network (LSGNN), can offer comparable or superior state of-the-art performance on both homophilic and heterophilic graphs. Meanwhile, the plug-and play model can significantly boost the performance of existing GNNs.
### 2023.10.12

- 【交叉注意力机制】研讨
   - 题目： Tree Cross Attention
   - 作者：Leo Feng, Frederick Tung, Hossein Hajimirsadeghi, Yoshua Bengio, Mohamed Osama Ahmed
   - 单位：arXiv
   - 解决问题：为避免原始Cross Attention的all-to-all计算，利用强化学习构建在Attention的计算过程中构建具有选择机制的树节点，降低模型计算复杂度的同时，避免模型的损失。
   - 关键思路：利用强化学习构建特征树，利用query查询机制进行token选择。
   - 摘要：Cross Attention is a popular method for retrieving information from a set of context tokens for making predictions. At inference time, for each prediction, Cross Attention scans the full set of O(N) tokens. In practice, however, often only a small subset of tokens are required for good performance. Methods such as Perceiver IO are cheap at inference as they distill the information to a smaller-sized set of latent tokens L<N on which cross attention is then applied, resulting in only O(L) complexity. However, in practice, as the number of input tokens and the amount of information to distill increases, the number of latent tokens needed also increases significantly. In this work, we propose Tree Cross Attention (TCA) - a module based on Cross Attention that only retrieves information from a logarithmic O(log(N)) number of tokens for performing inference. TCA organizes the data in a tree structure and performs a tree search at inference time to retrieve the relevant tokens for prediction. Leveraging TCA, we introduce ReTreever, a flexible architecture for token-efficient inference. We show empirically that Tree Cross Attention (TCA) performs comparable to Cross Attention across various classification and uncertainty regression tasks while being significantly more token-efficient. Furthermore, we compare ReTreever against Perceiver IO, showing significant gains while using the same number of tokens for inference.

- 【过度平滑和过度挤压关系】研讨
   -  题目：On the Trade-off between Over-smoothing and Over-squashing in Deep Graph Neural Networks
   -  作者：Jhony H. Giraldo，Thierry Bouwmans，Konstantinos Skianis，Fragkiskos D. Malliaros
   -  单位：Institut Polytechnique de Paris - Télécom Paris
   -  解决问题：提出了一种基于曲率的重新布线算法，利用过度平滑和过度挤压的平衡关系，提高了模型预测性能。
   -  关键思路：推导了过度平滑和过度挤压与谱间隙的关系，并根据曲率对图的边进行添加和删除。
   -  摘要：Graph Neural Networks (GNNs) have succeeded in various computer science applications, yet deep GNNs underperform their shallow counterparts despite deep learning’s success in other domains. Over-smoothing and over-squashing are key challenges when stacking graph convolutional layers, hindering deep representation learning and information propagation from distant nodes. Our work reveals that over-smoothing and over-squashing are intrinsically related to the spectral gap of the graph Laplacian, resulting in an inevitable trade-off between these two issues, as they cannot be alleviated simultaneously. To achieve a suitable compromise, we propose adding and removing edges as a viable approach. We introduce the Stochastic Jost and Liu Curvature Rewiring (SJLR) algorithm, which is computationally efficient and preserves fundamental properties compared to previous curvature-based methods. Unlike existing approaches, SJLR performs edge addition and removal during GNN training while maintaining the graph unchanged during testing. Comprehensive comparisons demonstrate SJLR’s competitive performance in addressing over-smoothing and over-squashing.
### 2023.10.19

- 【分子属性预测】研讨
   - 题目：Molecular property prediction by contrastive learning with attention-guided positive sample selection
   - 作者：Jinxian Wang, Jihong Guan, Shuigeng Zhou
   - 单位：Shanghai Key Lab of Intelligent Information Processing, and School of Computer Science, Fudan University
   - 解决问题：通过注意力机制，为对比学习选择最优的正对。
   - 关键思路：通过不同的结合注意力的mask方式，对smiles化学字符串表示进行数据增强。
   - 摘要：Motivation: Predicting molecular properties is one of the fundamental problems in drug design and discovery. In recent years, self-supervised learning (SSL) has shown its promising performance in image recognition, natural language processing, and single-cell data analysis. Contrastive learning (CL) is a typical SSL method used to learn the features of data so that the trained model can more effectively distinguish the data. One important issue of CL is how to select positive samples for each training example, which will significantly impact the performance of CL. Results: In this article, we propose a new method for molecular property prediction (MPP) by Contrastive Learning with Attention-guided Positive-sample Selection (CLAPS). First, we generate positive samples for each training example based on an attention-guided selection scheme. Second, we employ a Transformer encoder to extract latent feature vectors and compute the contrastive loss aiming to distinguish positive and negative sample pairs. Finally, we use the trained encoder for predicting molecular properties. Experiments on various benchmark datasets show that our approach outperforms the state-of-the-art (SOTA) methods in most cases.
   
- 【对抗样本】研讨
   - 题目：Towards Transferable Targeted Adversarial Examples
   - 作者：Zhibo Wang, Hongshan Yang, Yunhe Feng, Peng Sun, Hengchang Guo, Zhifei Zhang, Kui Ren   
   - 单位：School of Cyber Science and Technology, Zhejiang University, P. R. China ZJU-Hangzhou Global Scientific and Technological Innovation Center Department of Computer Science and Engineering, University of North Texas, USA College of Computer Science and Electronic engineering, Hunan University, P. R. China School of Cyber Science and Engineering, Wuhan University, P. R. China Adobe Research
   - 解决问题：使用生成对抗模型，来生成攻击成功率高且迁移性强的对抗样本。
   - 关键思路：利用一个生成器，两个判别器来进行生成对抗训练，一个判别器用来约束label信息，一个判别器用来约束Feature信息，最后使用随机掩码方式，来对图像进行优化。
   - 摘要：Transferability of adversarial examples is critical for black-box deep learning model attacks. While most existing studies focus on enhancing the transferability ofuntargeted adversarial attacks, few of them studied how to generate transferable targeted adversarial examples that can mislead models into predicting a specific class. Moreover, existing transferable targeted adversarial attacks usually fail to suf- ficiently characterize the target class distribution, thus suffering from limited transferability. In this paper, we pro-pose the Transferable Targeted Adversarial Attack (TTAA), which can capture the distribution information of the target class from both label-wise and feature-wise perspectives, to generate highly transferable targeted adversarial examples. To this end, we design a generative adversarial training framework consisting ofa generator to produce targeted adversarial examples, and feature-label dual discriminators to distinguish the generated adversarial examples from the target class images. Specifically, we design the label discriminator to guide the adversarial examples to learn label-related distribution information about the target class. Meanwhile, we design a feature discriminator, which extracts the feature-wise information with strong cross-model consistency, to enable the adversarial examples to learn the transferable distribution information. Furthermore, we introduce the random perturbation dropping to further enhance the transferability by augmenting the diversity ofadversarial examples used in the training process.Experiments demonstrate that our method achieves excellent performance on the transferability of targeted adversarial examples. The targeted fooling rate reaches 95.13% when transferred from VGG-19 to DenseNet-121, which significantly outperforms the state-of-the-art methods.
### 2023.11.2

- 【蛋白质对相互作用预测】研讨
  - 题目：Multifaceted protein–protein interaction prediction based on Siamese residual RCNN
  - 单位： University of California, Los Angeles
  - 解决问题：蛋白质对相互作用预测，蛋白质对相互作用类型预测
  - 关键思路：用CNN去卷积氨基酸间的局部特征，双向GRU去学习序列上的依赖，从而获得蛋白质的表征
  - 论文摘要：Sequence-based protein–protein interaction (PPI) prediction represents a fundamental computational biology problem. To address this problem, extensive research efforts have been made to extract predefined features from the sequences. Based on these features, statistical algorithms

 - 【蛋白质结构预测】研讨
    - 题目：Hierarchical graph learning for protein– protein interaction  
    - 作者：Ziqi Gao 1,2, Chenran Jiang3, Jiawen Zhang1, Xiaosen Jiang4, Lanqing Li 5, Peilin Zhao5, Huanming Yang 4, Yong Huang 6 & Jia Li 1,2  
    - 单位：香港科技大学、中国科学院大学、腾讯公司人工智能实验室
    - 解决问题：蛋白质相互作用预测
    - 关键思路：
      - （1）HIGH-PPI使用底部蛋白质内部视图(BGNN)和顶部蛋白质外部视图(TGNN)对结构蛋白质表示进行建模。在底视图中，HIGH-PPI通过将氨基酸残基视为节点，将物理邻接视为边来构建蛋白质图。因此，BGNN以协同方式整合了蛋白质3D结构和残基水平特性的信息。在顶视图中，HIGH-PPI以蛋白质图为节点，以相互作用为边构建PPI图，并与TGNN学习蛋白质对之间的关系。
      - （2）BGNN包含2个GCN块，输入是邻接矩阵和残基级特征矩阵，经过两个GCN块后，再经过自注意图(SAG)池化的读出操作并使用平均聚合来确保固定长度的嵌入向量输出。
      - （3）TGNN的输入是BGNN的输出，PPI图的节点特征使用3个GIN块的递归邻域聚合进行更新。两个任意蛋白质嵌入通过串联操作组合，然后应用多层感知器(MLP)作为分类器进行预测。
    - 论文摘要：Protein-Protein Interactions (PPIs) are fundamental means of functions and signalings in biological systems. The massive growth in demand and cost associated with experimental PPI studies calls for computational tools for automated prediction and understanding of PPIs. Despite recent progress, in silico methods remain inadequate in modeling the natural PPI hierarchy. Here we present a double-viewed hierarchical graph learning model, HIGH-PPI, to predict PPIs and extrapolate the molecular details involved. In this model, we create a hierarchical graph, in which a node in the PPI network (top outside-ofprotein view) is a protein graph (bottom inside-of-protein view). In the bottom view, a group of chemically relevant descriptors, instead of the protein sequences, are used to better capture the structure-function relationship of the protein. HIGH-PPI examines both outside-of-protein and inside-of-protein of the human interactome to establish a robust machine understanding of PPIs. This model demonstrates high accuracy and robustness in predicting PPIs. Moreover, HIGH-PPI can interpret the modes of action of PPIs by identifying important binding and catalytic sites precisely. Overall, “HIGH-PPI [https://github.com/zqgao22/HIGH-PPI]” is a domain-knowledge-driven and interpretable framework for PPI prediction studies.
### 2023.11.7

- 【图分类】研讨
   - 题目：Multi-scale Graph Pooling Approach with Adaptive Key Subgraph for Graph Representations
   - 作者：Yiqin Lv，Zhiliang Tian，Zheng Xie，Yiping Song∗
   - 单位：National University of Defense Technology，Changsha, China
   - 解决问题：
     - （1）分层池化中丢弃节点会错误丢掉一些节点并且这样的操作是不可逆的
     - （2）分层池化中聚合节点没有考虑到关键子图信息
   - 关键思路：在本文中，我们提出了一种多尺度图神经网络（MSGNN）模型，该模型不仅保留了图的拓扑信息，而且还保留了关键子图以获得更好的可解释性。 MSGNN在迭代过程中逐渐丢弃不重要的节点并保留重要的子图结构。关键子图首先根据经验选择，然后自适应演化，为下游任务定制特定的图结构。
   - 摘要：The recent progress in graph representation learning boosts the development ofmany graph classification tasks, such as protein classification and social network classification. One of the mainstream approaches for graph representation learning is the hierarchical pooling method. It learns the graph representation by gradually reducing the scale of the graph, so it can be easily adapted to large scale graphs. However, existing graph pooling methods discard the original graph structure during downsizing the graph, resulting in a lack of graph topological structure. In this paper, we propose a multi-scale graph neural network (MSGNN) model that not only retains the topological information of the graph but also maintains the key-subgraph for better interpretability. MSGNN gradually discards the unimportant nodes and retains the important subgraph structure during the iteration. The key subgraphs are first chosen by experience and then adaptively evolved to tailor specific graph structures for downstream tasks. The extensive experiments on seven datasets show that MSGNN improves the SOTA performance on graph classification and better retains key subgraphs.

- 【图同质性异质性问题】研讨
   - 题目：Finding the Missing-half: Graph Complementary Learning for Homophily-prone and Heterophily-prone Graphs
   - 作者：Yizhen Zheng，He Zhang，Vincent CS Lee，Yu Zheng，Xiao Wang，Shirui Pan
   - 单位：Monash University
   - 解决问题：提出图互补学习，利用更完备的拓扑信息，提高节点分类任务的准确率。
   - 关键思路：提出图互补的方法，为同质倾向图添加异质边（为异质倾向图添加同质边），补充节点连接，再提出一种新的卷积方法，对补全后的图进行卷积操作获得分类信息。
   - 摘要：Real-world graphs generally have only one kind of tendency in their connections. These connections are either homophily-prone or heterophilyprone. While graphs with homophily-prone edges tend to connect nodes with the same class (i.e.,intra-class nodes), heterophilyrone edges tend to build relationships between nodes with different classes (i.e., inter-class nodes). Existing GNNs only take the original graph during training. The problem with this approach is that it forgets to take into consideration the “missing-half” structural information, that is, heterophily-prone topology for homophily-prone graphs and homophilyprone topology for heterophily-prone graphs. In our paper, we introduce Graph cOmplementAry Learning, namely GOAL, which consists of two components: graph complementation and complemented graph convolution. The first component finds the missing-half structural information for a given graph to complement it. The complemented graph has two sets of graphs including both homophily- and heterophily-prone topology. In the latter component, to handle complemented graphs, we design a new graph convolution from the perspective of optimisation. The experiment results show that GOAL consistently outperforms all baselines in eight real-world datasets.
### 2023.11.14

- 【曲率与过度平滑和过度挤压关系】研讨
   - 题目：Revisiting Over-smoothing and Over-squashing Using Ollivier-Ricci Curvature
   - 作者：JKhang Nguyen，Hieu Nong，Vinh Nguyen，Nhat Ho，Stanley Osher，Tan Nguyen
   - 单位：FPT Software AI Center, Vietnam
   - 解决问题：将曲率与过平滑和过挤压联系了起来，提出了一种基于曲率的重新布线算法，来同时缓解图上过平滑和过挤压问题。
   - 关键思路：将高曲率与过平滑联系起来，低曲率与过挤压联系起来，并根据得到的这个关系对图的边进行添加和删除操作。添加边是通过计算出对两节点传输代价贡献最大的节点对，并将其连接起来。
   - 摘要：Graph Neural Networks (GNNs) had been demonstrated to be inherently susceptible to the problems of over-smoothing and over-squashing. These issues prohibit the ability of GNNs to model complex graph interactions by limiting their effectiveness in taking into account distant information. Our study reveals the key connection between the local graph geometry and the occurrence of both of these issues, thereby providing a unified framework for studying them at a local scale using the Ollivier-Ricci curvature. Specifically, we demonstrate that oversmoothing is linked to positive graph curvature while over-squashing is linked to negative graph curvature. Based on our theory, we propose the Batch Ollivier-Ricci Flow, a novel rewiring algorithm capable of simultaneously addressing both over-smoothing and over-squashing.
### 2023.12.5

- 【基于元学习的分子属性预测】研讨
   - 题目： Graph Sampling-based Meta-Learning for Molecular Property Prediction
   - 作者： Xiang Zhuang , Qiang Zhang, Bin Wu, Keyan Ding, Yin Fang and Huajun Chen
   - 单位： 浙江大学
   - 解决问题： 分子拥有多种属性，利用分子和属性之间一对多的关系。
   - 关键思路： 将分子数据集建模成分子属性图（MPG），通过描述分子和属性之间的关系，从一个更加粗粒的层面，来预测分子的属性。
   - 摘要：Molecular property is usually observed with a limited number of samples, and researchers have considered property prediction as a few-shot problem. One important fact that has been ignored by prior works is that each molecule can be recorded with several different properties simultaneously. To effectively utilize many-to-many correlations of molecules and properties, we propose a Graph Sampling-based Meta-learning (GS-Meta) framework for few-shot molecular property prediction. First, we construct a Molecule-Property relation Graph (MPG): molecule and properties are nodes, while property labels decide edges. Then, to utilize the topological information of MPG, we reformulate an episode in metalearning as a subgraph of the MPG, containing a target property node, molecule nodes, and auxiliary property nodes. Third, as episodes in the form of subgraphs are no longer independent of each other, wepropose to schedule the subgraph sampling process with a contrastive loss function, which considers the consistency and discrimination of subgraphs. Extensive experiments on 5 commonlyused benchmarks show GS-Meta consistently outperforms state-of-the-art methods by 5.71%-6.93% in ROC-AUC and verify the effectiveness of each
 proposed module. Our code is available at https: //github.com/HICAI-ZJU/GS-Meta.

- 【Transformer结构优化】研讨
   - 题目：Efficient Long-Range Transformers: You Need to Attend More,but Not Necessarily at Every Layer
   - 作者：Qingru Zhang, Dhananjay Ram, Cole Hawkins, Sheng Zha, Tuo Zhao
   - 单位：Georgia Institute of Technology, Amazon Web Service
   - 解决问题：混合使用多种注意力结构，降低了传统transformer的计算量，同时拥有传统transformer相当的效果。
   - 关键思路：将Full Attention与Block Attention结合使用，并将Full Attention放在transformer底层结构中。
   - 摘要：Pretrained transformer models have demonstrated remarkable performance across various natural language processing tasks. These models leverage the attention mechanism to capture long- and short-range dependencies in the sequence. However, the (full) attention mechanism incurs high computational cost-quadratic in the sequence length, which is not affordable in tasks with long sequences, e.g.,inputs with 8k tokens. Although sparse attention can be used to improve computational efficiency, as suggested in existing work, it has limited modeling capacity and often fails to capture complicated dependencies in long sequences. To tackle this challenge, we propose MASFormer, an easy-to-implement transformer variant with Mixed Attention Spans.Specifically, MASFormer is equipped with full attention to capture long-range dependencies,but only at a small number of layers. For the remaining layers, MASformer only employs sparse attention to capture short-range dependencies. Our experiments on natural language modeling and generation tasks show that a decoder-only MASFormer model of 1.3B parameters can achieve competitive performance to vanilla transformers with full attention while significantly reducing computational cost (up to 75%). Additionally, we investigate the effectiveness of continual training with long sequence data and how sequence length impacts downstream generation performance, which may be of independent interest.
### 2023.12.12

- 【对比学习鲁棒性问题】研讨
   - 题目：On the Adversarial Robustness of Graph Contrastive Learning Methods
   - 作者：Filippo Guerranti, Zinuo Yi∗, Anna Starovoit∗, Rafiq Kamel∗, Simon Geisler, Stephan Günnemann
   - 单位：Technical University of Munich
   - 解决问题：提出图对比学习存在不鲁棒性的问题，通过实验全面分析了图对比学习的不鲁棒性
   - 关键思路：提出了图对比学习鲁棒性的评估准则
   - 摘要：Contrastive learning (CL) has emerged as a powerful framework for learning representations of images and text in a self-supervised manner while enhancing model robustness against adversarial attacks. More recently, researchers have extended the principles of contrastive learning to graph-structured data, giving birth to the field of graph contrastive learning (GCL). However, whether GCL methods can deliver the same advantages in adversarial robustness as their counterparts in the image and text domains remains an open question. In this paper, we introduce a comprehensive robustness evaluation protocol tailored to assess the robustness of GCL models. We subject these models to adaptive adversarial attacks targeting the graph structure, specifically in the evasion scenario. We evaluate node and graph classification tasks using diverse real-world datasets and attack strategies. With our work, we aim to offer insights into the robustness of GCL methods and hope to open avenues for potential future research directions.

- 【对抗样本】研讨
   - 题目：Frequency-aware GAN for Adversarial Manipulation Generation
   - 作者：Peifei Zhu, Genki Osada, Hirokatsu Kataoka, Tsubasa Takahashi
   - 单位：LINE Corporation
   - 解决问题：对篡改检测模型进行对抗攻击，使篡改检测器预测成功率降低
   - 关键思路：使用GAN模型，一个生成器，一个判别器，生成器是编解码器组成，并且作者提到了使用频域攻击的重要性，作者认为世界丢失和高频中的伪影是对抗噪声明显的主要原因。
   - 摘要：Image manipulation techniques have drawn growing concerns as manipulated images might cause morality and security problems. Various methods have been proposed to detect manipulations and achieved promising performance. However, these methods might be vulnerable to adversarial attacks. In this work, we design an Adversarial Manipulation Generation (AMG) task to explore the vulnerability of image manipulationdetectors. We first propose an optimalloss function and extend existing attacks to generate adversarial examples. We observe that existing spatial attacks cause large degradation in image quality and find the loss of high-frequency detailed components might be its major reason. Inspired by this observation, we propose a novel adversarial attack that incorporates both spatial and fre-quency features into the GAN architecture to generate ad-versarial examples. We further design an encoder-decode architecture with skip connections of high-frequency components to preserve fine details. We evaluated our method on three image manipulation detectors (FCN, ManTra-Net and MVSS-Net) with three benchmark datasets (DEFACTO, CASIAv2 and COVER). Experiments show that our method generates adversarial examples significantly fast (0.01s per image), preserves better image quality (PSNR 30% higher than spatial attacks), and achieves a high attack success rate. We also observe that the examples generated by AMG can fool both classification and segmentation models, which indicates better transferability among different tasks.
### 2023.12.19

- 【ViT/MLP-Mixer迁移到图】研讨
   - 题目：A Generalization of ViT/MLP-Mixer to Graphs
   - 作者：Xiaoxin He  Bryan Hooi  Thomas Laurent  Adam Perold  Yann LeCun  Xavier Bresson
   - 单位：School of Computing, University of Singapore Institute of Data Science, National University of Singapore Loyola Marymount University Element, Inc. New York University Meta AI.Correspondence to: Xiaoxin He <he.xiaoxin@u.nus.edu>.
   - 解决问题：将ViT/MLP-Mixer迁移到图中
   - 关键思路：模仿ViT/MLP-Mixer在图片中的方式，对图数据进行切分。
   - 摘要：Graph Neural Networks (GNNs) have shown great potential in the field of graph representation learning. Standard GNNs define a local message-passing mechanism which propagates information over the whole graph domain by stacking multiple layers. This paradigm suffers from two major limitations, over-squashing and poor long-range dependencies, that can be solved using global attention but significantly increases the computational cost to quadratic complexity. In this work, we propose an alternative approach to overcome these structural limitations by leveraging the ViT/MLP-Mixer architectures introduced in computer vision. We introduce a new class of GNNs, called Graph ViT/MLP-Mixer, that holds
 three key properties. First, they capture long range dependency and mitigate the issue of over squashing as demonstrated on Long Range Graph Benchmark and TreeNeighbourMatch datasets. Second, they offer better speed and memory efficiency with a complexity linear to the number of nodes and edges, surpassing the related Graph Transformer and expressive GNN models. Third, they show high expressivity in terms of graph isomorphism as they can distinguish at least 3-WL non-isomorphic graphs. We test our architecture on 4 simulated datasets and 7 real-world benchmarks, and show highly competitive results on all of them. The source code is available for reproducibility at: https://github.com/XiaoxinHe/Graph-ViT-MLPMixer
### 2023.12.26
- 【蛋白质结构预测】研讨
   - 题目：Learning spatial structures and network correlations improves unknown protein–protein interaction prediction
   - 作者：Yangyue Fang∗, Chaojian Zhang∗, Yu Fu∗, and Tao Xue∗
   - 单位：∗Alibaba Business School, Hangzhou Normal University, Hangzhou, China
   - 解决问题：蛋白质相互作用预测
   - 关键思路：
     - （1）该论文的模型由三个部分组成：结构模块，网络模块和预测模块。结构模块和网络模块用于提取蛋白质结构信息和PPI网络的拓扑信息，预测模块结合结构模块和网络模块的输出特征得到最终的预测结果
     - （2）对于结构信息，SE（3）不变矩阵映射由AlphaFold 2预测的结构文件表示，结构特征通过使用CNN和空间金字塔池（SPP）提取。对于PPI网络拓扑信息，通过微调蛋白质预训练模型并使用图神经网络（GNN）学习蛋白质之间的相关性来获得节点特征。
   - 摘要： The mechanism of action of protein-protein in-teractions(PPIs) is complex, and prediction models have to learn multiple dimensions to achieve excellent generalization performance. Therefore, based on the concept that the spatial structure of proteins is closely related to protein functions and the topological information of PPI networks reflects t he correlation between proteins, we combine the protein structure information and the topological information of PPI networks to enhance the prediction performance.
	We present a new approach, SE3NET-PPI, for multi-type PPI prediction, retrieves protein structure information from the SE(3)-invariant matrix map generated by Alphafold2 and extracts the topological information of the PPI network using a graph neural network in the Siamese architecture. Results showed that our model outperforms several state-of-the-art methods under various dataset partitioning methods, with significant improvement in predicting invisible datasets. For example, in the case of Sring all-BFS, the miroc-F1 value of our model reaches 80.28 ± 0.43, compared to 75.87 ± 0.37 for GNN-PPI and 62.30± 0.41 for PIPR. The implementation and related datasets areavailable at https://github.com/FYY99117/SE3NET-PPI.
	Index Terms—protein-protein interaction networks, multi-dimension feature confusion, graph neural network, AlphaFold2.

- 【推荐结果多样性】研讨
   - 题目：DGRec: Graph Neural Network for Recommendation with Diversified Embedding Generation
   - 作者：Liangwei Yang，Shengjie Wang，Yunzhe Tao，Jiankai Sun，Xiaolong Liu，Taiqing Wang
   - 单位：University of Illinois at Chicago Chicago, USA ByteDance Inc. Seattle, USA
   - 关键思路：
     - （1）子模邻居选择，对比所选子集Su与u所有邻居的集合Nu，通过找到两者间最相似的物品来评估Su的多样性，接着对相似值求和，以找到多样化的邻居。
     - （2）层注意力，为了缓解高阶连接的过平滑问题，通过L层GNN产生embedding，再通过层注意力模块得到最终的表示。
     - （3）损失重新加权，在训练模型期间，基于类别重新加权样本损失，物品属于流行类别，降低其权重，若属于长尾类别，则增加权重。
   - 摘要： Graph Neural Network (GNN) based recommender systems have been attracting more and more attention in recent years due to their excellent performance in accuracy. Representing user-item interactions as a bipartite graph, a GNN model generates user and item representations by aggregating embeddings of their neighbors. However, such an aggregation procedure often accumulates information purely based on the graph structure, overlooking the redundancy of the aggregated neighbors and resulting in poor diversity of the recommended list. In this paper, we propose diversifying GNN-based recommender systems by directly improving the embedding generation procedure. Particularly, we utilize the following three modules: submodular neighbor selection to find a subset of diverse neighbors to aggregate for each GNN node, layer attention to assign attention weights for each layer, and loss reweighting to focus on the learning of items belonging to long-tail categories. Blending the three modules into GNN, we present DGRec (Diversified GNN-based Recommender System) for diversified recommendation. Experiments on real-world datasets demonstrate that the proposed method can achieve the best diversity while keeping the accuracy comparable to state-of-the-art GNN-based recommender systems. We open source DGRec at https://github.com/YangLiangwei/DGRec.
### 2024.1.2
- 【蛋白质对相互作用预测】研讨
   - 题目：AFTGAN: prediction of multi-type PPI based on attention free transformer and graph attention network
   - 作者：Yanlei Kang1, Arne Elofsson 2, Yunliang Jiang3, Weihong Huang4,Minzhe Yu4 and Zhong Li
   - 单位： Bioinformatics
   - 解决问题：蛋白质对相互作用预测，蛋白质对相互作用类型预测
   - 关键思路：增加蛋白质的初始特征嵌入，用AFT代替transformer去提取序列特征，用GAT去代替GNN去提取蛋白质对的关系特征
   - 论文摘要：Motivation: Protein–protein interaction (PPI) networks and transcriptional regulatory networks are critical in regulating cells and their signaling. A thorough understanding of PPIs can provide more insights into cellular physiology at
normal and disease states. Although numerous methods have been proposed to predict PPIs, it is still challenging
for interaction prediction between unknown proteins. In this study, a novel neural network named AFTGAN was
constructed to predict multi-type PPIs. Regarding feature input, ESM-1b embedding containing much biological information for proteins was added as a protein sequence feature besides amino acid co-occurrence similarity and one-hot coding. An ensemble network was also constructed based on a transformer encoder containing an AFT module (performing the weight operation on vital protein sequence feature information) and graph attention network (extracting the relational features of protein pairs) for the part of the network framework. Results: The experimental results showed that the Micro-F1 of the AFTGAN based on three partitioning schemes (BFS, DFS and the random mode) on the SHS27K and SHS148K datasets was 0.685, 0.711 and 0.867, as well as 0.745, 0.819 and 0.920, respectively, all higher than that of other popular methods. In addition, the experimental comparisons confirmed the performance superiority of the proposed model for predicting PPIs of unknown proteins on the STRING dataset.

- 【解决同质性和异质性问题】研讨
   - 题目：Beyond Homophily and Homogeneity Assumption:Relation-Based Frequency Adaptive GraphNeural Networks
   - 作者：Lirong Wu , Graduate Student Member,  Haitao Lin , Bozhen Hu, Cheng Tan , Zhangyang Gao, Zicheng Liu , Graduate Student Member, IEEE, and Stan Z. Li
   - 单位：IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
   - 解决问题：解决在图神经网络中，在不同架构下同异质图中的节点分类问题。
   - 关键思路：提出Relation Self-Learning Module模块去构建同异构图下的通用方式；提出Relation-Based Frequency-Adaptive Architecture模块从频率的角度去解决不同同质性下的分类问题。
   - 摘要：Graph neural networks (GNNs) have been playing important roles in various graph-related tasks. However, most
existing GNNs are based on the assumption of homophily, so they cannot be directly generalized to heterophily settings where connected nodes may have different features and class labels. Moreover, real-world graphs often arise from highly entangled latent factors, but the existing GNNs tend to ignore this and simply denote the heterogeneous relations between nodes as binaryvalued homogeneous edges. In this article, we propose a novel relation-based frequency adaptive GNN (RFA-GNN) to handle both heterophily and heterogeneity in a unified framework. RFA-GNN first decomposes an input graph into multiple relation graphs, each representing a latent relation. More importantly, we provide detailed theoretical analysis from the perspective of spectral signal processing. Based on this, we propose a relation-based frequency adaptive mechanism that adaptively picks up signals of different frequencies in each corresponding relation space in the message-passing process. Extensive experiments on synthetic and real-world datasets show qualitatively and quantitatively that RFA-GNN yields truly encouraging results for both the heterophily and heterogeneity settings.
### 2024.1.9
- 【图分类】研讨
   - 题目： ARE MORE LAYERS BENEFICIAL TO GRAPH TRANSFORMERS?
   - 作者：Haiteng Zhao1∗, Shuming Ma2, Dongdong Zhang2, Zhi-Hong Deng1†, Furu Wei2
   - 单位：Peking University   Microsoft Research
   - 解决问题：为什么更多的自注意力层成为Graph Transformer的劣势，以及如何通过正确的模型设计来解决这些问题。
   - 关键思路：基于子结构的局部注意机制，促进对更深层次Graph Transformer的局部子结构特征的关注，并提高学习表示的表达能力。
   - 摘要：Despite that going deep has proven successful in many neural architectures, the existing graph transformers are relatively shallow. In this work, we explore whether more layers are beneficial to graph transformers, and find that current graph transformers suffer from the bottleneck of improving performance by increasing depth.Our further analysis reveals the reason is that deep graph transformers are limited by the vanishing capacity of global attention, restricting the graph transformer from focusing on the critical substructure and obtaining expressive features. To this end, we propose a novel graph transformer model named DeepGraph that explicitly employs substructure tokens in the encoded representation, and applies local attention on related nodes to obtain substructure based attention encoding.Our model enhances the ability of the global attention to focus on substructures and promotes the expressiveness of the representations, addressing the limitation of self-attention as the graph transformer deepens. Experiments show that our method unblocks the depth limitation of graph transformers and results in state-of-the-art performance across various graph benchmarks with deeper models.
### 2024.1.16
- 【图像对抗】研讨
   - 題目：StyLess: Boosting the Transferability of Adversarial Examples 
   - 作者：Kaisheng Liang Bin Xiao
   - 单位：The Hong Kong Polytechnic University 
   - 解決问题：解决以往使用代理模型攻击的方法迁移性不好，之前的迁移性攻击的效果不好，并且收到很大的限制。
   - 关键思路：以往生成的对抗噪声中，包含着原图的风格信息和内容信息，作者认为风格信息对于攻击的迁移性效果较差，因此作者构建的模型通过加入随机的混合风格，从而使从原图中获取的风格信息减少。
   - 摘要：Adversarial attacks can mislead deep neural networks (DNNs) by adding imperceptible perturbations to benign examples. The attack transferability enables adversarial examples to attack black-box DNNs with unknown archi- tectures or parameters, which poses threats to many real- world applications. We find that existing transferable at- tacks do not distinguish between style and content features during optimization, limiting their attack transferability. To improve attack transferability, we propose a novel attack method called style-less perturbation (StyLess). Specifi- cally, instead of using a vanilla network as the surrogate model, we advocate using stylized networks, which encode different style features by perturbing an adaptive instance normalization. Our method can prevent adversarial exam- ples from using non-robust style features and help gener- ate transferable perturbations. Comprehensive experiments show that our method can significantly improve the transfer- ability of adversarial examples. Furthermore, our approach is generic and can outperform state-of-the-art transferable attacks when combined with other attack techniques. 1 

- 【Transformer结构优化】研讨
  - 题目：Segmented Recurrent Transformer: An Efficient Sequence-to-Sequence Model
  - 作者：Yinghan Long, Sayeed Shafayet Chowdhury, Kaushik Roy
  - 单位：Purdue University
  - 解决问题：针对Transformer计算复杂度高的问题，尤其是文本摘要任务，改变编码器与解码器之间交叉注意力的计算方式有效降低复杂度。 
  - 关键思路：使用分段计算的交叉注意力，降低Transformer的计算复杂度；同时采用循环记忆的方式来弥补分段造成的全局损失。
  - 摘要：Transformers have shown dominant performance across a range of domains including language and vision. However, their computational cost grows quadratically with the sequence length, making their usage prohibitive for resource-constrained applications. To counter this, our approach is to divide the whole
sequence into segments and apply attention to the individual segments. We propose a segmented recurrent transformer (SRformer) that combines segmented (local) attention with recurrent attention. The loss caused by reducing the attention window length is compensated by aggregating information across segments with recurrent attention. SRformer leverages Recurrent Accumulate-and-Fire (RAF) neurons’ inherent memory to update the cumulative product of keys and values. The segmented attention and lightweight RAF neurons ensure the efficiency of the proposed transformer. Such an approach leads to models with sequential processing capability at a lower computation/memory cost. We apply the proposed method to T5 and BART transformers. The modified models are tested on summarization datasets including CNN-dailymail, XSUM, ArXiv, and MediaSUM. Notably, using segmented inputs of varied sizes, the proposed model achieves 6-22% higher ROUGE1 scores than a segmented transformer and outperforms other recurrent transformer approaches. Furthermore, compared to full attention, the proposed model reduces the computational complexity of cross attention by around 40%.
### 2024.3.06
- 【推荐公平性】研讨
  - 题目：Heterophily-Aware Fair Recommendation using Graph Convolutional Networks
  - 作者：Nemat Gholinejad, Mostafa Haghir Chehreghani
  - 单位：Department of Computer Engineering Amirkabir University of Technology (Tehran Polytechnic) Tehran, Iran
  - 解决问题：基于GNN的推荐算法面临着不公平和合流行性偏见的挑战。 
  - 关键思路：使用两个独立的组件来生成公平感知嵌入，1）公平感知注意力，在GNN的归一化过程中结合了点积，以减少节点度的影响；2）异质特征加权，以在聚合过程中为不同的特征分配不同的权重。
  - 摘要：In recent years, graph neural networks (GNNs) have become a popular tool to improve the accuracy and performance of recommender systems. Modern recommender systems are not only designed to serve the end users, but also to benefit other participants, such as items and items providers. These participants may have different or conflicting goals and interests, which raise the need for fairness and popularity bias considerations. GNN-based recommendation methods also face the challenges of unfairness and popularity bias and their normalization and aggregation processes suffer from these challenges. In this paper, we propose a fair GNN-based recommender system, called HetroFair, to improve items’ side fairness. HetroFair uses two separate components to generate fairness-aware embeddings: i) fairness-aware attention which incorporates dot product in the normalization process of GNNs, to decrease the effect of nodes’ degrees, and ii) heterophily feature weighting to assign distinct weights to different features during the aggregation process. In order to evaluate the effectiveness of HetroFair, we conduct extensive experiments over six real-world datasets. Our experimental results reveal that HetroFair not only alleviates the unfairness and popularity bias on the items’ side, but also achieves superior accuracy on the users’ side. Our implementation is publicly available at https://github.com/NematGH/HetroFair
 
- 【transformer和gnn】研讨
  - 题目：Can Transformer and GNN Help Each Other?
  - 作者：Peiyan Zhang
  - 单位：香港科技大学
  - 解决问题：transformer和gnn是否能相互帮助。 
  - 关键思路：transformer和gnn之间相互利用。
  - 摘要：Although Transformer has achieved great success in natural lan- guage process and computer vision, it has difficulty generalizing to medium and large scale graph data for two important reasons:(i) High complexity. (ii) Failing to capture the complex and entangled structure information. In graph representation learning, Graph Neural Networks(GNNs) can fuse the graph structure and node attributes but have limited receptive fields. Therefore, we question that can we combine Transformer and GNNs to help each other? In this paper, we propose a new model named TransGNN where the Transformer layer and GNN layer are used alternately to improve each other. Specifically, to expand the receptive field and disen- tangle the information aggregation from edges, we propose using Transformer to aggregate more relevant nodes’ information to im- prove the message passing of GNNs. Besides, to capture the graph structure information, we utilize positional encoding and make use of the GNN layer to fuse the structure into node attributes, which improves the Transformer in graph data. We also propose to sample the most relevant nodes for Transformer and two efficient samples update strategies to lower the complexity. At last, we theoretically prove that TransGNN is more expressive than GNNs only with extra linear complexity. The experiments on eight datasets corroborate the effectiveness of TransGNN on node and graph classification tasks.
### 2024.3.22
- 【蛋白质残基与核酸结合】研讨
  - 题目：GraphBind: protein structural context embedded rules learned by hierarchical graph neural networks for recognizing nucleic-acid-binding residues
  - 作者：Ying Xia1, Chun-Qiu Xia1, Xiaoyong Pan 1,* and Hong-Bin Shen 1,2,*  
  - 单位：1Institute of Image Processing and Pattern Recognition, Shanghai Jiao Tong University, and Key Laboratory of System Control and Information Processing, Ministry of Education of China, Shanghai 200240, China and 2School of Life Sciences and Biotechnology, Shanghai Jiao Tong University, Shanghai 200240, China 
  - 解决问题： 基于序列和结构信息的蛋白质残基与核酸结合预测。
  - 关键思路：首先基于目标残基的局部环境构建图。初始节点特征向量包括进化保守性、二级结构信息、其他生物物理化学特征和位置嵌入。位置嵌入是从定义结构背景中残基的空间关系的几何知识计算的。初始边缘特征向量也来自于几何知识。然后，我们构建了一个层次图神经网络学习潜在的局部模式结合残基预测。设计了边缘更新模块、节点更新模块和图更新模块，用于学习目标残基的高级几何和生物化学特征以及固定大小的嵌入。此外，门控递归单元GRU用于堆叠多个GNN块，其利用所有块的信息并避免梯度消失问题。
  - 摘要：Knowledge of the interactions between proteins and nucleic acids is the basis of understanding various biological activities and designing new drugs. How to accurately identify the nucleic-acid-binding residues remains a challenging task. In this paper, we propose an accurate predictor, GraphBind, for identifying nucleic-acid-binding residues on proteins based on an end-to-end graph neural network. Considering that binding sites often behave in highly conservative patterns on local tertiary structures, we first construct graphs based on the structural contexts of target residues and their spatial neighborhood. Then, hierarchical graph neural networks (HGNNs) are used to embed the latent local patterns of structural and bio-physicochemical characteristics for binding residue recognition. We comprehensively evaluate GraphBind on DNA/RNA benchmark datasets. The results demonstrate the superior performance of GraphBind than state-of-the-art methods. Moreover, GraphBind is extended to other ligand-binding residue prediction to verify its generalization capability. Web server of GraphBind is freely available at http://www.csbio.sjtu.edu.cn/bioinf/GraphBind/. 
 
- 【数据蒸馏】研讨
  - 题目：Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data
  - 作者：Xin Zheng1 , Miao Zhang2 , Chunyang Chen1 
  - 单位：1Monash University, Australia
  - 解决问题：解决现实中数据集过大，计算复杂度高等问题。
  - 关键思路：将传统图压缩的方法中得到的邻接矩阵简化为单位矩阵，从而消除掉图结构，并将节点的上下文信息和拓扑信息嵌入到特征矩阵中，减少计算的时间复杂度。
  - 摘要：Graph condensation, which reduces the size of a large-scale graph by synthesizing a small-scale condensed graph as its substitution, has immediate benefits for various graph learning tasks. However, existing graph condensation methods rely on the joint optimization of nodes and structures in the condensed graph, and overlook critical issues in effectiveness and generalization ability. In this paper, we advocate a new Structure-Free Graph Condensation paradigm, named SFGC, to distill a largescale graph into a small-scale graph node set without explicit graph structures, i.e., graph-free data. Our idea is to implicitly encode topology structure information into the node attributes in the synthesized graph-free data, whose topology is reduced to an identity matrix. Specifically, SFGC contains two collaborative components: (1) a training trajectory meta-matching scheme for effectively synthesizing smallscale graph-free data; (2) a graph neural feature score metric for dynamically evaluating the quality of the condensed data. Through training trajectory metamatching, SFGC aligns the long-term GNN learning behaviors between the largescale graph and the condensed small-scale graph-free data, ensuring comprehensive and compact transfer of informative knowledge to the graph-free data. Afterward, the underlying condensed graph-free data would be dynamically evaluated with the graph neural feature score, which is a closed-form metric for ensuring the excellent expressiveness of the condensed graph-free data. Extensive experiments verify the superiority of SFGC across different condensation ratios.
### 2024.3.29
- 【Graph Transformer】研讨
  - 题目：TRANSITIVITY-PRESERVING GRAPH REPRESENTATION LEARNING FOR BRIDGING LOCAL CONNECTIVITY AND ROLE-BASED SIMILARITY
  - 作者：Van Thuy Hoang and O-Joun Lee
  - 单位：Department of Artificial Intelligence, The Catholic University of Korea
  - 解决问题：①缺乏捕捉局部结构的相似性，现有的PE可以为各个节点的子结构分配可区分的表示，但它们并不总是反映子结构之间的结构相似性；②现有模型缺乏将局部和全局结构融合
    现有的研究大多只考虑局部连接性，而忽略了远程连接性和节点的作用
  - 关键思路：本研究提出了一种新颖的图转换器模型 UGT，用于学习局部和全局结构特征并将它们集成到统一的表示中。 UGT 使用基于结构相似性的采样捕获具有相似角色的节点之间的远程依赖关系，使用 k 跳邻域和结构身份发现局部连接性，并通过学习暗示这两个方面的节点之间的转移概率来统一它们。
  - 摘要：Graph representation learning (GRL) methods, such as graph neural networks and graph transformer models, have been successfully used to analyze graph-structured data, mainly focusing on node classification and link prediction tasks. However, the existing studies mostly only consider local connectivity while ignoring long-range connectivity and the roles of nodes. In this paper, we propose Unified Graph Transformer Networks (UGT) that effectively integrate local and global structural information into fixed-length vector representations. First, UGT learns local structure by identifying the local substructures and aggregating features of the k-hop neighborhoods of each node. Second, we construct virtual edges, bridging distant nodes with structural similarity to capture the long-range dependencies. Third, UGT learns unified representations through self-attention, encoding structural distance and p-step transition probability between node pairs. Furthermore,we propose a self-supervised learning task that effectively learns transition probability to fuse local and global structural features, which could then be transferred to other downstream tasks.Experimental results on real-world benchmark datasets over various downstream tasks showed that UGT significantly outperformed baselines that consist of state-of-the-art models. In addition, UGT reaches the expressive power of the third-order Weisfeiler-Lehman isomorphism test (3d-WL) in distinguishing non-isomorphic graph pairs.

- 【图神经网络同质性、过平滑问题】研讨
  - 题目：Continuous Graph Neural Networks
  - 作者：Louis-Pascal A. C. Xhonneux    Meng Qu     Jian Tang 
  - 单位：ICML2020
  - 解决问题：文章通过ODE方程去拟合图神经网络的消息传递过程，将离散的传递过程推进为连续的扩散过程，既保证了在数据集上的性能，又避免了在深层神经网络中的过平滑问题。
  - 关键思路：通过常微分分程去拟合消息传递过程，一步到位的得到最后的特征表示。
  - 摘要：This paper builds on the connection between graph neural networks and traditional dynamical systems. We propose continuous graph neural networks (CGNN), which generalise existing graph neural networks with discrete dynamics in that they can be viewed as a specific discretisation scheme. The key idea is how to characterise the continuous dynamics of node representations, i.e. the derivatives of node representations, w.r.t. time. Inspired by existing diffusion-based methods on graphs (e.g. PageRank and epidemic models on social networks), we define the derivatives as a combination of the current node representations, the representations of neighbors, and the initial values of the nodes. We propose and analyse two possible dynamics on graphs—including each dimension of node representations (a.k.a. the feature channel) change independently or interact with each other—both with theoretical justification. The proposed continuous graph neural networks are robust to over-smoothing and hence allow us to build deeper networks, which in turn are able to capture the long-range dependencies between nodes. Experimental results on the task of node classification demonstrate the effectiveness of our proposed approach over competitive baselines
### 2024.4.7
- 【大模型在图任务上的应用】研讨
  - 题目：TALK LIKE A GRAPH: ENCODING GRAPHS FOR LARGE LANGUAGE MODELS
  - 作者：Bahare Fatemi, Jonathan Halcrow, Bryan Perozzi
  - 单位：Google Research
  - 解决问题：如何将大模型应用在图相关的任务上
  - 关键思路：将图结构信息编码为文本输入给大模型
  - 摘要：Graphs are a powerful tool for representing and analyzing complex relationships in real-world applications such as social networks, recommender systems, and computational finance. Reasoning on graphs is essential for drawing inferences about the relationships between entities in a complex system, and to identify hidden patterns and trends. Despite the remarkable progress in automated reasoning with natural text, reasoning on graphs with large language models (LLMs) remains an understudied problem. In this work, we perform the first comprehensive study of encoding graph-structured data as text for consumption by LLMs. We show that LLM performance on graph reasoning tasks varies on three fundamental levels: (1) the graph encoding method, (2) the nature of the graph task itself, and (3) interestingly, the very structure of the graph considered. These novel results provide valuable insight on strategies for encoding graphs as text. Using these insights we illustrate how the correct choice of encoders can boost performance on graph reasoning tasks inside LLMs by 4.8% to 61.8%, depending on the task.

- 【基于掩码的图自监督学习】研讨
  - 题目：GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner
  - 作者：Zhenyu Hou，Yufei He，Yukuo Cen，Xiao Liu
  - 单位：Tsinghua University, China
  - 解决问题：提出了一种基于掩码的图自编码器，解决输入特征的可判别性对模型性能的影响。
  - 关键思路：提出了两种解码策略，多视图随机重掩码解码和潜在表示预测，目的是对图形SSL的特征重构施加正则化。
  - 摘要：Graph self-supervised learning (SSL), including contrastive and generative approaches, offers great potential to address the fundamental challenge of label scarcity in real-world graph data. Among both sets of graph SSL techniques, the masked graph autoencoders (e.g., GraphMAE)—one type of generative methods—have recently produced promising results. The idea behind this is to reconstruct the node features (or structures)—that are randomly masked from the input—with the autoencoder architecture. However, the performance of masked feature reconstruction naturally relies on the discriminability of the input features and is usually vulnerable to disturbance in the features. In this paper, we present a masked self-supervised learning framework1 GraphMAE2 with the goal of overcoming this issue. The idea is to impose regularization on feature reconstruction for graph SSL. Specifically, we design the strategies of multi-view random re-mask decoding and latent representation prediction to regularize the feature reconstruction. The multi-view random re-mask decoding is to introduce randomness into reconstruction in the feature space, while the latent representation prediction is to enforce the reconstruction in the embedding space. Extensive experiments show that GraphMAE2 can consistently generate top results on various public datasets, including at least 2.45% improvements over state-of-the-art baselines on ogbnPapers100M with 111M nodes and 1.6B edges.
