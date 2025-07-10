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
### 2024.4.12
- 【基于多模态图分类】研讨
  - 题目：Multi-Modal Representation Learning for Molecular Property Prediction: Sequence, Graph, Geometry
  - 作者：Zeyu Wang， Tianyi Jiang， Jinhuan Wang， Qi Xuan*
  - 解决问题：通过多种模态表征分子，获得一个信息更丰富的图表征。
  - 关键思路：对分子的三种模态进行编码，然后对齐两两模态。
  - 摘要：Molecular property prediction refers to the task of labeling molecules with some biochemical properties, playing a pivotal role in the drug discovery and design process. Recently, with the advancement of machine learning, deep learning-based molecular property prediction has emerged as a solution to the resource-intensive nature of traditional methods, garnering significant attention. Among them, molecular representation learning is the key factor for molecular property prediction performance. And there are lots of sequence-based, graph-based, and geometry-based methods that have been proposed. However, the majority of existing studies focus solely on one modality for learning molecular representations, failing to comprehensively capture molecular characteristics and information. In this paper, a novel multi-modal representation learning model, which integrates the sequence, graph, and geometry characteristics, is proposed for molecular property prediction, called SGGRL. Specifically, we design a fusion layer to fusion the representation of different modalities. Furthermore, to ensure consistency across modalities, SGGRL is trained to maximize the similarity of representations for the same molecule while minimizing similarity for different molecules. To verify the effectiveness of SGGRL, seven molecular datasets, and several baselines are used for evaluation and comparison. The experimental results demonstrate that SGGRL consistently outperforms the baselines in most cases. This further underscores the capability of SGGRL to comprehensively capture molecular information. Overall, the proposed SGGRL model showcases its potential to revolutionize molecular property prediction by leveraging multi-modal representation learning to extract diverse and comprehensive molecular insights. Our code is released at https://github.com/Vencent-Won/SGGRL.

- 【图像对抗】研讨
  - 题目：On Evaluating Adversarial Robustness of Large Vision-Language Models
  - 作者：Yunqing Zhao∗1, Tianyu Pang∗†2, Chao Du†2, Xiao Yang3, Chongxuan Li4, Ngai-Man Cheung†1, Min Lin2 
  - 单位：Singapore University of Technology and Design, Sea AI Lab, Singapore, Tsinghua University 4Renmin University of China
  - 解决问题：目前大模型非常流行，关于大模型的论文也越来越多，本文主要探究了大模型的鲁棒性问题。
  - 关键思路：本文对text2img以及img2text进行攻击，使大模型无法正确识别，分别使用到基于查询的黑盒攻击和白盒攻击分别对不同的大模型进行攻击，均取得较好的高级效果。
  - 摘要：Large vision-language models (VLMs) such as GPT-4 have achieved unprecedented performance in response generation, especially with visual inputs, enabling more creative and adaptable interaction than large language models such as ChatGPT. Nonetheless, multimodal generation exacerbates safety concerns, since adversaries may successfully evade the entire system by subtly manipulating the most vulner- able modality (e.g., vision). To this end, we propose evaluating the robustness of open-source large VLMs in the most realistic and high-risk setting, where ad- versaries have only black-box system access and seek to deceive the model into returning the targeted responses. In particular, we first craft targeted adversarial examples against pretrained models such as CLIP and BLIP, and then transfer these adversarial examples to other VLMs such as MiniGPT-4, LLaVA, UniDiffuser, BLIP-2, and Img2Prompt. In addition, we observe that black-box queries on these VLMs can further improve the effectiveness of targeted evasion, resulting in a sur- prisingly high success rate for generating targeted responses. Our findings provide a quantitative understanding regarding the adversarial vulnerability of large VLMs and call for a more thorough examination of their potential security flaws before deployment in practice. Our project page: yunqing-me.github.io/AttackVLM/. 
### 2024.4.19
- 【层次图和过平滑】研讨
  - 题目：Self-Similar Graph Neural Network for Hierarchical Graph Learning
  - 作者：Zheng Zhang
  - 解决问题：层次图和过平滑问题
  - 关键思路：利用分形思想和层次图思想解决过平滑问题
  - 摘要：Many real-world networks, such as graph-structured molecules or social networks, exhibit latent hierarchi- cal structures at many different resolutions. Existing hierarchical graph neural networks (GNNs) mainly fo- cus on modifying graph global pooling regions into par- titioned clusters, while keeping the convolutional lay- ers unchanged. However, these approaches may suffer from a loss of expressive power in learned representa- tions due to the uncontrolled growth of the neighbor- hood, leading to a failure in capturing true hierarchies. Furthermore, many real-world hierarchical graphs pos- sess an underlying fractal structure, which is crucial to unraveling the formation mechanism of networks. Un- fortunately, existing hierarchical GNNs often overlook this important aspect of graph hierarchy. To tackle these challenges, this paper proposes a generic frame- work for hierarchical network representation learning. We propose the Self-Similar Graph Neural Network (SS- GNN), which leverages localized representations by ex- cluding redundant nodes and edges. At each resolution of the coarsened map, SS-GNN extracts both intra- and inter-cluster embeddings to preserve the discriminative power of the model with a theoretical guarantee. To ex- ploit the graph fractal structure, we introduce a novel module for measuring self-similarity between resolutions and a characterized objective function for automatic ad- justment of model parameters. We demonstrate the strength of our proposed framework through extensive experiments on 13 real-world datasets by outperforming the state-of-the-art GNN models.

- 【推荐公平性】研讨
  - 题目：Invariant Collaborative Filtering to Popularity Distribution Shif
  - 作者：An Zhang, Jingnan Zheng, Xiang Wang, Yancheng Yuan, Tat-Seng Chua
  - 单位：National University of Singapore Sea-NExT Joint Lab, National University of Singapore, University of Science and Technology of China
  - 解决思路：提出一个新的学习框架，不变的协同过滤（InvCF）, 集成了不变性和解纠缠原则。包括四个模块：偏好编码器，流行度编码器，表征增强模块和表征解耦模块。首先，流行度编码器和偏好编码器分别学习从流行度统计和历史交互到潜在表示空间的推理映射。然后根据提出的原则实现了增强和解纠缠模块，以驱动表示学习。
  - 摘要：Collaborative Filtering (CF) models, despite their great success, suffer from severe performance drops due to popularity distribution shifts, where these changes are ubiquitous and inevitable in real-
world scenarios. Unfortunately, most leading popularity debiasing strategies, rather than tackling the vulnerability of CF models to varying popularity distributions, require prior knowledge of the test distribution to identify the degree of bias and further learn the popularity-entangled representations to mitigate the bias. Consequently, these models result in signifcant performance benefts in the target test set, while dramatically deviating the recommendation from users’ true interests without knowing the popularity distribution in advance. In this work, we propose a novel learning framework, Invariant Collaborative Filtering (InvCF), to discover disentangled representations that faithfully reveal the latent preference and popularity semantics without making any assumption about the popularity distribution. At its core is the distillation of unbiased preference representations (i.e., user preference on item property), which are invariant to the change of popularity semantics, while fltering out the popularity feature that is unstable or outdated. Extensive experiments on fve benchmark datasets and four evaluation settings (i.e., synthetic long-tail, unbiased, temporal split, and out-of-distribution evaluations) demonstrate that InvCF outperforms the state-of-the-art baselines in terms of popularity generalization ability on real recommendations. Visualization studies shed light on the advantages of InvCF for disentangled representation learning. Our codes are available at https://github.com/anzhang314/InvCF.
### 2024.4.28
- 【蛋白质相互作用】研讨
  - 题目：MAPE-PPI: Towards effective and efficient Protein-Protein Interaction Prediction viaMicroenvironment-aware protein embedding
  - 作者：Lirong Wu1,2, Yijun Tian3, Yufei Huang1, Siyuan Li1, Haitao Lin1, Nitesh V Chawla3, Stan Z. Li1,†
  - 解决问题：蛋白质相互作用的预测
  - 关键思路：为了将两种蛋白质模式都考虑在内，通过氨基酸残基的序列和结构背景来定义残基的微环境，其描述了周围的化学性质和几何特征；提出了微环境感知的蛋白质嵌入PPI预测（MPAE-PPI），其通过足够大的微环境“词汇表”将微环境编码成化学上有意义的离散代码（即，码本）。此外，提出了一种新的预训练策略，即屏蔽码本建模（MCM）。通过随机掩蔽码本并重构输入来捕获不同微环境之间的依赖性。
  - 摘要：Protein-Protein Interactions (PPIs) are fundamental in various biological processes and play a key role in life activities. The growing demand and cost of experimental PPI assays require computational methods for efficient PPI prediction. While existing methods rely heavily on protein sequence for PPI prediction, it is the protein structure that is the key to determine the interactions. To take both protein modalities into account, we define the microenvironment of an amino acid residue by its sequence and structural contexts, which describe the surrounding chemical properties and geometric features. In addition, microenvironments defined in previous work are largely based on experimentally assayed physicochemical properties, for which the “vocabulary” is usually extremely small. This makes it difficult to cover the diversity and complexity of microenvironments. In this paper, we propose Microenvironment-Aware Protein Embedding for PPI prediction (MPAE-PPI), which encodes microenvironments into chemically meaningful discrete codes via a sufficiently large microenvironment “vocabulary” (i.e.,codebook). Moreover, we propose a novel pre-training strategy, namely Masked Codebook Modeling (MCM), to capture the dependencies between different microenvironments by randomly masking the codebook and reconstructing the input. With the learned microenvironment codebook, we can reuse it as an off-the-shelf tool to efficiently and effectively encode proteins of different sizes and functions for large-scale PPI prediction. Extensive experiments show that MAPE-PPI can scale to PPI prediction with millions of PPIs with superior trade-offs between effectiveness and computational efficiency than the state-of-the-art competitors. Codes are available at: https://github.com/LirongWu/MAPE-PPI.
### 2024.5.10
- 【神经网络】研讨
  - 题目：KAN: Kolmogorov–Arnold Networks
  - 作者：Ziming Liu1,4∗ Yixuan Wang2 Sachin Vaidya1 Fabian Ruehle3,4 James Halverson3,4 Marin Soljaciˇ c´ 1,4 Thomas Y. Hou2 Max Tegmark1,4
  - 单位：Massachusetts Institute of Technology 2 California Institute of Technology 3 Northeastern University 4 The NSF Institute for Artificial Intelligence and Fundamental Interactions
  - 解决问题：对神经网络模型的重新思考，将MLP中的激活函数参数化为可学习的激活函数。
  - 关键思路：将激活函数参数化，采用B样条去拟合每一个待学习的激活函数。在训练过程中通过稀疏化和熵惩罚去保留重要的激活函数，在通过符号化优化得到最后的拟合方程。
  - 摘要：Inspired by the Kolmogorov-Arnold representation theorem, we propose KolmogorovArnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs).While MLPs have fixed activation functions on nodes (“neurons”), KANs have learnable activation unctions on edges (“weights”). KANs have no linear weights at all – every weight parameter is replaced by a univariate function parametrized as a spline. We show that this seemingly simple change makes KANs outperform MLPs in terms of accuracy and interpretability. For accuracy, much smaller KANs can achieve comparable or better accuracy than much larger MLPs in data fitting and PDE solving. Theoretically and empirically, KANs possess faster neural scaling laws than MLPs. For interpretability, KANs can be intuitively visualized and can easily interact with human users. Through two examples in mathematics and physics, KANs are shown to be useful “collaborators” helping scientists (re)discover mathematical and physical laws. In summary, KANs are promising alternatives for MLPs, opening opportunities for further improving today’s deep learning models which rely heavily on MLPs.

- 【大语言模型敏感信息处理】研讨
  - 题目：CAN SENSITIVE INFORMATION BE DELETED FROMLLMS? OBJECTIVES FOR DEFENDING AGAINST EXTRACTION ATTACKS
  - 作者：Vaidehi Patil    Peter Hase    Mohit Bansal
  - 单位：UNC Chapel Hill
  - 解决问题：考虑大语言模型中敏感信息是否被删除干净，提出了三种获取敏感信息的攻击方式以及对应的防御方式。
  - 关键思路：就算对LLMs提前做了RLHF或者模型编辑的处理，但LLMs在迭代更新过程中其预测概率分布中仍有可能包含敏感信息的概率，因此可以考虑从中间隐藏层入手去对敏感信息进行攻防的研究。
  - 摘要：Pretrained language models sometimes possess knowledge that we do not wish them to, including memorized personal information and knowledge that could be used to harm people. To mitigate these safety and informational issues, we propose an attack-and-defense framework for studying the task of deleting sensitive information directly from model weights. We study direct edits to model weights because (1) this approach should guarantee that particular deleted information is never extracted by future prompt attacks, and (2) it should protect against whitebox attacks, which is necessary for making claims about safety/privacy in a setting where publicly available model weights could be used to elicit sensitive information. Our threat model assumes that an attack succeeds if the answer to a sensitive question is located among a set of B generated candidates, based on scenarios where the information would be insecure if the answer is among B candidates. Experimentally, we show that even state-of-the-art model editing methods such as ROME struggle to truly delete factual information from models like GPT-J, as our whitebox and blackbox attacks can recover “deleted” information from an edited model 38% of the time. These attacks leverage two key observations: (1) that traces of deleted information can be found in intermediate model hidden states, and (2) that applying an editing method for one question may not delete information across rephrased versions of the question. Finally, we provide new defense methods that protect against some extraction attacks, but we do not find a single universally effective defense method. Our results suggest that truly deleting sensitive information is a tractable but difficult problem, since even relatively low attack success rates have potentially severe mplications for the deployment of language models in a world where individuals enjoy ownership of their personal data, a right to privacy, and safety from harmful model outputs
### 2024.5.17
- 【图分类】研讨
  - 题目：Multi-View Robust Graph Representation Learning for Graph Classification
  - 作者：Guanghui Ma1 , Chunming Hu1,2,3,∗ , Ling Ge1 and Hong Zhang4
  - 单位：School of Computer Science and Engineering, Beihang University, Beijing, China
  - 解决问题：现有图分类模型存在的语义偏差和置信度崩溃问题
  - 关键思路：提出 MGRL 来同时提高图分类模型的性能和鲁棒性。 MGRL 利用实例视图一致性表示学习方法和类视图判别性表示学习方法来缓解语义偏差和置信度崩溃问题。
  - 摘要：The robustness of graph classification models plays an essential role in providing highly reliable applications. Previous studies along this line primarily focus on seeking the stability of the model in terms of overall data metrics (e.g., accuracy) when facing data perturbations, such as removing edges. Empirically, we find that these graph classification models also suffer from semantic bias and confidence collapse issues, which substantially hinder their applicability in real-world scenarios. To address these issues, we present MGRL, a multi-view representation learning model for graph classification tasks that achieves robust results. Firstly, we proposes an instance-view consistency representation learning method, which utilizes multi-granularity contrastive learning technique to perform semantic constraints on instance representations at both the node and graph levels, thus alleviating the semantic bias issue. Secondly, we proposes a class-view discriminative representation learning method, which employs the prototype-driven class distance optimization technique to adjust intra- and inter-class distances, thereby mitigating the confidence collapse issue. Finally, extensive experiments and visualizations on eight benchmark dataset demonstrate the effectiveness of MGRL.

- 【统一处理同质图和异质图方法】研讨
  - 题目：PC-Conv: Unifying Homophily and Heterophily with Two-fold Filtering
  - 作者：Bingheng Li，Erlin Pan，Zhao Kang
  - 单位：University of Electronic Science and Technology of China
  - 解决问题：基于几乎所有图中异质性和同质性节点共存的特点，提出了一种双重过滤机制来同时进行异质性和同质性信息的聚合，从而解决异质性和同质性不能同时处理的问题。
  - 关键思路：提出了一种双重过滤机制来提取异构图中的同质性和同构图中的异质性。扩展了图热方程来执行远距离全局信息的异亲聚合。所得滤波器用Possion-Charlier (PC)多项式精确地逼近。引入了一个图卷积PC-Conv及其实例化PCNet用于节点分类任务。
  - 摘要：ecently, many carefully crafted graph representation learning methods have achieved impressive performance on either strong heterophilic or homophilic graphs, but not both. Therefore, they are incapable of generalizing well across real-world graphs with different levels of homophily. This is attributed to their neglect of homophily in heterophilic graphs, and vice versa. In this paper, we propose a two-fold filtering mechanism to extract homophily in heterophilic graphs and vice versa. In particular, we extend the graph heat equation to perform heterophilic aggregation of global information from a long distance. The resultant filter can be exactly approximated by the Possion-Charlier (PC) polynomials. To further exploit information at multiple orders, we introduce a powerful graph convolution PC-Conv and its instantiation PCNet for the node classification task. Compared with state-of-the-art GNNs, PCNet shows competitive performance on wellknown homophilic and heterophilic graphs. Our implementation is available at https://github.com/uestclbh/PC-Conv.
### 2024.5.24
- 【图分类】研讨
  - 题目：Dual-Channel Learning Framework for Drug-Drug Interaction Prediction via Relation-Aware Heterogeneous Graph Transformer
  - 作者：Xiaorui Su1,2,3, Pengwei Hu1,2, Zhu-Hong You4, Philip S. Yu3, Lun Hu1,2,*
  - 单位：Xinjiang Technical Institutes of Physics and Chemistry, Urumqi, China
  - 解决问题：现有DDI模型，没有考虑到来自异构图的信息，并且长距离的分子可能会导致过挤压
  - 关键思路：使用双通道，生物信息学知识图，以及分子，并且在对节点之间的关系进行编码，然后将其加入到transformer的注意力分数计算当中，这样计算出来的分数不仅考虑到了节点和节点之间的关系，还考虑到了关系之间的关系，并且使用transformer能有效解决过挤压的问题。
  - 摘要：Identifying novel drug-drug interactions (DDIs) is a crucial task in pharmacology, as the interference between pharmacological substances can pose serious medical risks. In recent years, several network-based techniques have emerged for predicting DDIs. However, they primarily focus on local structures within DDI-related networks, often overlooking the significance of indirect connections between pairwise drug nodes from a global perspective. Additionally, effectively handling heterogeneous information present in both biomedical knowledge graphs and drug molecular graphs remains a challenge for improved performance of DDI prediction. To address these limitations, we propose a Transformer based relatIon-aware Graph rEpresentation leaRning framework (TIGER) for DDI prediction. TIGER leverages the Transformer architecture to effectively exploit the structure of heterogeneous graph, which allows it direct learning of long dependencies and high-order structures. Furthermore, TIGER incorporates a relation-aware self-attention mechanism, capturing a diverse range of semantic relations that exist betweenpairs of nodes in heterogeneous graph. In addition to these advancements, TIGER enhances predictive accuracy by modeling DDI prediction task using a dual-channel network, where drug molecular graph and biomedical knowledge graph arefed into two respective channels. By incorporating embeddings obtained at graph and node levels, TIGER can benefit from structural properties of drugs as well as rich contextual information provided by biomedical knowledge graph. Extensive experiments conducted on three real-world datasets demonstrate the effectiveness of TIGER in DDI prediction. Furthermore, case studies highlight its ability to provide a deeper understanding of underlying mechanisms of DDIs.

- 【大模型RAG】研讨
  - 题目：RECOMP: IMPROVING RETRIEVAL-AUGMENTED LMS WITH COMPRESSION AND SELECTIVE AUGMENTATION
  - 作者：Fangyuan Xu1, Weijia Shi2, Eunsol Choi1
  - 单位：Department of Computer Science 1The University of Texas at Austin 2University ofWashington
  - 解决问题：大语言模型外挂知识库导致token长度过度增加，以及文本过长会忽略关键信息。
  - 关键思路：训练一个用于压缩的模型，对检索到的知识库内容进行抽取或摘要式的压缩。
  - 摘要：Retrieving documents and prepending them in-context at inference time improves performance of language model (LMs) on a wide range of tasks. However, these documents, often spanning hundreds of words, make inference substantially more expensive. We propose compressing the retrieved documents into textual summaries prior to in-context integration. This not only reduces the computational costs but also relieves the burden of LMs to identify relevant information in long retrieved documents. We present two compressors – an extractive compressor which selects useful sentences from retrieved documents and an abstractive compressor which generates summaries by synthesizing information from multiple documents. Both compressors are trained to improve LMs’ performance on end tasks when the generated summaries are prepended to the LMs’ input, while keeping the summary concise. If the retrieved documents are irrelevant to the input or offer no additional information to LM, our compressor can return an empty string, implementing selective augmentation. We evaluate our approach on language modeling task and open domain question answering task. We achieve a compression rate of as low as 6% with minimal loss in performance for both tasks, significantly outperforming the off-the-shelf summarization models. We show that our compressors trained for one LM can transfer to other LMs on the language modeling task and provide summaries largely faithful to the retrieved documents.
### 2024.5.31
- 【图像对抗】研讨
  - 题目：Boosting Adversarial Transferability by Block Shuffle and Rotation
  - 作者：Kunyu Wang1 , Xuanran He2 , Wenxuan Wang1 , Xiaosen Wang3* 
  - 单位：Chinese University of Hong Kong, 2Nanyang Technological University, Huawei Singularity Security Lab
  - 解决问题：为了提高梯度攻击的攻击成功率，以及攻击的迁移性。
  - 关键思路：对初试图片进行随机的裁剪，旋转，以及洗牌操作，然后对该图像进行梯度攻击，最后将梯度进行平均。
  - 摘要：Adversarial examples mislead deep neural networks with imperceptible perturbations and have brought significant threats to deep learning. An important aspect is their transferability, which refers to their ability to deceive other models, thus enabling attacks in the black-box setting. Though various methods have been proposed to boost transferability, the performance still falls short compared with white-box attacks. In this work, we observe that existing input transformation based attacks, one of the main stream transfer-based attacks, result in different attention heatmaps on various models, which might limit the transferability. We also find that breaking the intrinsic relation of the image can disrupt the attention heatmap of the original image. Based on this finding, we propose a novel input transformation based attack called block shuffle and rotation (BSR). Specifically, BSR splits the input image into several blocks, then randomly shuffles and rotates these blocks to construct a set of new images for gradient calculation. Empirical evaluations on the ImageNet dataset demonstrate that BSR could achieve significantly better transferability than the existing input transformation based methods under single-model and ensemble-model settings. Combining BSR with the current input transformation method can further improve the transferability, which significantly outperforms the state-of-the-art methods. 
### 2024.7.25
- 【大模型】研讨
  - 题目：On the Anatomy of Attention
  - 作者：Nikhil Khatri∗ Tuomas Laakkonen∗ Jonathon Liu∗ Vincent Wang-Maścianica∗‡
  - 单位：Compositional Intelligence, Quantinuum 17 Beaumont St., Oxford OX1 2NA, UK
  - 解决问题：引入一种基于范畴论的图示形式主义来系统地描述和推理机器学习模型，特别是关注机制，并通过图形变换和分类来识别和探索不同注意力机制之间的关系。
  - 关键思路：利用范畴论中的字符串图示来表示深度学习架构，通过图形变换和重写规则来比较和推理不同的注意力机制，从而在保持计算细节的同时提供直观的架构表示。
  - 摘要：We introduce a category-theoretic diagrammatic formalism in order to systematically relate and reason about machine learning models. Our diagrams present architectures intuitively but without loss of essential detail, where natural relationships between models are captured by graphical transformations, and important differences and similarities can be identified at a glance. In this paper, we focus on attention mechanisms: translating folklore into mathematical derivations, and constructing a taxonomy of attention variants in the literature. As a first example of an empirical investigation underpinned by our formalism, we identify recurring anatomical components of attention, which we exhaustively recombine to explore a space of variations on the attention mechanism.

- 【大模型长文本处理】研讨
  - 题目：GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models
  - 作者：Shilong Li∗1, Yancheng He∗1, Hangyu Guo∗1, Xingyuan Bu∗†‡1, Ge Bai1, Jie Liu2,3,Jiaheng Liu1, Xingwei Qu4, Yangguang Li3, Wanli Ouyang2,3, Wenbo Su1, Bo Zheng1
  - 单位：1Alibaba Group, 2The Chinese University of Hong Kong, 3Shanghai AI Laboratory 4University of Manchester
  - 解决问题：大语言模型难以处理长文本，超出文本窗口大小。
  - 关键思路：将文本构建为图结构，使用agent阅读图，整合信息。
  - 摘要：Long-context capabilities are essential for large language models (LLMs) to tackle complex and long-input tasks. Despite numerous efforts made to optimize LLMs for long contexts, challenges persist in robustly processing long inputs. In this paper, we introduce GraphReader, a graph-based agent system designed to handle long texts by structuring them into a graph and employing an agent to explore this graph autonomously. Upon receiving a question, the agent first undertakes a step-by-step analysis and devises a rational plan. It then invokes a set of predefined functions to read node content and neighbors, facilitating a coarse-to-fine exploration of the graph. Throughout the exploration, the agent continuously records new insights and reflects on current circumstances to optimize the process until it has gathered sufficient information to generate an answer. Experimental results on the LV-Eval dataset reveal that GraphReader, using a 4k context window, consistently outperforms GPT-4-128k across context lengths from 16k to 256k by a large margin. Additionally, our approach demonstrates superior performance on four challenging single-hop and multi-hop benchmarks.
### 2024.8.1
- 【针对RAG型LLM代理的首个后门攻击】研讨
  - 题目：AGENTPOISON: Backdoor Attack against Retrieval-Augmented Generation-based LLM Agents
  - 作者：Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song, Bo Li
  - 单位：University of Chicago, University of Illinois, Urbana-Champaign, University of Wisconsin, Madison, University of California, Berkeley
  - 解决问题：揭示了当前LLM代理在安全性和可信度方面的漏洞，特别是依赖于未验证知识库所带来的风险。
  - 关键思路：提出了一种名为AGENTPOISON的新型红队攻击方法，通过毒化LLM代理的长期记忆或RAG知识库来植入后门触发器，从而在用户指令包含优化后的触发器时，诱导代理检索恶意演示并生成对抗性操作。
  - 摘要：Large language model (LLM) agents have demonstrated remarkable performance across various applications, primarily due to their advanced capabilities in reasoning, utilizing external knowledge and tools, calling APIs, and executing actions to interact with environments. Current agents typically utilize a memory module or a retrieval-augmented generation (RAG) mechanism, retrieving past knowledge and instances with similar embeddings from knowledge bases to inform task planning and execution. However, the reliance on unverified knowledge bases raises significant concerns about their safety and trustworthiness. To uncover such vulnerabilities, we propose a novel red teaming approach AGENTPOISON, the first backdoor attack targeting generic and RAG-based LLM agents by poisoning their long-term memory or RAG knowledge base. In particular, we form the trigger generation process as a constrained optimization to optimize backdoor triggers by mapping the triggered instances to a unique embedding space, so as to ensure that whenever a user instruction contains the optimized backdoor trigger, the malicious demonstrations are retrieved from the poisoned memory or knowledge base with high probability. In the meantime, benign instructions without the trigger will still maintain normal performance. Unlike conventional backdoor attacks, AGENTPOISON requires no additional model training or fine-tuning, and the optimized backdoor trigger exhibits superior transferability, in-context coherence, and stealthiness. Extensive experiments demonstrate AGENTPOISON's effectiveness in attacking three types of real-world LLM agents: RAG-based autonomous driving agent, knowledge-intensive QA agent, and healthcare EHRAgent. We inject the poisoning instances into the RAG knowledge base and long-term memories of these agents, respectively, demonstrating the generalization of AGENTPOISON. On each agent, AGENTPOISON achieves an average attack success rate of ≥80% with minimal impact on benign performance (≤1%) with a poison rate <0.1%. The code and data is available at https://github.com/BillChan226/AgentPoison.

- 【大语言模型长上下文处理】研讨
  -  题目：HUMAN-LIKE EPISODIC MEMORY FOR INFINITE CONTEXT LLMs
  -  作者：Zafeirios Fountas, Martin A Benfeghoul, Adnan Oomerjee, Fenia Christopoulou, Gerasimos Lampouras, Haitham Bou-Ammar, Jun Wang
  -  单位：Huawei Noah's Ark Lab, London, UK; University College London, UK
  -  解决问题：解决大语言模型在处理长上下文时的局限性，提升其在长序列任务中的表现。
  -  关键思路：通过结合人类情景记忆和事件认知的关键方面，EM-LLM利用贝叶斯惊喜和图论边界细化在线组织标记序列为连贯的情景事件，并通过两阶段记忆过程进行高效检索。
  -  摘要：Large language models (LLMs) have shown remarkable capabilities, but still struggle with processing extensive contexts, limiting their ability to maintain coherence and accuracy over long sequences. In contrast, the human brain excels at organising and retrieving episodic experiences across vast temporal scales, spanning a lifetime. In this work, we introduce EM-LLM, a novel approach that integrates key aspects of human episodic memory and event cognition into LLMs, enabling them to effectively handle practically infinite context lengths while maintaining computational efficiency. EM-LLM organises sequences of tokens into coherent episodic events using a combination of Bayesian surprise and graph-theoretic boundary refinement in an on-line fashion. When needed, these events are retrieved through a two-stage memory process, combining similarity-based and temporally contiguous retrieval for efficient and human-like access to relevant information. Experiments on the LongBench dataset demonstrate EM-LLM's superior performance, outperforming the state-of-the-art InfLLM model with an overall relative improvement of 4.3% across various tasks, including a 33% improvement on the PassageRetrieval task. Furthermore, our analysis reveals strong correlations between EM-LLM's event segmentation and human-perceived events, suggesting a bridge between this artificial system and its biological counterpart. This work not only advances LLM capabilities in processing extended contexts but also provides a computational framework for exploring human memory mechanisms, opening new avenues for interdisciplinary research in AI and cognitive science.
### 2024.9.13
- 【安全遗忘：防御越狱攻击的有效且可泛化的解决方案】研讨
  - 题目：Safe Unlearning: A Surprisingly Effective and Generalizable Solution to Defend Against Jailbreak Attacks
  - 作者：Zhexin Zhang, Junxiao Yang, Pei Ke, Shiyao Cui, Chujie Zheng, Hongning Wang, Minlie Huang
  - 单位：The Conversational AI (CoAI) group, DCST, Tsinghua University
  - 解决问题：如何有效防御大语言模型（LLMs）在安全对齐后仍易受越狱攻击的问题。
  - 关键思路：通过直接遗忘模型中的有害知识，而不是仅仅识别有害查询，来防止模型生成有害响应，即使面对未见过的越狱提示。
  - 摘要：Large Language Models (LLMs) are known to be vulnerable to jailbreak attacks, even after safety alignment. An important observation is that, while different types of jailbreak attacks can generate significantly different queries, they mostly result in similar responses that are rooted in the same harmful knowledge(e.g., detailed steps to make a bomb). Therefore, we conjecture that directly unlearn the harmful knowledge in the LLM can be a more effective way to defend against jailbreak attacks than the mainstream supervised fine-tuning(SFT) based approaches.Our extensive experiments confirmed our insight and suggested surprising generalizability of our unlearning-based approach: using only 20 raw harmful questions without any jailbreak prompt during training, our solution reduced the Attack Success Rate(ASR) in Vicuna-7B on out-of-distribution(OOD) harmful questions wrapped with various complex jailbreak prompts from 82.6% to 7.7%. This significantly outperforms Llama2-7B-Chat, which is fine-tuned on about 0.1M safety alignment samples but still has an ASR of 21.9% even under the help of an additional safety system prompt. Further analysis reveals that the generalization ability of our solution stems from the intrinsic relatedness among harmful responses across harmful questions(e.g., response patterns, shared steps and actions, and similarity among their learned representations in the LLM). Our code is available at https://github.com/thu-coai/SafeUnlearning.

- 【视觉语言模型的构建与理解】研讨
  -  题目：Building and better understanding vision-language models: insights and future directions
  -  作者：Hugo Laurencon*, Andrés Marafioti, Victor Sanh, Léo Tronchon*
  -  单位：Hugging Face
  -  解决问题：在视觉语言模型（VLMs）的开发中，缺乏对数据、架构和训练方法的共识，导致模型性能的提升受限。
  -  关键思路：通过使用开放数据集和简单管道构建强大的VLM，并创建一个大规模的文档理解数据集Docmaxix，以提升文档理解任务的性能。
  -  摘要：The field of vision-language models (VLMs), which take images and texts as inputs and output texts, is rapidly evolving and has yet to reach consensus on several key aspects of the development pipeline, including data, architecture, and training methods. This paper can be seen as a tutorial for building a VLM. We begin by providing a comprehensive overview of the current state-of-the-art approaches, highlighting the strengths and weaknesses of each, addressing the major challenges in the field, and suggesting promising research directions for underexplored areas. We then walk through the practical steps to build Idefics3-8B, a powerful VLM that significantly outperforms its predecessor Idefics2-8B, while being trained efficiently, exclusively on open datasets, and using a straightforward pipeline. These steps include the creation of Docmaxix, a dataset for improving document understanding capabilities, which is 240 times larger than previously available datasets. We release the model along with the datasets created for its training.
### 2024.9.29
- 【结合凝聚子图的图对比学习】研讨
  - 题目：Safe Unlearning: A Surprisingly Effective and Generalizable Solution to Defend Against Jailbreak Attacks
  - 作者：Zhexin Zhang, Junxiao Yang, Pei Ke, Shiyao Cui, Chujie Zheng, Hongning Wang, Minlie Huang
  - 单位：The Conversational AI (CoAI) group, DCST, Tsinghua University
  - 解决问题：如何有效防御大语言模型（LLMs）在安全对齐后仍易受越狱攻击的问题。
  - 关键思路：通过直接遗忘模型中的有害知识，而不是仅仅识别有害查询，来防止模型生成有害响应，即使面对未见过的越狱提示。
  - 摘要：Graph contrastive learning (GCL) has emerged as a state-of-the-art strategy for learning representations of diverse graphs including social and biomedical networks. GCL widely uses stochastic graph topology augmentation, such as uniform node dropping, to generate augmented graphs. However, such stochastic augmentations may severely damage the intrinsic properties of a graph and deteriorate the following representation learning process. We argue that incorporating an awareness of cohesive subgraphs during the graph augmentation and learning processes has the potential to enhance GCL performance. To this end, we propose a novel unified framework called CTAug, to seamlessly integrate cohesion awareness into various existing GCL mechanisms. In particular, CTAug comprises two specialized modules: topology augmentation enhancement and graph learning enhancement. The former module generates augmented graphs that carefully preserve cohesion properties, while the latter module bolsters the graph encoder's ability to discern subgraph patterns. Theoretical analysis shows that CTAug can strictly improve existing GCL mechanisms. Empirical experiments verify that CTAug can achieve state-of-the-art performance for graph representation learning, especially for graphs with high degrees.

- 【无监督深度图结构学习】研讨
  -  题目：Towards Unsupervised Deep Graph Structure Learning
  -  作者：Yixin Liu, Yu Zheng, Daokun Zhang, Hongxu Chen, Hao Peng, Shirui Pan
  -  单位：Monash University, La Trobe University, Monash Suzhou Research Institute, University of Technology Sydney, Beihang University
  -  解决问题：现有深度图结构学习方法依赖于监督信号（如标签），导致对标签的依赖、边分布偏差和应用任务限制等问题，本文提出了一种无监督的图结构学习范式。
  -  关键思路：提出了一种新颖的无监督图结构学习框架SUBLIME，利用自监督对比学习，通过生成“锚图”作为学习目标，并设计了一种自举机制来持续更新锚图，以提供稳定的监督信号。
  -  摘要：In recent years, graph neural networks(GNNs) have emerged as a successful tool in a variety of graph-related applications. How-ever, the performance of GNNs can be deteriorated when noisy connections occur in the original graph structures; besides, the de-pendence on explicit structures prevents GNNs from being applied to general unstructured scenarios. To address these issues, recently emerged deep graph structure learning(GSL) methods propose to jointly optimize the graph structure along with GNN under the supervision of a node classification task. Nonetheless, these meth-ods focus on a supervised learning scenario, which leads to several problems, i.e., the reliance on labels, the bias of edge distribution,and the limitation on application tasks. In this paper, we propose a more practical GSL paradigm, unsupervised graph structure learning,where the learned graph topology is optimized by data itself with-out any external guidance(i.e., labels). To solve the unsupervised GSL problem, we propose a novel StrUcture Bootstrapping con-trastive LearnIng fraMEwork(SUBLIME for abbreviation) with the aid of self-supervised contrastive learning. Specifically, we generate a learning target from the original data as an“anchor graph”, and use a contrastive loss to maximize the agreement between the an-chor graph and the learned graph. To provide persistent guidance,we design a novel bootstrapping mechanism that upgrades the anchor graph with learned structures during model learning. We also design a series of graph learners and post-processing schemes to model the structures to learn. Extensive experiments on eight benchmark datasets demonstrate the significant effectiveness of our proposed SUBLIME and high quality of the optimized graphs.
### 2024.10.15
- 【序列建模层的表达能力提升】研讨
  - 题目：Learning to (Learn at Test Time): RNNs with Expressive Hidden States
  - 作者：Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin
  - 单位：1 Stanford University. 2 UC San Diego. 3 UC Berkeley. 4 Meta AI.
  - 解决问题：现有RNN层在长上下文中表现受限，因为其隐藏状态的表示能力不足，而自注意力机制虽然表现良好但计算复杂度高。
  - 关键思路：提出一种新的序列建模层，称为Test-Time Training (TTT) 层，其中隐藏状态本身是一个机器学习模型，更新规则是自监督学习的一步。通过这种方式，TTT层可以在测试时进行训练，从而提高长上下文中的表达能力。
  - 摘要：Self-attention performs well in long context but has quadratic complexity. Existing RNN layers have linear complexity, but their performance in long context is limited by the expressive power of their hidden state. We propose a new class of sequence modeling layers with linear complexity and an expressive hidden state. The key idea is to make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning. Since the hidden state is updated by training even on test sequences, our layers are called Test-Time Training(TTT) layers. We consider two instantiations: TTT-Linear and TTT-MLP, whose hidden state is a linear model and a two-layer MLP respectively. We evaluate our instantiations at the scale of 125M to 1.3 B parameters, comparing with a strong Transformer and Mamba, a modern RNN. Both TTT-Linear and TTT-MLP match or exceed the baselines. Similar to Transformer, they can keep reducing perplexity by conditioning on more tokens, while Mamba cannot after 16k context. With preliminary systems optimization, TTT-Linear is already faster than Transformer at 8k context and matches Mamba in wall-clock time. TTT-MLP still faces challenges in memory I/O, but shows larger potential in long context, pointing to a promising direction for future research.

- 【大语言模型长上下文扩展方法】研讨
  -  题目：Training-Free Long-Context Scaling of Large Language Models
  -  作者：Chenxin An， Fei Huang， Jun Zhang， Shansan Gong，Xipeng Qiu，Chang Zhou，Lingpeng Kong  
  -  单位：1The University of Hong Kong， 2Alibaba Group， 3Fudan University.
  -  解决问题：提出了一种无需训练的方法，使大语言模型能够在不进行额外训练的情况下支持超过100k token的长上下文窗口。
  -  关键思路：通过将长序列的注意力计算分解为基于块的模块，Dual Chunk Attention (DCA) 方法有效地捕捉了同一块内（Intra-Chunk）和不同块间（Inter-Chunk）的相对位置信息，并与 Flash Attention 无缝集成。
  -  摘要：The ability of Large Language Models(LLMs)to process and generate coherent text is markedly weakened when the number of input tokens ex-ceeds their pretraining length. Given the expen-sive overhead of finetuning large-scale models with longer sequences, we propose Dual Chunk Attention(DCA), which enables LLAMA2 70B to support context windows of more than 100k tokens without continual training. By decom-posing the attention computation for long se-quences into chunk-based modules, DCA man-ages to effectively capture the relative positional information of tokens within the same chunk(Intra-Chunk) and across distinct chunks(Inter-Chunk), as well as integrates seamlessly with Flash Attention. In addition to its impressive extrapolation capability, DCA achieves perfor-mance on practical long-context tasks that is com-parable to or even better than that of finetuned models. When compared with proprietary mod-els, our training-free 70B model attains 94% of the performance of gpt-3.5-16k, indicating it is a viable open-source alternative. All code and data used in this work are released at https://github.com/HKUNLP/ChunkLlama.
### 2024.10.22
- 【基于图神经网络的启发式学习统一框架】研讨
  - 题目：Heuristic Learning with Graph Neural Networks: A Unified Framework for Link Prediction
  - 作者：Juzheng Zhang，Lanning Wei，Zhen Xu，Quanming Yao
  - 单位：Department of Electronic Engineering, Tsinghua University, Beijing, China
  - 解决问题：提出了一种统一框架，将局部和全局启发式方法结合，以解决传统启发式方法在泛化到不同图上的挑战，并克服图神经网络在链接预测任务中无法有效利用拓扑信息的问题。
  - 关键思路：通过矩阵乘法统一局部和全局启发式方法，提出Heuristic Learning Graph Neural Network (HL-GNN)，采用层内传播和层间连接，能够在不增加计算复杂度的情况下达到较深的层数，并自适应地平衡节点特征和拓扑信息。
  - 摘要：Link prediction is a fundamental task in graph learning, inherently shaped by the topology of the graph. While traditional heuristics are grounded in graph topology, they encounter challenges in gen-eralizing across diverse graphs. Recent research efforts have aimed to leverage the potential of heuristics, yet a unified formulation accommodating both local and global heuristics remains undiscov-ered. Drawing insights from the fact that both local and global heuristics can be represented by adjacency matrix multiplications,we propose a unified matrix formulation to accommodate and gener-alize various heuristics. We further propose the Heuristic Learning Graph Neural Network(HL-GNN) to efficiently implement the for-mulation. HL-GNN adopts intra-layer propagation and inter-layer connections, allowing it to reach a depth of around 20 layers with lower time complexity than GCN. HL-GNN is proven to be more expressive than heuristics and conventional GNNs, and it can adap-tively trade-off between node features and topological information.Extensive experiments on the Planetoid, Amazon, and OGB datasets underscore the effectiveness and efficiency of HL-GNN. It outper-forms existing methods by a large margin in prediction performance.Additionally, HL-GNN is several orders of magnitude faster than heuristic-inspired methods while requiring only a few trainable pa-rameters. The case study further demonstrates that the generalized heuristics and learned weights are highly interpretable. The code is available at https://github.com/LARS-research/HL-GNN1.

- 【社交机器人检测方法】研讨
  -  题目：SEBOT: Structural Entropy Guided Multi-View Contrastive Learning for Social Bot Detection
  -  作者：Yingguang Yang, Hao Peng, Qi Wu, Buyun He, Renyu Yang, Zhifeng Hao, Yong Liao
  -  单位：University of Science and Technology of China, Beihang University
  -  解决问题：解决了现有基于图的社交机器人检测方法无法充分利用隐藏图信息且易受对抗性机器人行为影响的问题。
  -  关键思路：SEBOT通过结构熵优化图结构和子图粒度，设计超越同质性假设的消息传递编码器，并采用多视图对比学习增强检测性能。
  -  摘要：Recent advancements in social bot detection have been driven by the adoption of Graph Neural Networks. The social graph, con-structed from social network interactions, contains benign and bot accounts that influence each other. However, previous graph-based detection methods that follow the transductive message-passing paradigm may not fully utilize hidden graph information and are vulnerable to adversarial bot behavior. The indiscriminate message passing between nodes from different categories and communi-ties results in excessively homogeneous node representations, ul-timately reducing the effectiveness of social bot detectors. In this paper, we propose SEBOT, a novel multi-view graph-based con-trastive learning-enabled social bot detector. In particular, we use structural entropy as an uncertainty metric to optimize the entire graph's structure and subgraph-level granularity, revealing the im-plicitly existing hierarchical community structure. And we design an encoder to enable message passing beyond the homophily as-sumption, enhancing robustness to adversarial behaviors of social bots. Finally, we employ multi-view contrastive learning to maxi-mize mutual information between different views and enhance the detection performance through multi-task learning. Experimental results demonstrate that our approach significantly improves the performance of social bot detection compared with SOTA methods.
### 2024.11.1
- 【AGENT WORKFLOW MEMORY】研讨
  - 题目：AGENT WORKFLOW MEMORY
  - 作者：Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, Graham Neubig
  - 单位：Carnegie Mellon University, Massachusetts Institute of Technology
  - 解决问题：旨在解决当前基于语言模型的代理在处理具有复杂行动轨迹的长视距任务时表现不佳的问题，尤其是在任务上下文或环境发生变化时缺乏鲁棒性。
  - 关键思路：提出了一种名为Agent Workflow Memory (AWM)的方法，通过从过去的经验中提取可重用的任务工作流，并将这些工作流选择性地提供给代理以指导后续的行动生成。AWM可以灵活地应用于离线和在线场景，从而提高代理在复杂任务中的表现和适应性。
  - 摘要：Despite the potential of language model-based agents to solve real-world tasks such as web navigation, current methods still struggle with long-horizon tasks with complex action trajectories. In contrast, humans can flexibly solve com-plex tasks by learning reusable task workflows from past experiences and using them to guide future actions. To build agents that can similarly benefit from this process, we introduce Agent Workflow Memory(AWM), a method for inducing commonly reused routines, i.e., workflows, and selectively providing workflows to the agent to guide subsequent generations. AWM flexibly applies to both of-fline and online scenarios, where agents induce workflows from training examples beforehand or from test queries on the fly. We experiment on two major web navi-gation benchmarks-Mind2Web and WebArena-that collectively cover 1000+tasks from 200+ domains across travel, shopping, and social media, among others.AWM substantially improves the baseline results by 24.6% and 51.1% relative success rate on Mind2Web and WebArena while reducing the number of steps taken to solve WebArena tasks successfully. Furthermore, online AWM robustly generalizes in cross-task, website, and domain evaluations, surpassing baselines from 8.9 to 14.0 absolute points as train-test task distribution gaps widen.https://github.com/zorazrw/agent-workflow-memory

- 【长上下文LLM加速方法】研讨
  -  题目：Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction
  -  作者：Zhenmei Shi, Yifei Ming, Xuan-Phi Nguyen, Yingyu Liang, Shafiq Joty
  -  单位：Salesforce AI Research
  -  解决问题：本文解决了大型语言模型在处理长上下文输入时计算资源消耗大和延迟高的问题。
  -  关键思路：GemFilter通过在早期层识别相关token并将其作为过滤器来选择和压缩输入token，从而显著减少后续处理的上下文长度。
  -  摘要：Large Language Models(LLMs) have demonstrated remarkable capabilities in handling long context inputs, but this comes at the cost of increased computational resources and latency. Our research introduces a novel approach for the long context bottleneck to accelerate LLM infer-ence and reduce GPU memory consumption. Our research demonstrates that LLMs can identify relevant tokens in the early layers before generating answers to a query. Leveraging this insight,we propose an algorithm that uses early layers of an LLM as filters to select and compress input tokens, significantly reducing the context length for subsequent processing. Our method, Gem-Filter, demonstrates substantial improvements in both speed and memory efficiency compared to existing techniques, such as standard attention and SnapKV/H2O. Notably, it achieves a 2.4x speedup and 30% reduction in GPU memory usage compared to SOTA methods. Evalua-tion on the Needle in a Haystack task shows that GemFilter significantly outperforms standard attention, SnapKV and demonstrates comparable performance on the LongBench challenge.GemFilter is simple, training-free, and broadly applicable across different LLMs. Crucially,it provides interpretability by allowing humans to inspect the selected input sequence. These findings not only offer practical benefits for LLM deployment, but also enhance our understand-ing of LLM internal mechanisms, paving the way for further optimizations in LLM design and inference. Our code is available at https://github.com/SalesforceAIResearch/GemFilter.
### 2024.11.5
- 【图神经网络中的oversquashing问题】研讨
  - 题目：Oversquashing in GNNs through the lens of information contraction and graph expansion
  - 作者：Pradeep Kr. Banerjee, Kedar Karhadkar, Yu Guang Wang, Uri Alon, Guido Montúfar
  - 单位：MPI MiS, UCLA, SJTU, CMU
  - 解决问题：解决了图神经网络(GNNs)在处理长距离节点交互任务时出现的“oversquashing”现象，即信息在多层传播过程中被过度压缩的问题。
  - 关键思路：提出了一种基于信息收缩的分析框架来理解oversquashing现象，并设计了一种局部图重连算法G-RLEF，通过随机局部边翻转来缓解oversquashing，该算法受到扩展图构造的启发，能够在不改变节点度数和不破坏图连通性的情况下提高图的扩展性。
  - 摘要：Graph neural networks(GNNs) provide a powerful framework for modeling complex structural and relational data from diverse domains ranging from biological and social networks to knowledge graphs. Despite their empirical successes, the expressivity of GNNs is limited particularly for tasks involving long-range node interactions. For such tasks, increasing the number of layers can lead to a phenomenon called “oversquashing”. We present a framework for analyzing oversquashing based on information contraction. Our analysis is guided by a model of reliable computation due to von Neumann that lends a new insight into oversquashing as signal quenching in noisy computation graphs. Building on this, we propose a graph rewiring algorithm aimed at alleviating oversquashing. Our algorithm employs a random local edge flip primitive motivated by an expander graph construction. We compare the spectral expansion properties of our algorithm with that of an existing curvature-based non-local rewiring strategy. Synthetic experiments show that while our algorithm in general has a slower rate of expansion, it is overall computationally cheaper, preserves the node degrees exactly and never disconnects the graph.

- 【图卷积网络特征平滑度控制】研讨
  -  题目：Learning to Control the Smoothness of Graph Convolutional Network Features
  -  作者：Shih-Hsin Wang, Justin Baker, Cory Hauck, Bao Wang
  -  单位：University of Utah, Oak Ridge National Laboratory and University of Tennessee
  -  解决问题：解决了深度图卷积网络中节点特征过度平滑的问题，提出了一种新的策略来调整节点特征的平滑度，以适应数据和任务，从而提高节点分类的准确性。
  -  关键思路：该方法通过几何关系建立输入和输出之间的关系，并在学习过程中引入可调节的平滑项，以控制节点特征的平滑度。
  -  摘要：The pioneering work of Oono and Suzuki [ICLR, 2020] and Cai and Wang [arXiv:2006.13318] initializes the analysis of the smoothness of graph convolutional network (GCN) features. Their results reveal an intricate empirical correlation between node classification accuracy and the ratio of smooth to non-smooth feature components. However, the optimal ratio that favors node classification is unknown, and the non-smooth features of deep GCN with ReLU or leaky ReLU activation function diminish. In this paper, we propose a new strategy to let GCN learn node features with a desired smoothness—adapting to data and tasks—to enhance node classification. Our approach has three key steps: (1) We establish a geometric relationship between the input and output of ReLU or leaky ReLU. (2) Building on our geometric insights, we augment the message-passing process of graph convolutional layers (GCLs) with a learnable term to modulate the smoothness of node features with computational efficiency. (3) We investigate the achievable ratio between smooth and non-smooth feature components for GCNs with the augmented message-passing scheme. Our extensive numerical results show that the augmented message-passing schemes significantly improve node classification for GCN and some related models.
### 2024.11.12
- 【知识数据库投毒攻击】研讨
  - 题目：PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models
  - 作者：Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia
  - 单位：Pennsylvania State University, Illinois Institute of Technology
  - 解决问题：提出了PoisonedRAG，一种针对检索增强生成（RAG）系统的新颖知识污染攻击，旨在通过向RAG系统的知识数据库中注入少量恶意文本，诱导大型语言模型生成攻击者选择的答案。
  - 关键思路：PoisonedRAG通过将恶意文本分解为两个子文本，分别满足检索条件和生成条件，从而在知识数据库中注入恶意文本，诱导大型语言模型生成攻击者期望的答案。
  - 摘要：Large language models (LLMs) have achieved remarkable success due to their exceptional generative capabilities. Despite their success, they also have inherent limitations such as a lack of up-to-date knowledge and hallucination. Retrieval-Augmented Generation (RAG) is a state-of-the-art technique to mitigate these limitations. The key idea of RAG is to ground the answer generation of an LLM on external knowledge retrieved from a knowledge database. Existing studies mainly focus on improving the accuracy or efficiency of RAG, leaving its security largely unexplored. We aim to bridge the gap in this work. We find that the knowledge database in a RAG system introduces a new and practical attack surface. Based on this attack surface, we propose PoisonedRAG, the first knowledge corruption attack to RAG, where an attacker could inject a few malicious texts into the knowledge database of a RAG system to induce an LLM to generate an attacker-chosen target answer for an attacker-chosen target question. We formulate knowledge corruption attacks as an optimization problem, whose solution is a set of malicious texts. Depending on the background knowledge (e.g., black-box and white-box settings) of an attacker on a RAG system, we propose two solutions to solve the optimization problem, respectively. Our results show PoisonedRAG could achieve a 90% attack success rate when injecting five malicious texts for each target question into a knowledge database with millions of texts. We also evaluate several defenses and our results show they are insufficient to defend against PoisonedRAG, highlighting the need for new defenses.

- 【LLM位置编码】研讨
  -  题目：YaRN: Efficient Context Window Extension of Large Language Models
  -  作者：Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole
  -  单位：Nous Research, EleutherAI, University of Geneva
  -  解决问题：解决了大型语言模型在预训练后无法泛化到超过其训练序列长度的问题。
  -  关键思路：YaRN通过结合NTK-by-parts插值和动态缩放技术，高效地扩展了基于RoPE的位置嵌入模型的上下文窗口，减少了训练数据和步骤的需求。
  -  摘要：Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models fail to generalize past the sequence length they were trained on. We present YaRN(Yet another RoPE extensioN method), a compute-efficient method to extend the context window of such models, requiring 10x less tokens and 2.5x less training steps than previous methods. Using YaRN, we show that LLaMA models can effectively utilize and extrapolate to context lengths much longer than their original pre-training would allow, while also surpassing previous the state-of-the-art at context window extension. In addition, we demonstrate that YaRN exhibits the capability to extrapolate beyond the limited context of a fine-tuning dataset. The models fine-tuned using YaRN has been made available and reproduced online up to 128k context length at https://github.com/jquesnelle/yarn.
### 2024.11.21
- 【信息感知的无监督多重图结构学习】研讨
  - 题目：Beyond Redundancy: Information-aware Unsupervised Multiplex Graph Structure Learning
  - 作者：Zhixiang Shen, Shuo Wang, Zhao Kang
  - 单位：School of Computer Science and Engineering,University of Electronic Science and Technology of China, Chengdu, Sichuan, China
  - 解决问题：解决了现有无监督多重图学习方法忽视图结构的可靠性，导致在处理现实世界数据时性能下降的问题。
  - 关键思路：InfoMGF通过图结构精炼来消除不相关的噪声，并同时最大化视图共享和视图独特的任务相关信息，从而实现非冗余的多重图融合。
  - 摘要：Unsupervised Multiplex Graph Learning (UMGL) aims to learn node representations on various edge types without manual labeling. However, existing research overlooks a key factor: the reliability of the graph structure. Real-world data often exhibit a complex nature and contain abundant task-irrelevant noise, severely compromising UMGL's performance. Moreover, existing methods primarily rely on contrastive learning to maximize mutual information across different graphs, limiting them to multiplex graph redundant scenarios and failing to capture view-unique task-relevant information. In this paper, we focus on a more realistic and challenging task: to unsupervisedly learn a fused graph from multiple graphs that preserve sufficient task-relevant information while removing task-irrelevant noise. Specifically, our proposed Information-aware Unsupervised Multiplex Graph Fusion framework (InfoMGF) uses graph structure refinement to eliminate irrelevant noise and simultaneously maximizes view-shared and view-unique task-relevant information, thereby tackling the frontier of non-redundant multiplex graph. Theoretical analyses further guarantee the effectiveness of InfoMGF. Comprehensive experiments against various baselines on different downstream tasks demonstrate its superior performance and robustness. Surprisingly, our unsupervised method even beats the sophisticated supervised approaches. The source code and datasets are available at https://github.com/zxlearningdeep/InfoMGF.

- 【图池化方法】研讨
  -  题目：ReiPool: Reinforced Pooling Graph Neural Networks for Graph-Level Representation Learning
  -  作者：Hao Peng, Chuan Zhou, Zhao Li, Shan Xue, Jian Yang
  -  单位：Beihang University, Chinese Academy of Sciences, Zhejiang Lab, Macquarie University
  -  解决问题：解决了现有图池化方法在图结构粗化过程中破坏全局结构、无法自适应选择池化层数的问题。
  -  关键思路：ReiPool通过自适应混合图粗化策略保留全局结构，并引入多智能体强化学习自适应执行图粗化过程，生成最具代表性的粗化图，同时设计图级对比学习增强全局信息保留。
  -  摘要：Graph pooling technique as the essential component of graph neural networks has gotten increasing attention recently and it aims to learn graph-level representations for the whole graph. Besides, graph pooling is important in graph classification and graph generation tasks. However, current graph pooling methods mainly coarsen a sequence of small-sized graphs to capture hier-archical structures, potentially resulting in the deterioration of the global structure of the original graph and influencing the quality of graph representations. Furthermore, these methods artificially se-lect the number of graph pooling layers for different graph datasets rather than considering each graph individually. In reality, the structure and size differences among graphs necessitate a specific number of graph pooling layers for each graph. In this work, we propose reinforced pooling graph neural networks via adaptive hybrid graph coarsening networks. Specifically, we design a hybrid graph coarsening strategy to coarsen redundant structures of the original graph while retaining the global structure. In addition,we introduce multi-agent reinforcement learning to adaptively perform the graph coarsening process to extract the most repre-sentative coarsened graph for each graph, enhancing the quality of graph-level representations.Finally,we design graph-level contrast to improve the preservation of global information in graph-level representations. Extensive experiments with rich baselines on six benchmark datasets show the effectiveness of ReiPool1.
### 2024.11.28
- 【ID定制方法】研讨
  - 题目：PuLID: Pure and Lightning ID Customization via Contrastive Alignment
  - 作者：Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, Qian He
  - 单位：ByteDance Inc.
  - 解决问题：解决了现有ID定制方法在插入ID信息时对原始模型行为产生干扰的问题。
  - 关键思路：PuLID通过引入一个Lightning T2I分支，并结合对比对齐损失和精确的ID损失，最小化对原始模型的干扰，同时确保高ID保真度。
  - 摘要：We propose Pure and Lightning ID customization(PuLID), a novel tuning-free ID customization method for text-to-image generation. By incorporating a Lightning T2I branch with a standard diffusion one, PuLID introduces both contrastive alignment loss and accurate ID loss, minimizing disruption to the original model and ensuring high ID fidelity. Experiments show that PuLID achieves superior performance in both ID fidelity and editability. Another attractive property of PuLID is that the image elements(e.g., background, lighting, composition, and style) before and after the ID insertion are kept as consistent as possible. Codes and models will be available at https://github.com/ToTheBeginning/PuLID.

- 【大语言模型长上下文推理加速】研讨
  -  题目：SQUEEZED ATTENTION: Accelerating Long Context Length LLM Inference
  -  作者：Coleman Hooper*, Sehoon Kim*, Hiva Mohammadzadeh, Monishwaran Maheswaran, June Paik, Michael W. Mahoney, Kurt Keutzer, Amir Gholami
  -  单位：UC Berkeley, FuriosaAI, ICSI, LBNL
  -  解决问题：解决了大语言模型在长上下文推理中因输入提示长度增加导致的计算和内存效率低下的问题。
  -  关键思路：SQUEEZED ATTENTION通过离线K-means聚类固定上下文的关键值对，并在推理时通过比较查询与聚类中心快速识别重要键值对，从而减少计算和内存开销。
  -  摘要：Emerging Large Language Model(LLM) applications require long input prompt in order to perform complex downstream tasks like document analysis and code generation. For these long context length applications, the length of the input prompt poses a significant challenge in terms of inference efficiency since the inference costs increase linearly with sequence length. However, for many of these applications, much of the context in the prompt is fixed across different user inputs, thereby providing the opportunity to perform offline optimizations to process user inputs quickly, as they are received. In this work, we propose SQUEEZED ATTENTION as a mechanism to accelerate LLM applications where a large portion of the input prompt is fixed. To accomplish this, we first leverage K-means clustering offline to group the keys for the fixed context based on semantic similarity and represent each cluster with a single centroid value. During inference, we compare query tokens from the user input with the centroids to predict which of the keys from the fixed context are semantically relevant and need to be loaded during inference. We then compute exact attention using only these important keys from the fixed context, thereby reducing bandwidth and computational costs. We also extend our method to use a hierarchical centroid lookup to identify important keys,which can reduce the complexity of attention from linear to logarithmic with respect to the fixed context length.To realize our method's efficiency benefits, we implement optimized Triton kernels for centroid comparison and sparse FlashAttention with important keys, achieving more than 4x speedups during both the prefill and generation phases for long-context inference. Furthermore, we have extensively evaluated our method on various long-context benchmarks including LongBench, where it achieves a 3.1x reduction in KV cache budget without accuracy loss.For applications where small accuracy degradation is allowed, we can achieve up to an 8x reduction with less than 0.5 point accuracy gap for the LLaMA-2-7B-32K, LWM-Text-Chat-1M, and Longchat-7B-v1.5-32K models. Our code is available at https://github.com/SqueezeAILab/SqueezedAttention.
### 2024.12.5
- 【图神经网络解释】研讨
  - 题目：TOWARD HUMAN-INTERPRETABLE EXPLANATIONS IN A UNIFIED FRAMEWORK FOR GNNS
  - 作者：Anonymous authors
  - 单位：Double-blind review
  - 解决问题：解决了现有图神经网络(GNNs)解释方法缺乏人类可解释性和统一框架的问题。
  - 关键思路：UO-Explainer利用预定义的图lets和轨道作为解释单元，将GNN权重分解为轨道单元，以提取模型级别的类特定图模式和实例级别的重要子图。
  - 摘要：As Graph Neural Networks(GNNs) are increasingly applied across various do-mains, explainability has become a critical factor for real-world applications. Exist-ing post-hoc explainability methods primarily focus on estimating the importance of edges, nodes, or subgraphs in the input graph to identify substructures crucial for predictions. However, these methods often lack human interpretability and do not provide a unified framework that incorporates both model-level and instance-level explanations. In this context, we propose leveraging a set of graphlets-small,connected, non-isomorphic induced subgraphs widely used in various scientific fields-and their associated orbits as human-interpretable units to decompose GNN predictions. Domain experts can select the most relevant graphlets as inter-pretable units and request unified explanations based on these units. To address this problem, we introduce UO-Explainer, the Unified and Orbit-based Explainer for GNNs, which utilizes predefined orbits that are generalizable and universal across graph domains as interpretable units. Our model decomposes GNN weights into orbit units to extract class-specific graph patterns(model-level) and to identify important subgraphs within individual data instances for prediction(instance-level).Extensive experimental results demonstrate that UO-Explainer outperforms ex-isting baselines in providing meaningful and interpretable explanations across both synthetic and real-world datasets. Our code and datasets are available at https://anonymous.4open.science/r/uoexplainer-F12C.

- 【社交机器人检测方法】研讨
  -  题目：LMBot: Distilling Graph Knowledge into Language Model for Graph-less Deployment in Twitter Bot Detection
  -  作者：Zijian Cai, Zhaoxuan Tan, Zhenyu Lei, Zifeng Zhu, Hongrui Wang, Qinghua Zheng, Minnan Luo
  -  单位：Xi'an Jiaotong University, University of Notre Dame, University of Virginia
  -  解决问题：解决了基于图的Twitter机器人检测方法在推理时依赖多跳邻居信息，导致API查询耗时和采样偏差的问题。
  -  关键思路：LMBot通过将图神经网络的知识蒸馏到语言模型中，使得语言模型能够在没有图结构的情况下进行推理，从而避免了数据依赖和采样偏差问题。
  -  摘要：As malicious actors employ increasingly advanced and widespread bots to disseminate misinformation and manipulate public opinion, the detection of Twitter bots has become a crucial task. Though graph-based Twitter bot detection methods achieve state-of-the-art performance, we find that their inference depends on the neighbor users multi-hop away from the targets, and fetching neighbors is time-consuming and may introduce sampling bias. At the same time, our experiments reveal that after finetuning on Twitter bot detection task, pretrained language models achieve competitive performance while do not require a graph structure during deploy-ment. Inspired by this finding, we propose a novel bot detection framework LMBot1 that distills the graph knowledge into language models(LMs) for graph-less deployment in Twitter bot detection to combat data dependency challenge. Moreover, LMBot is compati-ble with graph-based and graph-less datasets. Specifically, we first represent each user as a textual sequence and feed them into the LM for domain adaptation. For graph-based datasets, the output of LM serves as input features for the GNN, enabling LMBot to optimize for bot detection and distill knowledge back to the LM in an iterative, mutually enhancing process. Armed with the LM,we can perform graph-less inference with graph knowledge, which resolves the graph data dependency and sampling bias issues. For datasets without graph structure, we simply replace the GNN with an MLP, which also shows strong performance. Our experiments demonstrate that LMBot achieves state-of-the-art performance on four Twitter bot detection benchmarks. Extensive studies also show that LMBot is more robust, versatile, and efficient compared to ex-isting graph-based Twitter bot detection methods.
### 2024.12.12
- 【DeCoRe: 通过对比检索头来解码以减轻幻觉】研讨
  - 题目：DECORE: DECODING BY CONTRASTING RETRIEVAL HEADS TO MITIGATE HALLUCINATIONS
  - 作者：Aryo Pradipta Gema, K Chen, JinK Ahmed Abdulaal, V Tom Diethe, Philip Teare, Beatrice Alex, Pasquale Minervini, A Amrutha Saseendran
  - 单位：University of Edinburgh, United Kingdom, AMiniml.AI, United Kingdom, Centre for AI, Data Science & Artificial Intelligence, R&D, AstraZeneca, United Kingdom, University College London, United Kingdom
  - 解决问题：提出了一种新的解码策略，旨在通过对比检索头的输出来减少大型语言模型在生成内容时的幻觉现象。
  - 关键思路：该方法通过掩盖检索头并对比基础模型和被掩盖模型的输出来动态减少幻觉，利用条件熵作为指导来放大上下文中的信息。
  - 摘要：Large Language Models (LLMs) often hallucinate, producing unfaithful or factually incorrect outputs by misrepresenting the provided context or incorrectly recalling internal knowledge. Recent studies have identified specific attention heads within the Transformer architecture, known as retrieval heads, responsible for extracting relevant contextual information. We hypothesise that masking these retrieval heads can induce hallucinations and that contrasting the outputs of the base LLM and the masked LLM can reduce hallucinations. To this end, we propose Decoding by Contrasting Retrieval Heads (DeCoRe), a novel training-free decoding strategy that amplifies information found in the context and model parameters. DeCoRe mitigates potentially hallucinated responses by dynamically contrasting the outputs of the base LLM and the masked LLM, using conditional entropy as a guide. Our extensive experiments confirm that DeCoRe significantly improves performance on tasks requiring high contextual faithfulness, such as summarisation (XSum by 18.6%), instruction following (MemoTrap by 10.9%), and open-book question answering (NQ-Open by 2.4% and NQ-Swap by 5.5%).

- 【AVATAR: 自动化优化LLM智能体框架】研讨
  -  题目：AVATAR: Optimizing LLM Agents for Effective Tool Utilization
  -  作者：Shirley Wu, Shiyu Zhao, Qian Huang, Kexin Huang, Michihiro Yasunaga, Kaidi Cao, Vassilis N. Ioannidis, Karthik Subbian, Jure Leskovec, James Zou
  -  单位：Stanford University, Amazon
  -  解决问题：解决了如何自动化地优化大型语言模型（LLM）代理，使其能够更有效地利用外部工具和知识来提高任务性能的问题。
  -  关键思路：AVATAR通过一个比较器模块，使用对比推理技术从正负样本中迭代生成全面的提示，以优化LLM代理的工具使用策略。
  -  摘要：Large language model(LLM) agents have demonstrated impressive capabilities in utilizing external tools and knowledge to boost accuracy and reduce hallucinations. However, developing prompting techniques that enable LLM agents to effectively use these tools and knowledge remains a heuristic and labor-intensive task. Here, we introduce AVATAR, a novel and automated framework that optimizes an LLM agent to effectively leverage provided tools, improving performance on a given task. During optimization, we design a comparator module to iteratively deliver insightful and comprehensive prompts to the LLM agent by contrastively reasoning between positive and negative examples sampled from training data. We demonstrate AVATAR on four complex multimodal retrieval datasets featuring textual, visual, and relational information, and three general question-answering(QA) datasets. We find AVATAR consistently outperforms state-of-the-art approaches across all seven tasks, exhibiting strong generalization ability when applied to novel cases and achieving an average relative improvement of 14% on the Hit@1 metric for the retrieval datasets and 13% for the QA datasets. Code and dataset are available at https://github.com/zou-group/avatar.
### 2024.12.19
- 【图分类中的数据增强策略】研讨
  - 题目：A Simple Data Augmentation for Graph Classification: A Perspective of Equivariance and Invariance
  - 作者：YONGDUO SUI, SHUYAO WANG, JIE SUN, ZHIYUAN LIU, QING CUI, LONGFEI LI, JUN ZHOU, XIANG WANG, XIANGNAN HE
  - 单位：University of Science and Technology of China, National University of Singapore, Ant Group
  - 解决问题：解决了图分类中如何同时提高对稳定特征的敏感性和对环境特征的不变性，以应对分布外（OOD）问题的挑战。
  - 关键思路：通过混合稳定特征和交换环境特征来训练模型，使其对稳定特征具有等变性（equivariance），对环境特征具有不变性（invariance）。
  - 摘要：In graph classification, the out-of-distribution (OOD) issue is attracting great attention. To address this issue, a prevailing idea is to learn stable features, on the assumption that they are substructures causally determining the label and that their relationship with the label is stable to the distributional uncertainty. In contrast, the complementary parts termed environmental features, fail to determine the label solely and hold varying relationships with the label, thus ascribed to the possible reason for the distribution shift. Existing generalization efforts mainly encourage the model's insensitivity to environmental features. While the sensitivity to stable features is promising to distinguish the crucial clues from the distributional uncertainty but largely unexplored. A paradigm of simultaneously exploring the sensitivity to stable features and insensitivity to environmental features is until-now lacking to achieve the generalizable graph classification, to the best of our knowledge. In this work, we conjecture that generalizable models should be sensitive to stable features and insensitive to environmental features. To this end, we propose a simple yet effective augmentation strategy for graph classification: Equivariant and Invariant Cross-Data Augmentation (EI-CDA). By employing equivariance, given a pair of input graphs, we first estimate their stable and environmental features via masks. Then we linearly mix the estimated stable features of two graphs and encourage the model predictions faithfully reflect their mixed semantics. Meanwhile, by using invariance, we swap the estimated environmental features of two graphs and keep the predictions invariant. This simple yet effective strategy endows the models with both sensitivity to stable features and insensitivity to environmental features. Extensive experiments show that EI-CDA significantly improves performance and outperforms leading baselines.

- 【利用大语言模型增强异质图建模】研讨
  -  题目：Exploring the Potential of Large Language Models for Heterophilic Graphs
  -  作者：Anonymous ACL submission
  -  单位：Anonymous
  -  解决问题：解决了现有方法在处理异质图时忽略节点关联的丰富文本数据的问题，这些数据可以提供对异质性上下文的更深层次理解。
  -  关键思路：提出了一种两阶段框架LLM4HeG，通过微调大语言模型来区分同质和异质边，并在图神经网络中自适应地管理不同边类型的消息传播。
  -  摘要：Large language models(LLMs) have presented significant opportunities to enhance various ma-chine learning applications, including graph neural networks(GNNs). By leveraging the vast open-world knowledge within LLMs, we can more effectively interpret and utilize tex-tual data to better characterize heterophilic graphs, where neighboring nodes often have different labels. However, existing approaches for heterophilic graphs overlook the rich tex-tual data associated with nodes, which could unlock deeper insights into their heterophilic contexts. In this work, we explore the potential of LLMs for modeling heterophilic graphs and propose a novel two-stage framework: LLM-enhanced edge discriminator and LLM-guided edge reweighting. In the first stage, we fine-tune the LLM to better identify homophilic and heterophilic edges based on the textual content of their nodes. In the second stage, we adap-tively manage message propagation in GNNs for different edge types based on node features, structures, and heterophilic or homophilic char-acteristics. To cope with the computational demands when deploying LLMs in practical scenarios, we further explore model distilla-tion techniques to fine-tune smaller, more ef-ficient models that maintain competitive per-formance. Extensive experiments validate the effectiveness of our framework, demonstrating the feasibility of using LLMs to enhance node classification on heterophilic graphs.
### 2024.12.26
- 【黑盒越狱框架】研讨
  - 题目：JailPO: A Novel Black-box Jailbreak Framework via Preference Optimization against Aligned LLMs
  - 作者：Hongyi Li, Jiawei Ye, Jie Wu, Tianjie Yan, Chu Wang, Zhixin Li
  - 单位：School of Computer Science, Fudan University, Shanghai, China
  - 解决问题：解决了现有越狱技术在大规模语言模型（LLMs）对齐评估中存在的可扩展性、效率和通用性问题。
  - 关键思路：JailPO通过训练攻击模型自动生成隐蔽的越狱提示，并引入基于偏好优化的攻击方法来提高越狱效果，从而在黑盒访问条件下实现高效、通用和鲁棒的越狱攻击。
  - 摘要：Large Language Models(LLMs) aligned with human feed-back have recently garnered significant attention. However,it remains vulnerable to jailbreak attacks, where adversaries manipulate prompts to induce harmful outputs. Exploring jailbreak attacks enables us to investigate the vulnerabili-ties of LLMs and further guides us in enhancing their secu-rity. Unfortunately, existing techniques mainly rely on hand-crafted templates or generated-based optimization, posing challenges in scalability, efficiency and universality. To ad-dress these issues, we present JailPO, a novel black-box jail-break framework to examine LLM alignment. For scalabil-ity and universality, JailPO meticulously trains attack models to automatically generate covert jailbreak prompts. Further-more, we introduce a preference optimization-based attack method to enhance the jailbreak effectiveness, thereby im-proving efficiency. To analyze model vulnerabilities, we pro-vide three flexible jailbreak patterns. Extensive experiments demonstrate that JailPO not only automates the attack process while maintaining effectiveness but also exhibits superior per-formance in efficiency, universality, and robustness against defenses compared to baselines. Additionally, our analysis of the three JailPO patterns reveals that attacks based on com-plex templates exhibit higher attack strength, whereas covert question transformations elicit riskier responses and are more likely to bypass defense mechanisms.

- 【CREAM: 高效长上下文扩展方法】研讨
  - 题目：An Efficient Recipe for Long Context Extension via Middle-Focused Positional Encoding
  - 作者：Tong Wu, Yanpeng Zhao, Zilong Zheng
  - 单位：State Key Laboratory of General Artificial Intelligence, BIGAI, Beijing, China
  - 解决问题：解决了现有方法在扩展预训练大型语言模型（LLMs）的上下文长度时，通常需要在目标长度上进行微调，并且在利用上下文中间部分信息方面存在不足的问题。
  - 关键思路：CREAM通过操纵位置索引来插值位置编码，提出了一种简单且高效的微调方法，能够在不增加计算开销的情况下，显著扩展LLMs的上下文长度，并通过截断高斯分布鼓励模型关注上下文的中间部分。
  - 摘要：Transformer-based Large Language Models (LLMs) are typically pre-trained with a fixed context window size. However, many downstream applications necessitate the processing of significantly longer contexts. To address these issues, we propose Continuity-Relativity indExing with gAussian Middle (CREAM), which interpolates positional encodings by manipulating position indices. CREAM is training-efficient: it only requires fine-tuning at the pre-trained context window and can extend LLMs to a much longer target context length. To ensure that the model focuses more on the information in the middle, we introduce a truncated Gaussian to encourage sampling from the middle part of the context during fine-tuning, thus alleviating the “Lost-in-the-Middle” problem faced by long-context LLMs.
### 2025.1.2
- 【动态图变换器用于社交机器人检测】研讨
  - 题目：Dynamicity-aware Social Bot Detection with Dynamic Graph Transformers
  - 作者：Buyun He, Yingguang Yang, Qi Wu, Hao Liu, Renyu Yang, Hao Peng, Xiang Wang, Yong Liao, Pengyuan Zhou
  - 单位：University of Science and Technology of China, Beihang University, Harbin Engineering University
  - 解决问题：解决了社交机器人检测中的动态性建模问题，通过考虑社交网络的动态变化和行为模式演化，提高社交机器人检测的准确性。
  - 关键思路：BotDGT框架通过结构模块捕捉社交网络的历史拓扑信息，并通过时间模块整合历史上下文，利用自注意力机制建模用户行为模式的演化。
  - 摘要：Detecting social bots has evolved into a pivotal yet intricate task, aimed at combating the dissemination of misinformation and preserving the authenticity of online interactions. While earlier graph-based approaches, which leverage topological structure of social networks, yielded notable outcomes, they overlooked the inherent dynamicity of social networks. To tackle these challenges, we propose BotDGT, a novel framework that not only considers the topological structure, but also effectively incorporates dynamic nature of social network. Experimental results demonstrate the superiority of BotDGT against the leading methods that neglected the dynamic nature of social networks in terms of accuracy, recall, and F1-score.

- 【经典图神经网络作为强基线】研讨
  - 题目：Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification
  - 作者：Yuankai Luo, Lei Shi, Xiao-Ming Wu
  - 单位：Beihang University, The Hong Kong Polytechnic University
  - 解决问题：重新评估了经典图神经网络（GNN）在节点分类任务中的表现，挑战了图变换器（GTs）在该领域的优越性。
  - 关键思路：通过对经典GNN模型进行超参数调优，发现它们在节点分类任务中能够达到或超过最先进的图变换器的性能。
  - 摘要：Graph Transformers (GTs) have recently emerged as popular alternatives to traditional message-passing Graph Neural Networks (GNNs), due to their theoretically superior expressiveness and impressive performance reported on standard node classification benchmarks, often significantly outperforming GNNs. In this paper, we conduct a thorough empirical analysis to reevaluate the performance of three classic GNN models (GCN, GAT, and GraphSAGE) against GTs. Our findings suggest that the previously reported superiority of GTs may have been overstated due to suboptimal hyperparameter configurations in GNNs. Remarkably, with slight hyperparameter tuning, these classic GNN models achieve state-of-the-art performance, matching or even exceeding that of recent GTs across 17 out of the 18 diverse datasets examined. Additionally, we conduct detailed ablation studies to investigate the influence of various GNN configurations—such as normalization, dropout, residual connections, and network depth—on node classification performance. Our study aims to promote a higher standard of empirical rigor in the field of graph machine learning, encouraging more accurate comparisons and evaluations of model capabilities.
### 2025.2.15
- 【视觉-语言模型（VLMs）对抗性攻击】研讨
  - 题目：Text-Guided Attention is All You Need for Zero-Shot Robustness in Vision-Language Models
  - 作者：Lu Yu, Haiyang Zhang, Changsheng Xu
  - 单位：Tianjin University of Technology, University of Chinese Academy of Sciences
  - 解决问题：解决了大规模预训练视觉-语言模型（VLMs）在对抗性攻击下的零样本鲁棒性问题，特别是CLIP模型在面对对抗性扰动时的脆弱性。
  - 关键思路：提出了一种基于文本引导注意力的方法（TGA-ZSR），通过注意力细化模块和对齐模型约束模块来增强模型的鲁棒性，同时保持其在清洁样本上的性能。
  - 摘要：Pre-trained vision-language models (e.g., CLIP) have shown impressive zero-shot capabilities but are susceptible to adversarial examples. We propose Text-Guided Attention for Zero-Shot Robustness (TGA-ZSR), which includes an Attention Refinement module and an Attention-based Model Constraint module to enhance the model's robustness while maintaining performance on clean samples. Experiments show that TGA-ZSR improves zero-shot robust accuracy by 9.58% over state-of-the-art techniques across 16 datasets.

- 【ONE FOR ALL: 统一图模型的分类任务】研讨
  - 题目：ONE FOR ALL: TOWARDS TRAINING ONE GRAPH MODEL FOR ALL CLASSIFICATION TASKS
  - 作者：Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, Muhan Zhang
  - 单位：Washington University in St. Louis, Nanyang Technological University, Peking University
  - 解决问题：本文旨在解决构建一个统一的图模型来处理不同领域和任务的挑战，特别是节点、链接和图级别的分类任务。
  - 关键思路：本文提出了One for All (OFA)框架，通过文本属性图（TAGs）统一不同领域的图数据，并引入节点兴趣子图（NOI）和图提示范式（GPP）来实现跨域和上下文学习。
  - 摘要：Designing a single model to address multiple tasks has been a long-standing objective in artificial intelligence. We propose One for All (OFA), the first general framework that can use a single graph model to address the challenges of diverse graph tasks. OFA uses text-attributed graphs to unify different graph data and introduces the concept of nodes-of-interest to standardize different tasks. It also introduces a novel graph prompting paradigm that enables in-context learning without fine-tuning. OFA performs well across different tasks, making it the first general-purpose across-domains classification model on graphs.
### 2025.2.25
- 【语音伪造检测框架】研讨
  - 题目：Discriminative Feature Decoupling Enhancement for Speech Forgery Detection
  - 作者：Yijun Bei, Xing Zhou, Erteng Liu, Yang Gao, Sen Lin, Kewei Gao, Zunlei Feng
  - 单位：School of Software Technology, Zhejiang University; State Key Laboratory of Blockchain and Security, Zhejiang University; Hangzhou High-Tech Zone(Binjiang) Institute of Blockchain and Data Security; Ningbo Donghai Group Co., Ltd.
  - 解决问题：解决了现有语音伪造检测方法在提取混合特征时存在冗余信息，且未能充分考虑人声特征和背景环境声音多样性的问题。
  - 关键思路：DEEM框架通过交换解耦策略将语音分离为人声特征和背景声音特征，并分别通过时间维度和频谱维度聚合增强这些特征，最终利用异构图注意力网络进行伪造检测。
  - 摘要：The emergence of AIGC has brought attention to the issue of generating realistic deceptive content. While AIGC has the potential to revolutionize content creation, it also facilitates criminal activities. Specifically, the manipulation of speech has been exploited in tele-fraud and financial fraud schemes, posing a significant threat to societal security. Current deep learning-based methods for detecting forged speech extract mixed features from the original speech, which often contain redundant information. Moreover, these methods fail to consider the distinct characteristics of human voice-specific features and the diversity of background environmental sounds. This paper introduces a framework called Discriminative fEature dEcoupling enhanceMent(DEEM) for detecting speech forgery. Initially, the framework decouples the original speech into human voice features and background sound features. Subsequently, DEEM enhances voice-specific features through temporal dimension aggregation and improves continuity-related features in the background sound map via spectral-dimension aggregation. By employing the decoupling enhancement features, extensive experiments demonstrate that DEEM achieves an accuracy improvement of over 5% on FoR dataset compared to the state-of-the-art methods.

- 【SAug: 结构不平衡感知图增强】研讨
  - 题目：SAug: Structural Imbalance Aware Augmentation for Graph Neural Networks
  - 作者：KE-JIA CHEN, WENHUI MU, ZULONG LIU, ZHENG LIU
  - 单位：School of Computer Science and Technology, Nanjing University of Posts and Telecommunications, China
  - 解决问题：解决图中结构不平衡的问题，提升图神经网络（GNN）模型的鲁棒性和性能。
  - 关键思路：通过选择性图增强方法，针对不同节点组（中心节点和尾节点）采用不同的增强策略，减少结构不平衡，改善节点表示学习。
  - 摘要：Graph machine learning (GML) has made great progress in node classification, link prediction, graph classification and so on. However, graphs in reality are often structurally imbalanced, that is, only a few hub nodes have a denser local structure and higher influence. The imbalance may compromise the robustness of existing GML models, especially in learning tail nodes. This paper proposes a selective graph augmentation method to solve this problem. Firstly, a Pagerank-based sampling strategy is designed to identify hub nodes and tail nodes in the graph. Secondly, a selective augmentation strategy is proposed, which drops the noise neighbors of hub nodes on one side, and discovers the latent neighbors and generates pseudo neighbors for tail nodes on the other side. Also, it can alleviate the structural imbalance between two types of nodes. Finally, a GNN model is retrained on the augmented graph. Extensive experiments demonstrate that the proposed method can significantly improve the backbone GNNs and achieve superior performance to its competitors of graph augmentation methods and hub/tail aware methods.
### 2025.2.25
- 【语音伪造检测框架】研讨
  - 题目：Discriminative Feature Decoupling Enhancement for Speech Forgery Detection
  - 作者：Yijun Bei, Xing Zhou, Erteng Liu, Yang Gao, Sen Lin, Kewei Gao, Zunlei Feng
  - 单位：School of Software Technology, Zhejiang University; State Key Laboratory of Blockchain and Security, Zhejiang University; Hangzhou High-Tech Zone(Binjiang) Institute of Blockchain and Data Security; Ningbo Donghai Group Co., Ltd.
  - 解决问题：解决了现有语音伪造检测方法在提取混合特征时存在冗余信息，且未能充分考虑人声特征和背景环境声音多样性的问题。
  - 关键思路：DEEM框架通过交换解耦策略将语音分离为人声特征和背景声音特征，并分别通过时间维度和频谱维度聚合增强这些特征，最终利用异构图注意力网络进行伪造检测。
  - 摘要：The emergence of AIGC has brought attention to the issue of generating realistic deceptive content. While AIGC has the potential to revolutionize content creation, it also facilitates criminal activities. Specifically, the manipulation of speech has been exploited in tele-fraud and financial fraud schemes, posing a significant threat to societal security. Current deep learning-based methods for detecting forged speech extract mixed features from the original speech, which often contain redundant information. Moreover, these methods fail to consider the distinct characteristics of human voice-specific features and the diversity of background environmental sounds. This paper introduces a framework called Discriminative fEature dEcoupling enhanceMent(DEEM) for detecting speech forgery. Initially, the framework decouples the original speech into human voice features and background sound features. Subsequently, DEEM enhances voice-specific features through temporal dimension aggregation and improves continuity-related features in the background sound map via spectral-dimension aggregation. By employing the decoupling enhancement features, extensive experiments demonstrate that DEEM achieves an accuracy improvement of over 5% on FoR dataset compared to the state-of-the-art methods.

- 【SAug: 结构不平衡感知图增强】研讨
  - 题目：SAug: Structural Imbalance Aware Augmentation for Graph Neural Networks
  - 作者：KE-JIA CHEN, WENHUI MU, ZULONG LIU, ZHENG LIU
  - 单位：School of Computer Science and Technology, Nanjing University of Posts and Telecommunications, China
  - 解决问题：解决图中结构不平衡的问题，提升图神经网络（GNN）模型的鲁棒性和性能。
  - 关键思路：通过选择性图增强方法，针对不同节点组（中心节点和尾节点）采用不同的增强策略，减少结构不平衡，改善节点表示学习。
  - 摘要：Graph machine learning (GML) has made great progress in node classification, link prediction, graph classification and so on. However, graphs in reality are often structurally imbalanced, that is, only a few hub nodes have a denser local structure and higher influence. The imbalance may compromise the robustness of existing GML models, especially in learning tail nodes. This paper proposes a selective graph augmentation method to solve this problem. Firstly, a Pagerank-based sampling strategy is designed to identify hub nodes and tail nodes in the graph. Secondly, a selective augmentation strategy is proposed, which drops the noise neighbors of hub nodes on one side, and discovers the latent neighbors and generates pseudo neighbors for tail nodes on the other side. Also, it can alleviate the structural imbalance between two types of nodes. Finally, a GNN model is retrained on the augmented graph. Extensive experiments demonstrate that the proposed method can significantly improve the backbone GNNs and achieve superior performance to its competitors of graph augmentation methods and hub/tail aware methods.
### 2025.3.4
- 【长上下文序列时的计算复杂性和内存需求问题】研讨
  - 题目：Unshackling Context Length: An Efficient Selective Attention Approach through Query-Key Compression
  - 作者：Haoyu Wang, Tong Teng, Tianyu Guo, An Xiao, Duyu Tang, Hanting Chen, Yunhe Wang
  - 单位：Huawei Noah's Ark Lab, Huawei CBG
  - 解决问题：解决了大型语言模型（LLMs）在处理长上下文序列时的计算复杂性和内存需求问题。
  - 关键思路：提出了一种高效的令牌级选择性注意力算法（ESA），通过压缩查询和键向量来降低计算复杂度，并在令牌级别上选择最重要的令牌进行注意力计算。
  - 摘要：Handling long-context sequences efficiently remains a significant challenge in large language models(LLMs). Existing methods for token selection in sequence extrapolation either employ a permanent eviction strategy or select tokens by chunk, which may lead to the loss of critical information. We propose Efficient Selec-tive Attention(ESA), a novel approach that ex-tends context length by efficiently selecting the most critical tokens at the token level to com-pute attention. ESA reduces the computational complexity of token selection by compressing query and key vectors into lower-dimensional representations. We evaluate ESA on long se-quence benchmarks with maximum lengths up to 256k using open-source LLMs with context lengths of 8k and 32k. ESA outperforms other selective attention methods, especially in tasks requiring the retrieval of multiple pieces of in-formation, achieving comparable performance to full-attention extrapolation methods across various tasks, with superior results in certain tasks.

- 【xRAG：用于检索增强生成的极端上下文压缩】研讨
  - 题目：xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token
  - 作者：Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, Dongyan Zhao
  - 单位：Peking University, Microsoft, National Key Laboratory of General Artificial Intelligence
  - 解决问题：解决检索增强生成（RALM）中因将整个文档加入提示而导致的推理成本增加和上下文限制问题。
  - 关键思路：xRAG通过将文档嵌入作为检索模态特征，利用模态融合方法将其无缝集成到语言模型的表示空间中，从而实现极端的上下文压缩。
  - 摘要：This paper introduces xRAG, a novel context compression method designed specifically for retrieval-augmented generation. xRAG redefines the use of document embeddings in dense retrieval—traditionally limited to retrieval purposes—by integrating them as features from the retrieval modality. Through a modality fusion approach, xRAG effectively merges these embeddings into the language model’s representation space, eliminating the need for their textual counterparts and achieving an extreme compression rate. In xRAG, the modality bridge is the only trainable component, while the retriever and language model remain frozen. This design choice allows for the reuse of offline-constructed document embeddings and preserves the plug-and-play nature of retrieval augmentation. Experimental results demonstrate that xRAG achieves an average improvement of over 10% across six knowledge-intensive tasks, compatible with various language model backbones, ranging from a dense 7B model to an 8x7B Mixture of Experts configuration. xRAG not only significantly outperforms previous context compression methods but also matches the performance of uncompressed models on several benchmarks, while reducing overall FLOPs by a factor of 3.53. This work pioneers new avenues in retrieval-augmented generation through multimodal fusion, potentially setting a groundwork for future developments in efficient and scalable retrieval systems.
### 2025.3.11
- 【边提示调优】研讨
  - 题目：EDGE PROMPT TUNING FOR GRAPH NEURAL NETWORKS
  - 作者：Xingbo Fu, Yinhan He, Jundong Li
  - 单位：University of Virginia
  - 解决问题：本文旨在解决预训练图神经网络（GNNs）与下游任务之间的目标差距问题，特别是通过设计有效的图提示来桥接这一差距。
  - 关键思路：本文提出了一种新的边缘提示调优方法（EdgePrompt），通过在图的边学习额外的提示向量，并通过消息传递机制将这些提示整合到预训练的GNN模型中，以更好地嵌入图结构信息，从而提升下游任务的性能。
  - 摘要：Pre-training powerful Graph Neural Networks (GNNs) with unlabeled graph data in a self-supervised manner has emerged as a prominent technique in recent years. However, inevitable objective gaps often exist between pre-training and down-stream tasks. To bridge this gap, graph prompt tuning techniques design and learn graph prompts by manipulating input graphs or reframing downstream tasks as pre-training tasks without fine-tuning the pre-trained GNN models. While recent graph prompt tuning methods have proven effective in adapting pre-trained GNN models for downstream tasks, they overlook the crucial role of edges in graph prompt design, which can significantly affect the quality of graph representations for downstream tasks. In this study, we propose EdgePrompt, a simple yet effective graph prompt tuning method from the perspective of edges. Unlike previous studies that design prompt vectors on node features, EdgePrompt manipulates input graphs by learning additional prompt vectors for edges and incorporates the edge prompts through message passing in the pre-trained GNN models to better embed graph structural information for downstream tasks.

- 【社交机器人检测框架】研讨
  - 题目：Unmasking Social Robots' Camouflage: A GNN-Random Forest Framework for Enhanced Detection
  - 作者：Weijian Fan, Chunhua Wang, Xiao Han, Chichen Lin
  - 单位：School of Data Science and Intelligent Media, Communication University of China, Beijing, 100024, China; School of Computer and Cyber Sciences, Communication University of China, Beijing, 100024, China; Institute of Communication Studies, Communication University of China, Beijing, 100024, China; State Key Laboratory of Media Convergence and Communication, Communication University of China, Beijing, 100024, China
  - 解决问题：提出了一种新的社交机器人检测框架HORFBot，解决了现有图神经网络方法因同质性假设而无法有效检测与真实用户互动的社交机器人的问题。
  - 关键思路：HORFBot通过图增强模块提高图的同质性，利用对比学习训练多个图卷积网络，并将图神经网络作为随机森林的基础分类器，通过集成学习提高检测性能。
  - 摘要：The proliferation of robot accounts on social media platforms has posed a significant negative impact, necessitating robust measures to counter network anomalies and safeguard content integrity. Social robot detection has emerged as a pivotal yet intricate task, aimed at mitigating the dissemination of misleading information. While graph-based approaches have attained remarkable performance in this realm, they grapple with a fundamental limitation:the homogeneity assumption in graph convolution allows social robots to stealthily evade detection by mingling with genuine human profiles. To unravel this challenge and thwart the camouflage tactics, this work proposed an innovative social robot detection framework based on enhanced HOmogeneity and Random Forest(HORFBot).At the core of HORFBot lies a homogeneous graph enhancement strategy,intricately woven with edge-removal techniques, to meticulously dissect the graph into multiple revealing subgraphs. Subsequently, leveraging the power of contrastive learning, the proposed methodology meticulously trains multiple graph convolutional networks,each honed to discern nuances within these tailored subgraphs. The culminating stage involves the fusion of these feature-rich base classifiers, harmoniously aggregating their insights to produce a comprehensive detection outcome. Extensive experiments on three social robot detection datasets have shown that this method effectively improves the accuracy of social robot detection and outperforms comparative methods.
### 2025.3.18
- 【LLM记忆】研讨
  - 题目：A-MEM: Agentic Memory for LLM Agents
  - 作者：Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang
  - 单位： Rutgers University，Ant Group
  - 解决问题：解决了现有记忆系统在动态组织记忆和适应多样化任务方面的局限性，尤其是在复杂、开放性任务中缺乏灵活性和通用性的问题。
  - 关键思路：A-MEM通过借鉴Zettelkasten方法，设计了一个动态且自我进化的记忆系统，使LLM代理能够自主生成上下文描述、动态建立记忆连接，并基于新经验智能地进化现有记忆。
  - 摘要：While large language model(LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current mem-ory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed op-erations and structures limit their adaptability across diverse tasks. To address this limita-tion, this paper proposes a novel agentic mem-ory system for LLM agents that can dynam-ically organize memories in an agentic way. Following the basic principles of the Zettelkas-ten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a compre-hensive note containing multiple structured at-tributes, including contextual descriptions, key-words, and tags. The system then analyzes historical memories to identify relevant con-nections, establishing links where meaningful similarities exist. Additionally, this process en-ables memory evolution- as new memories are integrated, they can trigger updates to the con-textual representations and attributes of exist-ing historical memories, allowing the memory network to continuously refine its understand-ing. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision mak-ing, allowing for more adaptive and context-aware memory management. Empirical exper-iments on six foundation models show supe-rior improvement against existing SOTA base-lines. The source code for evaluating perfor-mance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.

- 【图像木马越狱框架】研讨
  - 题目：ImgTrojan: Jailbreaking Vision-Language Models with ONE Image
  - 作者：Xijia Tao*, Shuai Zhong*, Lei Li*, Qi Liu, Lingpeng Kong
  - 单位：The University of Hong Kong
  - 解决问题：揭示了视觉-语言模型（VLMs）在面对图像数据投毒攻击时的脆弱性，即通过少量恶意图像即可绕过其安全机制。
  - 关键思路：通过在训练数据中注入带有恶意越狱提示的图像-文本对，使得VLM在推理时对看似正常的图像产生有害响应，从而实现越狱攻击。
  - 摘要：There has been an increasing interest in the alignment of large language models(LLMs)with human values. However, the safety issues of their integration with a vision module, or vision language models(VLMs), remain rel-atively underexplored. In this paper, we pro-pose a novel jailbreaking attack against VLMs,aiming to bypass their safety barrier when a user inputs harmful instructions. A scenario where our poisoned(image, text) data pairs are included in the training data is assumed.By replacing the original textual captions with malicious jailbreak prompts, our method can perform jailbreak attacks with the poisoned im-ages. Moreover, we analyze the effect of poison ratios and positions of trainable parameters on our attack's success rate. For evaluation, we design two metrics to quantify the success rate and the stealthiness of our attack. Together with a list of curated harmful instructions, a bench-mark for measuring attack efficacy is provided.We demonstrate the efficacy of our attack by comparing it with baseline methods.
### 2025.3.25
- 【THE BELIEF STATE TRANSFORMER】研讨
  - 题目：THE BELIEF STATE TRANSFORMER
  - 作者：Ada Langford, Dinesh Jayaraman, Alex Lamb, John Langford
  - 单位：Microsoft Research, University of Pennsylvania, UT Austin, University of Alberta
  - 解决问题：解决了传统前向Transformer在处理需要规划和目标导向的任务时的局限性，特别是在需要同时考虑前缀和后缀信息的复杂序列预测任务中表现不佳的问题。
  - 关键思路：Belief State Transformer通过同时编码前缀和后缀，并预测前缀的下一个 token 和后缀的前一个 token，从而学习到一个紧凑的信念状态，使得模型能够在目标导向的任务中更有效地进行解码和推理。
  - 摘要：We introduce the “Belief State Transformer”, a next-token predictor that takes both a prefix and suffix as inputs, with a novel objective of predicting both the next token for the prefix and the previous token for the suffix. The Belief State Transformer effectively learns to solve challenging problems that conventional forward-only transformers struggle with, in a domain-independent fashion. Key to this success is learning a compact belief state that captures all relevant information necessary for accurate predictions. Empirical ablations show that each component of the model is essential in difficult scenarios where standard Transformers fall short. For the task of story writing with known prefixes and suffixes, our approach outperforms the Fill-in-the-Middle method for reaching known goals and demonstrates improved performance even when the goals are unknown. Altogether, the Belief State Transformer enables more efficient goal-conditioned decoding, better test-time inference, and high-quality text representations on small scale problems. Website:https://sites.google.com/view/belief-state-transformer

- 【蛋白质效应子预测模型】研讨
  - 题目：Contrastive-learning of language embedding and biological features for cross modality encoding and effector prediction
  - 作者：Yue Peng, Junze Wu, Yi Sun, Yuanxing Zhang, Qiyao Wang, Shuai Shao
  - 单位：School of Computer Science, East China University of Science and Technology, Shanghai, China; School of Basic Medical Sciences, East China University of Science and Technology, Shanghai, China; College of Life Sciences, East China Normal University, Shanghai, China
  - 解决问题：识别和表征革兰氏阴性菌分泌的毒力蛋白对于解析微生物致病性和开发治疗策略至关重要，但现有的效应子预测模型的准确性和敏感性仍面临挑战。
  - 关键思路：CLEF模型通过对比学习将预训练的语言模型表示与补充的生物特征相结合，生成有意义的跨模态表示，从而提高效应子预测的性能。
  - 摘要：Identifying and characterizing virulence proteins secreted by Gram-negative bacteria are fundamental for deciphering microbial pathogenicity as well as aiding the development of therapeutic strategies. Effector predictors utilizing pre-trained protein language models(PLMs) have shown sound performance by leveraging extensive evolutionary and sequential protein features. However, the accuracy and sensitivity of effector prediction remain challenging. Here, we introduce a model named Contrastive-learning of Language Embedding and Biological Features(CLEF) leveraging contrastive learning to integrate PLM representations with supplementary biological features. Biologically information is captured in learned contextualized embeddings to yield meaningful representations. With cross-modality biological features, CLEF outperforms state-of-the-art(SOTA) models in predicting type III, type IV, and type VI secreted effectors(T3SEs/T4SEs/T6SEs) in enteric pathogens.
### 2025.4.1
- 【LLM位置编码】研讨
  - 题目：3D-RPE: Enhancing Long-Context Modeling Through 3D Rotary Position Encoding
  - 作者：Xindian Ma, Wenyuan Liu, Peng Zhang, Nan Xu
  - 单位：College of Intelligence and Computing, Tianjin University, Tianjin, China; Beijing Wenge Technology Co. Ltd.
  - 解决问题：解决了基于RoPE的大语言模型在长上下文建模中的长期衰减和位置分辨率下降的问题。
  - 关键思路：通过将旋转位置编码从二维扩展到三维球面，控制长距离衰减并提高位置分辨率，从而增强长上下文建模能力。
  - 摘要：Inspired by the Bloch Sphere representation, we propose a novel rotary position encoding on a three-dimensional sphere, named 3D Rotary Position Encoding(3D-RPE). 3D-RPE is an advanced version of the widely used 2D Rotary Position Encoding(RoPE), with two major advantages for modeling long contexts: controllable long-term decay and improved position resolution. For controllable long-term decay, 3D-RPE allows for the regulation of long-term decay within the chunk size, ensuring the modeling of relative positional information between tokens at a distant relative position. For enhanced position resolution, 3D-RPE can mitigate the degradation of position resolution caused by position interpolation on RoPE. We have conducted experiments on long-context Natural Language Understanding(NLU) and long-sequence Language Modeling(LM) tasks. From the experimental results, 3D-RPE achieved performance improvements over RoPE, especially in long-context NLU tasks.

- 【结构平衡传播缓解图神经网络过平滑】研讨
  - 题目：Oversmoothing as Loss of Sign: Towards Structural Balance in Graph Neural Networks
  - 作者：Jiaqi Wang, Xinyi Wu, James Cheng, Yifei Wang
  - 单位：The Chinese University of Hong Kong, MIT IDSS & LIDS
  - 解决问题：本文提出了一种新的统一视角，通过将抗过平滑技术解释为在有符号图上的消息传递，来解决图神经网络中的过平滑问题。
  - 关键思路：通过引入标签和特征信息构建结构平衡图，使得节点特征在长距离传播中收敛到各自簇的共享值，同时保持簇间的区分度，从而有效缓解过平滑现象。
  - 摘要：Oversmoothing is a common issue in graph neural networks(GNNs), where node representations become excessively homogeneous as the number of layers increases,resulting in degraded performance. Various strategies have been proposed to combat oversmoothing in practice, yet they are based on different heuristics and lack a unified understanding of their inherent mechanisms. In this paper, we show that three major classes of anti-oversmoothing techniques can be mathematically interpreted as message-passing over signed graphs comprising both positive and negative edges. By analyzing the asymptotic behavior of signed graph propagation,we demonstrate that negative edges can repel nodes to a certain extent, providing deeper insights into how these methods mitigate oversmoothing. Furthermore, our results suggest that the structural balance of a signed graph-where positive edges exist only within clusters and negative edges appear only between clusters-is crucial for clustering node representations in the long term through signed graph propagation. Motivated by these observations, we propose a solution to mitigate oversmoothing with theoretical guarantees-Structural Balance Propagation(SBP),by incorporating label and feature information to create a structurally balanced graph for message-passing. Experiments on nine datasets against twelve baselines demonstrate the effectiveness of our method, highlighting the value of our signed graph perspective.
### 2025.4.8
- 【FLOOD框架】研讨
  - 题目：FLOOD: A Flexible Invariant Learning Framework for Out-of-Distribution Generalization on Graphs
  - 作者：Yang Liu, Yunshan Ma, Xiang Ao, Kuan Li, Fuli Feng, Tat-Seng Chua, Qing He
  - 单位：Institute of Computing Technology, Chinese Academy of Sciences; National University of Singapore; University of Chinese Academy of Sciences; University of Science and Technology of China
  - 解决问题：解决了图神经网络（GNNs）在分布外（OOD）设置下泛化能力不足的问题，特别是在训练和测试数据分布不同的情况下。
  - 关键思路：FLOOD结合了不变学习（invariant learning）和自举学习（bootstrapped learning），通过数据增强构建多个环境来学习不变表示，并在测试阶段通过自监督任务灵活调整共享编码器以适应目标分布。
  - 摘要：Graph Neural Networks(GNNs) have achieved remarkable suc-cess in various domains but most of them are developed under the in-distribution assumption. Under out-of-distribution(OOD)settings, they suffer from the distribution shift between the train-ing set and the test set and may not generalize well to the test distribution. Several methods have tried the invariance principle to improve the generalization of GNNs in OOD settings. However,in previous solutions, the graph encoder is immutable after the invariant learning and cannot be adapted to the target distribution flexibly. Confronting the distribution shift, a flexible encoder with refinement to the target distribution can generalize better on the test set than the stable invariant encoder. To remedy these weak-nesses, we propose a Flexible invariant Learning framework for Out-Of-Distribution generalization on graphs(FLOOD), which com-prises two key components, invariant learning and bootstrapped learning. The invariant learning component constructs multiple environments from graph data augmentation and learns invariant representation under risk extrapolation. Besides, the bootstrapped learning component is devised to be trained in a self-supervised way with a shared graph encoder with the invariant learning part.During the test phase, the shared encoder is flexible to be refined with the bootstrapped learning on the test set. Extensive exper-iments are conducted for both transductive and inductive node classification tasks. The results demonstrate that FLOOD consis-tently outperforms other graph OOD generalization methods and effectively improves the generalization ability.
### 2025.4.15
- 【注意力汇聚机制】研讨
  - 题目：Why do LLMs attend to the first token?
  - 作者：Federico Barbero1,, Alvaro Arroyo1,, Xiangming Gu2, Christos Perivolaropoulos3, Michael Bronstein1, Petar Velikovi3, Razvan Pascanu3
  - 单位：1University of Oxford 2National University of Singapore 3Google DeepMind
  - 解决问题：解释了大型语言模型（LLMs）为何在注意力机制中倾向于将大量注意力集中在序列的第一个标记（如〈bos〉）上，揭示了这种“注意力汇聚”现象在避免信息过混合中的重要性。
  - 关键思路：通过理论分析和实验验证，提出注意力汇聚是一种防止Transformer架构中信息过混合的机制，特别是在模型深度增加或上下文长度变长时，通过将注意力集中在第一个标记来减缓表示坍塌和过平滑问题。
  - 摘要：Large Language Models(LLMs) tend to attend heavily to the first token in the sequence- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or al-leviate it. Attention sinks have been connected to quantisation difficulties,security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shal-lowly answered: Why do LLMs learn such patterns and how are they being used?In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to exist-ing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intu-itions and show how choices such as context length, depth, and data pack-ing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.
### 2025.4.22
- 【推理-检索框架】研讨
  - 题目：CHAIN-OF-ACTION: FAITHFUL AND MULTIMODAL QUESTION ANSWERING THROUGH LARGE LANGUAGE MODELS
  - 作者：Zhenyu Pan, Haozheng Luo, Manling Li, Han Liu
  - 单位：Department of Computer Science, Northwestern University, Evanston, IL 60208, USA; Department of Statistics and Data Science, Northwestern University, Evanston, IL 60208, USA
  - 解决问题：本文提出了一种新的推理-检索框架，旨在增强大型语言模型（LLMs）在回答复杂问题时的忠实性和多步推理能力，克服了当前QA应用中的两个主要挑战：与实时或领域事实不一致的不可靠幻觉，以及在组合信息上的弱推理性能。
  - 关键思路：本文提出了一个新颖的推理-检索机制，通过系统化提示和预设计的动作将复杂问题分解为推理链，并提出了三种可插拔的“即插即用”动作，用于从异构源中检索实时信息，同时提出了多参考信仰分数（MRFS）来验证答案中的冲突，从而提高答案的可靠性。
  - 摘要：We present a Chain-of-Action(CoA) framework for multimodal and retrieval-augmented Question-Answering(QA). Compared to the literature, CoA overcomes two major challenges of current QA applications:(i) unfaithful hallucination that is inconsistent with real-time or domain facts and(ii) weak reasoning performance over compositional information. Our key contribution is a novel reasoning-retrieval mechanism that decomposes a complex question into a reasoning chain via systematic prompting and pre-designed actions. Methodologically, we propose three types of domain-adaptable ‘Plug-and-Play’ actions for retrieving real-time information from heterogeneous sources. We also propose a multi-reference faith score to verify conflicts in the answers. In addition, our system demonstrates that detecting the knowledge boundaries of LLMs can significantly reduce both system latency and LLM usage in QA tasks. Empirically, we exploit both public benchmarks and a Web3 case study to demonstrate the capability of CoA over other methods.
### 2025.4.29
- 【半监督社交机器人检测】研讨
  - 题目：Semi-Supervised Social Bot Detection with Relational Graph Attention Transformers and Characteristics of the social environment
  - 作者：Di Huang, Jinbao Song, Xingyu Zhang
  - 单位：a State Key Laboratory of Media Convergence and Communication, Communication University Of China, Beijing, 100024, China; bSchool of Information and Communication Engineering, Communication University Of China, Beijing, 100024, China; c School of Data Science And Media Intelligence, Communication University Of China, Beijing, 100024, China
  - 解决问题：解决现有基于图的方法在社交机器人检测中未能有效提取和融合社交网络边关系信息以及处理非直接相连节点间相似特征关系的问题。
  - 关键思路：提出SRGAT框架，利用关系图注意力变换器学习节点表示，构建社交环境特征连接非直接相连节点形成新图特征，并与节点特征融合，通过语义注意力网络聚合消息，最终使用MLP进行机器人检测。
  - 摘要：Social bot detection is an important and challenging task in social network analysis and maintaining social network security. In recent years, graph neural networks(GNNs) have been widely used and applied to social bot detection research, effectively improving the performance of detection methods. However, current graph-based methods rarely extract and fuse the rich relationship information contained in the edges of social networks with node information, and lack effective feature fusion and processing of the relationship between two points that do not directly intersect. To address this challenge, we propose a Semi-Supervised Social Bot Detection with Relational Graph Attention Transformers(SRGAT) and Characteristics of the social environment.We first construct a graph network based on user metadata, tweet information, and multi-relationships of topological structure. Secondly, we propose to use relational graph attention transformers to learn node representation. Then we propose to construct social environment features, connect two non-intersecting points to form a new graph feature, and fuse it with the node features learned above to form a comprehensive node graph feature. Next, we use semantic attention networks to aggregate messages across users and relations.Finally, we use Multilayer Perceptron(MLP) to conduct bot detection. We use a consistency loss to boost the detection performance of the model for limited annotated data. Extensive experimental results on three real datasets show that our SRGAT has shown advanced performance in the experiment. Finally, further experiments show the effectiveness of our model, which can improve the detection performance of social bot detection.

- 【安全对齐的脆弱性评估】研讨
  - 题目：Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications
  - 作者：Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, Peter Henderson
  - 单位：Princeton University
  - 解决问题：研究大型语言模型（LLMs）在安全性对齐方面的脆弱性，特别是通过剪枝和低秩修改来识别和隔离关键的安全区域。
  - 关键思路：通过识别和隔离对安全性至关重要的神经元和秩，研究这些区域的稀疏性如何影响模型的安全性，并展示即使在限制修改的情况下，模型仍然容易受到低成本微调攻击的影响。
  - 摘要：Large language models(LLMs) show inherent brittleness in their safety mechanisms, as evidenced by their susceptibility to jailbreaking and even non-malicious fine-tuning. This study explores this brittleness of safety alignment by leveraging pruning and low-rank modifications. We develop methods to identify critical regions that are vital for safety guardrails, and that are disentangled from utility-relevant regions at both the neuron and rank levels. Surprisingly, the isolated regions we find are sparse, comprising about 3% at the parameter level and 2.5% at the rank level. Removing these regions compromises safety while only mildly impacting utility, corroborating the inherent brittleness of the model's safety mechanisms. Moreover, we show that LLMs remain vulnerable to low-cost fine-tuning attacks even when modifications to the safety-critical regions are restricted. These findings underscore the urgent need for more robust safety strategies in LLMs.
### 2025.5.8
- 【LOGIN框架】研讨
  - 题目：LOGIN: A Large Language Model Consulted Graph Neural Network Training Framework
  - 作者：Yiran Qiao, Xiang Ao, Yang Liu, Jiarong Xu, Xiaoqian Sun, Qing He
  - 单位：Institute of Computing Technology, Chinese Academy of Sciences, University of Chinese Academy of Sciences, Beijing, China; School of Management, Fudan University, Shanghai, China
  - 解决问题：解决了传统图神经网络（GNNs）在不同类型图数据上性能受限的问题，通过引入大语言模型（LLMs）提升GNNs的训练效果。
  - 关键思路：LOGIN框架通过交互式地利用LLMs对不确定节点进行咨询，结合特征更新和结构优化策略，增强GNNs的语义理解和拓扑结构学习能力。
  - 摘要：Recent prevailing works on graph machine learning typically follow a similar methodology that involves designing advanced variants of graph neural networks(GNNs) to maintain the superior performance of GNNs on different graphs. In this paper, we aim to streamline the GNN design process and leverage the advantages of Large Language Models(LLMs) to improve the performance of GNNs on downstream tasks. We formulate a new para-digm, coined “LLMs-as-Consultants”, which integrates LLMs with GNNs in an interactive manner. A framework named LOGIN(LLM cOnsulted GNN traINing) is instantiated, empowering the interactive utilization of LLMs within the GNN training process. First, we attentively craft concise prompts for spotted nodes, carrying comprehensive semantic and topological information, and serving as input to LLMs. Second, we refine GNNs by devising a complementary coping mechanism that utilizes the responses from LLMs, depending on their correctness. We empirically evaluate the effectiveness of LOGIN on node classification tasks across both ho-mophilic and heterophilic graphs. The results illustrate that even ba-sic GNN architectures, when employed within the proposed LLMs-as-Consultants paradigm, can achieve comparable performance to advanced GNNs with intricate designs. Our codes are available at https://github.com/QiaoYRan/LOGIN.

- 【自我注意模块中的大值】研讨
  - 题目：Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding
  - 作者：Mingyu Jin, Kai Mei, Wujiang Xu, Mingjie Sun, Ruixiang Tang, Mengnan Du, Zirui Liu, Yongfeng Zhang
  - 单位：Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China; University of Chinese Academy of Sciences, Beijing, China
  - 解决问题：揭示了在现代Transformer架构的大型语言模型（LLMs）中，自我注意模块中的大规模值在上下文知识理解中的关键作用。
  - 关键思路：通过系统地分析自我注意查询（Q）和键（K）中的大规模值，发现这些值在特定区域内集中出现，并且对上下文知识的理解至关重要，而不会显著影响参数化知识的检索。
  - 摘要：Large language models(LLMs) have achieved remarkable success in contextual knowledge understanding. In this paper, we show for the first time that these concentrated massive values consistently emerge in specific regions of attention queries(Q) and keys(K) while not having such patterns in values(V) in various modern transformer-based LLMs. Through extensive experiments, we further demonstrate that these massive values play a critical role in interpreting contextual knowledge(i.e., knowledge obtained from the current context window) rather than in retrieving parametric knowledge stored within the model's parameters. Our further investigation of quantization strategies reveals that ignoring these massive values leads to a pronounced drop in performance on tasks requiring rich contextual understanding, aligning with our analysis. Finally,we trace the emergence of concentrated massive values and find that such concentration is caused by Rotary Positional Encoding(RoPE) and it appears since very first layers. These findings shed new light on how Q and K operate in LLMs and offer practical insights for model design and optimization. The code is available at https://github.com/MingyuJ666/Rope_with_LLM.
### 2025.5.20
- 【隐式概率重连图神经网络】研讨
  - 题目：Implicit Probabilistically Rewired Message-Passing Neural Networks for Mitigating Over-Squashing and Under-Reaching
  - 作者：Chendi Qian, Andrei Manolache, Christopher Morris, Mathias Niepert
  - 单位：Computer Science Department, RWTH Aachen University, Germany; Computer Science Department, University of Stuttgart, Germany; IMPRS-IS, Germany; Bitdefender, Romania
  - 解决问题：提出了一种新的隐式概率重连消息传递神经网络（IPR-MPNN），以解决图神经网络（MPNNs）在处理长距离信息时的过挤压和欠到达问题。
  - 关键思路：IPR-MPNN通过在图中添加虚拟节点并学习隐式概率重连，有效地绕过了图变换器的二次复杂度，提升了模型的表达能力和适应性。
  - 摘要：Message-passing graph neural networks(MPNNs) have emerged as a powerful paradigm for graph-based machine learning. Despite their effectiveness, MPNNs face challenges such as under-reaching and over-squashing, where limited receptive fields and structural bottlenecks hinder information flow in the graph. While graph transformers hold promise in addressing these issues, their scalability is limited due to quadratic complexity regarding the number of nodes, rendering them impractical for larger graphs. Here, we propose implicitly rewired message-passing neural networks(IPR-MPNNs), a novel approach that integrates implicit probabilistic graph rewiring into MPNNs. By introducing a small number of virtual nodes, i.e.,adding additional nodes to a given graph and connecting them to existing nodes, in a differentiable, end-to-end manner, IPR-MPNNs enable long-distance message propagation, circumventing quadratic complexity. Theoretically, we demonstrate that IPR-MPNNs surpass the expressiveness of traditional MPNNs. Empirically, we validate our approach by showcasing its ability to mitigate under-reaching and over-squashing effects, achieving state-of-the-art performance across multiple graph datasets. Notably, IPR-MPNNs outperform graph transformers while maintaining significantly faster computational efficiency.

- 【视觉语言模型】研讨
  - 题目：Leveraging Vision Language Models for Specialized Agricultural Tasks
  - 作者：Muhammad Arbab Arshad, Talukder Zaki Jubery, Tirtho Roy, Rim Nassiri, Asheesh K. Singh, Arti Singh, Chinmay Hegde, Baskar Ganapathysubramanian, Aditya Balu, Adarsh Krishnamurthy, Soumik Sarkar
  - 单位：Iowa State University, USA;  New York University, USA
  - 解决问题：本文解决了农业领域中由于缺乏标注数据而难以开发专门模型的问题，通过评估视觉语言模型（VLMs）在植物应激表型分析中的潜力，提供了一种新的解决方案。
  - 关键思路：通过构建AgEval基准，评估多种最先进的VLMs在植物应激表型分析任务中的零样本和少样本学习性能，并引入相关指标来量化模型在不同类别上的表现差异，探索示例选择对模型可靠性的影响。
  - 摘要：As Vision Language Models(VLMs) become increasingly accessible to farmers and agricultural experts, there is a growing need to evaluate their potential in specialized tasks. We present AgEval, a comprehensive benchmark for assessing VLMs' capabilities in plant stress phenotyping, offering a solution to the challenge of limited annotated data in agriculture. Our study explores how general-purpose VLMs can be leveraged for domain-specific tasks with only a few annotated examples, providing insights into their behavior and adaptability. AgEval encompasses 12 diverse plant stress phenotyping tasks, evaluating zero-shot and few-shot in-context learning performance of state-of-the-art models including Claude, GPT, Gemini, and LLaVA. Our results demonstrate VLMs' rapid adaptability to specialized tasks, with the best-performing model showing an increase in F1 scores from 46.24% to 73.37% in 8-shot identification. To quantify performance disparities across classes, we introduce metrics such as the coefficient of variation(CV), revealing that VLMs' training impacts classes differently, with CV ranging from 26.02% to 58.03%. We also find that strategic example selection enhances model reliability, with exact category examples improving F1 scores by 15.38% on average. AgEval establishes a framework for assessing VLMs in agricultural applications, offering valuable benchmarks for future evaluations. Our findings suggest that VLMs, with minimal few-shot examples, show promise as a viable alternative to traditional specialized models in plant stress phenotyping, while also highlighting areas for further refinement. Results and benchmark details are available at: https://github.com/arbab-ml/AgEval
### 2025.5.27
- 【去噪扩散概率模型】研讨
  - 题目：Denoising Diffusion Probabilistic Models
  - 作者：Jonathan Ho, Ajay Jain, Pieter Abbeel
  - 单位：UC Berkeley
  - 解决问题：展示了扩散概率模型能够生成高质量图像样本，有时甚至优于其他类型的生成模型。
  - 关键思路：通过训练加权变分界限，结合去噪分数匹配与Langevin动力学的新型连接，实现高质量的图像合成。
  - 摘要：We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models nat-urally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. Our imple-mentation is available at https://github.com/hojonathanho/diffusion.

- 【注意力机制】研讨
  - 题目：TOKEN STATISTICS TRANSFORMER: LINEAR-TIME ATTENTION VIA VARIATIONAL RATE REDUCTION
  - 作者：Ziyang Wu, Tianjiao Ding, Yifu Lu, Druv Pai, Jingyuan Zhang, Weida Wang, Yaodong Yu, Yi Ma, Benjamin D. Haeffele
  - 单位：UC Berkeley, UPenn, UMich, THU, Tsinghua SIGS, JHU
  - 解决问题：提出了一种线性时间复杂度的注意力机制，显著提高了Transformer架构的计算效率，同时保持了其性能。
  - 关键思路：通过变分率减少的优化，推导出一种新的注意力模块，称为Token Statistics Self-Attention (TSSA)，该模块避免了计算token之间的成对相似性，而是基于输入token特征的经验二阶矩统计进行低秩投影。
  - 摘要：The attention operator is arguably the key distinguishing factor of transformer architectures, which have demonstrated state-of-the-art performance on a vari-ety of tasks. However, transformer attention operators often impose a significant computational burden, with the computational complexity scaling quadratically with the number of tokens. In this work, we propose a novel transformer atten-tion operator whose computational complexity scales linearly with the number of tokens. We derive our network architecture by extending prior work which has shown that a transformer style architecture naturally arises by“white-box”architecture design, where each layer of the network is designed to implement an incremental optimization step of a maximal coding rate reduction objective(MCR2). Specifically, we derive a novel variational form of the MCR2 objec-tive and show that the architecture that results from unrolled gradient descent of this variational objective leads to a new attention module called Token Statistics Self-Attention(TSSA). TSSA has linear computational and memory complexity and radically departs from the typical attention architecture that computes pair-wise similarities between tokens. Experiments on vision, language, and long se-quence tasks show that simply swapping TSSA for standard self-attention, which we refer to as the Token Statistics Transformer(TOST), achieves competitive per-formance with conventional transformers while being significantly more compu-tationally efficient and interpretable. Our results also somewhat call into ques-tion the conventional wisdom that pairwise similarity style attention mechanisms are critical to the success of transformer architectures. Code will be available at https://github.com/RobinWu218/ToST.
### 2025.6.4
- 【不确定性感知图结构学习】研讨
  - 题目：Uncertainty-Aware Graph Structure Learning
  - 作者：Shen Han, Zhezheng Hao, Yan Feng, Zhiyao Zhou, Sheng Zhou, Gang Wang, Jiawei Chen*, Chun Chen, Can Wang
  - 单位：The State Key Laboratory of Blockchain and Data Security, Zhejiang University, Hangzhou, China; Northwestern Polytechnical University, Xi'an, China; Bangsun Technology, Hangzhou, China
  - 解决问题：解决了现有图结构学习方法忽视节点信息质量和过度依赖对称图结构的问题。
  - 关键思路：UnGSL通过估计节点信息的不确定性并利用它来调整有向连接的强度，从而自适应地减少高不确定性节点的影响，并学习非对称图结构。
  - 摘要：Graph Neural Networks(GNNs) have become a prominent approach for learning from graph-structured data. However, their effective-ness can be significantly compromised when the graph structure is suboptimal. To address this issue, Graph Structure Learning(GSL)has emerged as a promising technique that refines node connec-tions adaptively. Nevertheless, we identify two key limitations in existing GSL methods: 1) Most methods primarily focus on node similarity to construct relationships, while overlooking the quality of node information. Blindly connecting low-quality nodes and aggregating their ambiguous information can degrade the perfor-mance of other nodes. 2) The constructed graph structures are often constrained to be symmetric, which may limit the model's flexibility and effectiveness.
To overcome these limitations, we propose an Uncertainty-aware Graph Structure Learning(UnGSL) strategy. UnGSL esti-mates the uncertainty of node information and utilizes it to adjust the strength of directional connections, where the influence of nodes with high uncertainty is adaptively reduced. Importantly,UnGSL serves as a plug-in module that can be seamlessly integrated into existing GSL methods with minimal additional computational cost. In our experiments, we implement UnGSL into six representative GSL methods, demonstrating consistent performance improvements. The code is available at https://github.com/UnHans/UnGSL.

- 【联合图重连与特征去噪】研讨
  - 题目：JOINT GRAPH REWIRING AND FEATURE DENOISING VIA SPECTRAL RESONANCE
  - 作者：Jonas Linkerhagner, Cheng Shi, Ivan Dokmani
  - 单位：Department of Mathematics and Computer Science, University of Basel, Basel, Switzerland
  - 解决问题：提出了一个算法来联合去噪特征和重连图，以提高下游节点分类任务的性能。
  - 关键思路：JDR通过对齐图和特征矩阵的主导谱空间来近似解决相关的非凸优化问题，从而改进图和特征之间的对齐，提升图神经网络的性能。
  - 摘要：When learning from graph data, the graph and the node features both give noisy information about the node labels. In this paper we propose an algorithm to jointly denoise the features and rewire the graph(JDR), which improves the performance of downstream node classification graph neural nets(GNNs). JDR works by aligning the leading spectral spaces of graph and feature matrices. It approximately solves the associated non-convex optimization problem in a way that handles graphs with multiple classes and different levels of homophily or heterophily. We theoretically justify JDR in a stylized setting and show that it consistently outperforms existing rewiring methods on a wide range of synthetic and real-world node classification tasks.
### 2025.6.9
- 【初始token的注意力】研讨
  - 题目：ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training
  - 作者：Feijiang Han, Xiaodong Yu, Jianheng Tang, Lyle Ungar
  - 单位：University of Pennsylvania, AMD, Peking University
  - 解决问题：提出了一种无需训练的方法，通过调整大型语言模型（LLMs）中初始token的注意力来提升模型性能。
  - 关键思路：ZeroTuning通过理论分析和实验验证，发现初始token作为注意力汇聚点，其注意力的调整可以显著影响后续token的注意力分布，从而提升模型在多种任务上的表现。
  - 摘要：Training-free methods for enhancing large language models(LLMs) have attracted growing interest recently, with token-level attention tuning emerging as an in-terpretable and promising direction. However, existing methods typically rely on auxiliary mechanisms to identify important or irrelevant task-specific tokens,introducing potential bias and limiting applicability. In this work, we uncover a sur-prising and elegant alternative: the semantically empty initial token Llama) serves as a powerful and underexplored control point for optimizing model behavior. Through theoretical analysis, we show that tuning the initial token's attention sharpens or flattens the attention distribution over subsequent tokens, and its role as an attention sink amplifies this effect. Empirically, we find that:(1)tuning its attention improves LLM performance across tasks more effectively than tuning other task-specific tokens;(2) the effect follows a consistent trend across layers, with earlier layers having greater impact, but varies across attention heads,with different heads showing distinct preferences in how they attend to this token. Based on these findings, we propose ZeroTuning, a training-free approach that improves LLM performance by applying head-specific attention adjustments to this special token. Despite tuning only one token, ZeroTuning achieves higher average performance on text classification, multiple-choice QA, and multi-turn con-versation tasks across models such as LLama, Qwen, and DeepSeek. For example, ZeroTuning improves Llama-3.1-8B by 11.71% on classification tasks,2.64% on QA tasks, and raises its multi-turn score from 7.804 to 7.966. The method is also robust to limited resources, few-shot settings, long contexts, quantization, decoding strategies, and prompt variations. Our work sheds light on a previously overlooked control point in LLMs, offering new insights into both inference-time tuning and model interpretability.

- 【机器人检测】研讨
  - 题目：Boosting Bot Detection via Heterophily-Aware Representation Learning and Prototype-Guided Cluster Discovery
  - 作者：Buyun He, Xiaorui Jiang, Qi Wu, Hao Liu, Yingguang Yang, Yong Liao
  - 单位：of China, Hefei, China
  - 解决问题：解决了现有基于图的机器人检测方法在标签依赖和跨社区泛化能力方面的局限性，特别是在面对交互伪装和分布式部署的挑战时。
  - 关键思路：BotHP通过引入异质性感知的表示学习和原型引导的聚类发现，利用双编码器架构同时建模同质性和异质性模式，并通过预训练和微调方案增强检测性能和泛化能力。
  - 摘要：Detecting social media bots is essential for maintaining the se-curity and trustworthiness of social networks. While contempo-rary graph-based detection methods demonstrate promising results,their practical application is limited by label reliance and poor gener-alization capability across diverse communities. Generative Graph Self-Supervised Learning(GSL) presents a promising paradigm to overcome these limitations, yet existing approaches predomi-nantly follow the homophily assumption and fail to capture the global patterns in the graph, which potentially diminishes their effectiveness when facing the challenges of interaction camouflage and distributed deployment in bot detection scenarios. To this end,we propose BotHP, a generative GSL framework tailored to boost graph-based bot detectors through heterophily-aware representa-tion learning and prototype-guided cluster discovery. Specifically,BotHP leverages a dual-encoder architecture, consisting of a graph-aware encoder to capture node commonality and a graph-agnostic encoder to preserve node uniqueness. This enables the simulta-neous modeling of both homophily and heterophily, effectively countering the interaction camouflage issue. Additionally, BotHP incorporates a prototype-guided cluster discovery pretext task to model the latent global consistency of bot clusters and identify spa-tially dispersed yet semantically aligned bot collectives. Extensive experiments on two real-world bot detection benchmarks demon-strate that BotHP consistently boosts graph-based bot detectors,improving detection performance, alleviating label reliance, and enhancing generalization capability.
### 2025.6.17
- 【非同质图预训练与提示学习】研讨
  - 题目：Non-Homophilic Graph Pre-Training and Prompt Learning
  - 作者：Xingtong Yu, Yuan Fang, Jie Zhang, Renhe Jiang
  - 单位：Singapore Management University, National University of Singapore, The University of Tokyo
  - 解决问题：解决了现有图预训练和提示学习方法未区分同质性和异质性图特征的问题，特别是在非同质图上的表现不足。
  - 关键思路：ProNoG框架通过条件网络生成节点特定的提示，以捕捉每个节点的非同质性特征，并在预训练阶段选择非同质性任务以提高模型适应性。
  - 摘要：Graphs are ubiquitous for modeling complex relationships between objects across various fields. Graph neural networks(GNNs) have become a mainstream technique for graph-based applications, but their performance heavily relies on abundant labeled data. To re-duce labeling requirement, pre-training and prompt learning has become a popular alternative. However, most existing prompt meth-ods do not distinguish between homophilic and heterophilic char-acteristics in graphs. In particular, many real-world graphs are non-homophilic-neither strictly nor uniformly homophilic-as they exhibit varying homophilic and heterophilic patterns across graphs and nodes. In this paper, we propose ProNoG, a novel pre-training and prompt learning framework for such non-homophilic graphs. First, we examine existing graph pre-training methods, providing in-sights into the choice of pre-training tasks. Second, recognizing that each node exhibits unique non-homophilic characteristics, we pro-pose a conditional network to characterize node-specific patterns in downstream tasks. Finally, we thoroughly evaluate and analyze ProNoG through extensive experiments on ten public datasets.

- 【多轮越狱框架】研讨
  - 题目：Tempest: Automatic Multi-Turn Jailbreaking of Large Language Models with Tree Search
  - 作者：Andy Zhou, Ron Arel
  - 单位：Intology AI
  - 解决问题：解决了现有单轮越狱方法无法有效模拟真实世界攻击者通过多轮对话逐步侵蚀大语言模型（LLM）安全边界的问题。
  - 关键思路：Tempest通过树搜索策略同时探索多条对抗路径，并结合部分合规性跟踪机制，系统性地利用模型在对话中的渐进式安全漏洞累积，从而实现高效的多轮越狱攻击。
  - 摘要：We introduce Tempest, a multi-turn adversarial framework that models the gradual erosion of Large Language Model(LLM) safety through a tree search perspective. Unlike single-turn jail-breaks that rely on one meticulously engineered prompt, Tempest expands the conversation at each turn, branching out multiple adversarial prompts that exploit partial compliance from previous responses. Through a cross-branch learning mechanism, successful attack patterns and partial compliance signals are systemati-cally shared across parallel conversation paths,enabling more efficient discovery of model vul-nerabilities. By tracking these incremental pol-icy leaks and re-injecting them into subsequent queries, Tempest reveals how minor conces-sions can accumulate into fully disallowed out-puts. Evaluations on the JailbreakBench dataset show that Tempest achieves a 100% success rate on GPT-3.5-turbo and 97% on GPT-4 in a single multi-turn run, significantly outperform-ing both single-turn methods and multi-turn baselines such as Crescendo or GOAT while using fewer queries. This tree search method-ology offers an in-depth view of how model safeguards degrade over successive dialogue turns, demonstrating that exploring multiple conversation paths simultaneously is crucial for comprehensive safety testing of language models.
### 2025.6.25
- 【多通道解缠图神经网络】研讨
  - 题目：Multi-Channel Disentangled Graph Neural Networks with Different Types of Self-constraints
  - 作者：Zhuomin Liang, Liang Bai, Xian Yang, Jiye Liang Fellow, IEEE
  - 单位：School of Computer and Information Technology, Shanxi University, China
  - 解决问题：解决了多通道图神经网络在整合不同类型自监督信息时存在的冲突和信息丢失问题。
  - 关键思路：通过解缠表示学习将节点表示分为共享和互补部分，并分别施加一致性约束、图重建约束和对齐约束，以有效整合多通道的自监督信息。
  - 摘要：Graph Neural Network(GNN) is a popular semi-supervised graph representation learning method, whose perfor-mance strongly relies on the quality and quantity of labeled nodes. Given the insufficiency of labeled nodes in many real applications, many multi-channel GNNs have been developed to extract self-supervised information by leveraging consistency and complementarity among augmented graphs from different channels. However, these methods often struggle to balance conflicting self-supervised constraints, enhancing certain types of information at the expense of others. To tackle this problem,we propose a Multi-channel Disentangled Graph Neural Net-work(MD-GraphNet), which effectively classifies self-supervised constraints by learning disentangled representations. Specifically,our model enforces consistency constraints for shared represen-tations, graph reconstruction constraints for complementary(or private) representations, and aligning constraints for fused repre-sentations. Our model overcomes the confusion and loss problems of different types of self-supervised signals. Experimental results on benchmark datasets demonstrate the effectiveness of MD-GraphNet for semi-supervised node classification.

- 【知识冲突缓解框架】研讨
  - 题目：Micro-Act: Mitigate Knowledge Conflict in Question Answering via Actionable Self-Reasoning
  - 作者：Nan Huo, Jinyang Li, Bowen Qin, Ge Qu, Xiaolong Li, Xiaodong Li, Chenhao Ma, Reynold Cheng
  - 单位：The University of Hong Kong, BAAI, Xiamen University, The Chinese University of Hong Kong, Shenzhen
  - 解决问题：解决了检索增强生成（RAG）系统中因检索到的外部知识与大型语言模型（LLMs）的固有参数知识之间的冲突而导致的性能下降问题。
  - 关键思路：MICRO-ACT通过层次化动作空间自动感知上下文复杂性，并将知识比较分解为细粒度的可操作步骤，从而实现更精确的冲突检测和缓解。
  - 摘要：Retrieval-Augmented Generation(RAG) systems commonly suffer from Knowledge Conflicts, where retrieved external knowledge contradicts the inherent, parametric knowledge of large language models(LLMs). It adversely affects performance on downstream tasks such as question answering(QA). Existing approaches often attempt to mitigate conflicts by directly comparing two knowledge sources in a side-by-side manner, but this can overwhelm LLMs with extraneous or lengthy contexts, ultimately hindering their ability to identify and mitigate inconsistencies. To address this issue, we propose MICRO-ACT, a framework with a hierarchical action space that automatically perceives context complexity and adaptively decomposes each knowledge source into a sequence of fine-grained comparisons. These comparisons are represented as actionable steps, enabling reasoning beyond the superficial context. Through extensive experiments on five benchmark datasets, MICRO-ACT consistently achieves significant increase in QA accuracy over state-of-the-art baselines across all 5 datasets and 3 conflict types, especially in temporal and semantic types where all baselines fail significantly. More importantly, MICRO-ACT exhibits robust performance on non-conflict questions simultaneously, highlighting its practical value in real-world RAG applications. Code can be found at https://github.com/Nan-Huo/Micro-Act.
### 2025.7.1
- 【位置编码】研讨
  -  题目：PaTH Attention: Position Encoding via Accumulating Householder Transformations
  -  作者：Songlin Yang, Yikang Shen, Kaiyue Wen, Shawn Tan, Mayank Mishra, Liliang Ren, Rameswar Panda, Yoon Kim
  -  单位： Massachusetts Institute of Technology ，MIT-IBM Watson AI Lab 3Stanford University 4Microsoft
  -  解决问题：解决了RoPE（Rotary Position Encoding）在表达能力上的局限性，特别是在需要顺序推理的任务中表现不佳的问题。
  -  关键思路：PaTH通过累积数据依赖的Householder-like变换来动态调整位置编码，从而增强Transformer的表达能力，并开发了高效的训练和推理算法。
  -  摘要：The attention mechanism is a core primitive in modern large language models(LLMs) and AI more broadly. Since attention by itself is permutation-invariant,position encoding is essential for modeling structured domains such as language.Rotary position encoding(RoPE) has emerged as the de facto standard approach for position encoding and is part of many modern LLMs. However, in RoPE the key/query transformation between two elements in a sequence is only a function of their relative position and otherwise independent of the actual input. This limits the expressivity of RoPE-based transformers. This paper describes PaTH, a flexible data-dependent position encoding scheme based on accumulated products of Householder(like) transformations, where each transformation is data-dependent,i.e., a function of the input. We derive an efficient parallel algorithm for training through exploiting a compact representation of products of Householder matrices,and implement a FlashAttention-style blockwise algorithm that minimizes I/O cost.Across both targeted synthetic benchmarks and moderate-scale real-world language modeling experiments, we find that PaTH demonstrates superior performance compared to RoPE and other recent baselines.

- 【Mean Flows for One-step Generative Modeling】研讨
  - 题目：Mean Flows for One-step Generative Modeling
  - 作者：Zhengyang Geng，Mingyang Deng ，Xingjian Bai，J. Zico Kolter，Kaiming He 
  - 单位： CMU ，MIT
  - 解决问题：提出了一种新的框架，用于解决一步生成模型中训练不稳定和需要复杂预处理的问题。
  - 关键思路：引入平均速度的概念，构建一个内在的关系来指导神经网络训练，从而实现更稳定和高效的一步生成。
  - 摘要：We propose a principled and effective framework for one-step generative modeling.We introduce the notion of average velocity to characterize flow fields, in contrast to instantaneous velocity modeled by Flow Matching methods. A well-defined identity between average and instantaneous velocities is derived and used to guide neural network training. Our method, termed the MeanFlow model, is self-contained and requires no pre-training, distillation, or curriculum learning. MeanFlow demon-strates strong empirical performance: it achieves an FID of 3.43 with a single function evaluation(1-NFE) on ImageNet 256x256 trained from scratch, signifi-cantly outperforming previous state-of-the-art one-step diffusion/flow models. Our study substantially narrows the gap between one-step diffusion/flow models and their multi-step predecessors, and we hope it will motivate future research to revisit the foundations of these powerful models.
### 2025.7.8
- 【XAAttention框架】研讨
  -  题目：XAAttention: Block Sparse Attention with Antidiagonal Scoring
  -  作者：Ruyi Xu*, Guangxuan Xiao*, Haofeng Huang, Junxian Guo, Song Han​​
  -  单位：*MIT, ​​MIT-IBM Watson AI Lab, Amazon Science Hub, MIT AI Hardware Program, National Science Foundation, Hyundai, Samsung
  -  解决问题：解决了长上下文Transformer模型中注意力机制的二次复杂度导致的计算成本高昂问题，同时平衡了准确性和效率。
  -  关键思路：XAAttention通过利用注意力矩阵中反对角线值的总和作为块重要性的代理，实现了对非必要块的精确识别和剪枝，从而显著加速推理过程。
  -  摘要：Long-Context Transformer Models(LCTMs) are vital for real-world applications but suffer high computational costs due to attention's quadratic complexity. Block-sparse attention mitigates this by focusing computation on critical regions, yet existing methods struggle with balancing accu-racy and efficiency due to costly block impor-tance measurements. In this paper, we intro-duce XAttention, a plug-and-play framework that dramatically accelerates long-context inference in Transformers models using sparse attention.XAAttention's key innovation is the insight that the sum of antidiagonal values(i.e., from the lower-left to upper-right) in the attention ma-trix provides a powerful proxy for block impor-tance. This allows for precise identification and pruning of non-essential blocks, resulting in high sparsity and dramatically accelerated inference.Across comprehensive evaluations on demand-ing long-context benchmarks-including RULER and LongBench for language, VideoMME for video understanding, and VBench for video gener-ation-XAttention achieves accuracy compara-ble to full attention while delivering substantial computational gains. We demonstrate up to 13.5 acceleration in attention computation. These re-sults underscore XAAttention's ability to unlock the practical potential of block sparse attention,paving the way for scalable and efficient deploy-ment of LCTMs in real-world applications.

- 【人像抠图框架】研讨
  - 题目：EFormer: Enhanced Transformer towards Semantic-Contour Features of Foreground for Portraits Matting
  - 作者：Zitao Wang, Qiguang Miao, Peipei Zhao, Yue Xi
  - 单位：
  - 解决问题：解决了现有基于Transformer的人像抠图方法在捕捉高频轮廓信息方面的不足，导致边缘预测不准确的问题。
  - 关键思路：EFormer通过引入跨注意力模块在不同分辨率特征之间引导模型自主定位和捕捉高频细节特征，同时结合语义和轮廓检测器（SCD）以及特征提取分支，分别提取精细的高频轮廓特征和完整的低频语义信息，最终融合这两种特征生成更精确的人像抠图。
  - 摘要：The portrait matting task aims to extract an alpha matte with complete semantics and finely-detailed contours. In comparison to CNN-based approaches, transformers with self-attention module have a better capacity to capture long-range dependencies and low-frequency semantic in-formation of a portrait. However, the recent research shows that self-attention mechanism struggles with model-ing high-frequency contour information and capturing fine contour details, which can lead to bias while predicting the portrait's contours. To deal with this issue, we pro-pose EFormer to enhance the model's attention towards both of the low-frequency semantic and high-frequency con-tour features. For the high-frequency contours, our re-search demonstrates that cross-attention module between different resolutions can guide our model to allocate atten-tion appropriately to these contour regions. Supported on this, we can successfully extract the high-frequency detail information around the portrait's contours, which are pre-viously ignored by self-attention. Based on cross-attention module, we further build a semantic and contour detector(SCD) to accurately capture both of the low-frequency se-mantic and high-frequency contour features. And we de-sign contour-edge extraction branch and semantic extrac-tion branch to extract refined high-frequency contour fea-tures and complete low-frequency semantic information, re-spectively. Finally, we fuse the two kinds of features and leverage segmentation head to generate a predicted portrait matte. Experiments on VideoMatte240K(JPEG SD Format)and Adobe Image Matting(AIM) datasets demonstrate that EFormer outperforms previous portrait matte methods.
