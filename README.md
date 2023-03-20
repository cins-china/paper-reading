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
- 王泓淏 Assassinswhh（https://github.com/Assassinswhh）
- 朱鑫鹏 CharlieZhuZhu（https://github.com/CharlieZhuZhu）
- 沈雅文 Cheesepackage （https://github.com/Cheesepackage）
- 蒋玉洁 Thealuv (https://github.com/Thealuv)
- 李雅杰 Echo-yajie（https://github.com/Echo-yajie）
- 高美琳 Lumenlin（https://github.com/Lumenlin）
- NomanChowdhury（https://github.com/NomanChowdhury）



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
### 2023.02.27
- 【方面级情感分析综述】研讨
   - 分享汇报：李攀
   - 动机：细粒度方面级情感分析任务杂乱，缺乏统一的分类，以及总结统一的建模范式
   - 难点：如何学习各情感要素的表征
   - 方法总结：通常采用四种建模范式（Token-level、Sequence-level、MRC、Sequence-to-Sequence）和管道方法来解决该类任务
   - 认识：对ABSA任务的建模方法总结，以及更好表征ABSA任务，对ASTE任务建模有新的启发
### 2023.3.13

- 【分子属性预测综述】
   - 分享汇报： 王泓淏
   - 动机：帮助医疗和化学领域
   - 难点：如何获取一个良好的图级别表征
   - 方法总结：2D分子表征， 3D分子表征， 对比学习相互补充
   - 对自己方向的认识：2D和3D分子相互补充，如何编码3D图里的信息，消息传递机制设计与改良

- 【图神经网络节点分类综述】
   - 分享汇报： 张阿聪
   - 动机：解决图神经网络的异质性问题
   - 难点：如何寻找更优的高阶邻居节点，如何聚合高阶邻居节点信息
   - 方法总结：通过目标节点和高阶邻居的表征的相似性选取高阶邻居节点；利用拼接或者筛选表征维度聚合信息

