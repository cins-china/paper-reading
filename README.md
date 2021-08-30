# Paper-Reading



# Details
[:arrow_up:](#table-of-contents)

#### 2021.08
 <details/>
<summary/>
  <a href="https://transacl.org/ojs/index.php/tacl/article/view/1853">Spanbert: Improving pre-training by representing and predicting spans</a>  --- SpanBERT--- by<i> Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, Omer Levy
</a>(<a href="https://github.com/facebookresearch/SpanBERT">Github</a>)</summary><blockquote><p align="justify">
We present SpanBERT, a pre-training method that is designed to better represent and predict spans of text. Our approach extends BERT by (1) masking contiguous random spans, rather than random tokens, and (2) training the span boundary representations to predict the entire content of the masked span, without relying on the individual token representations within it. SpanBERT consistently outperforms BERT and our better-tuned baselines, with substantial gains on span selection tasks such as question answering and coreference resolution. In particular, with the same training data and model size as BERT-Large, our single model obtains 94.6% and 88.7% F1 on SQuAD 1.1 and 2.0 respectively. We also achieve a new state of the art on the OntoNotes coreference resolution task (79.6% F1), strong performance on the TACRED relation extraction benchmark, and even gains on GLUE.

  主要贡献：Span Mask机制，不再对随机的单个token添加mask，随机对邻接分词添加mask；Span Boundary Objective(SBO)训练，使用分词边界表示预测被添加mask分词的内容；一个句子的训练效果更好。

</p></blockquote></details>

