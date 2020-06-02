# Machine Learning

## Basic

#### Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

-  ICML 2015 Google
- internal covariate shift
  - DNN中因为每层neural的变化，数据分布会不断变化。类比蝴蝶效应。模型深层每次输入的数据分布可能方差非常大。大大增加了训练收敛的难度。
- batch norm
  - 在每一层，计算mini batch mean & mini batch var。然后进行norm变换。
  - 引入两个可学习参数gamma, beta。 对norm变换以后的x进行一次线性变换(当然也就拥有了可以变回原样的能力)。
  - 优势：范化性好，收敛快，一定程度可以取代dropout

#### DECOUPLED WEIGHT DECAY REGULARIZATION

-  ICLR (Poster) 2019 
- weight decay 是L2 norm的等价形式（仅限SGD，即学习率全局不变，因为等价的公式中，学习率是耦合在里面的）
- 在adaptive gradient方法中，将weight decay 和 L2 norm decouple，可以获得更好的范化能力
- 虽然adam本身自带学习率衰减，论文还是指出解耦weight decay后（adamW），仍可以获得15%的指标提升

#### An overview of gradient descent optimization algorithms

- CoRR 2016
- Gradient descent总结性文章，主要包括BGD, SGD, Mini-batch GD, Momentum, Nesterov Accelerated Gradient, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam
- 提供了一些SGD优化推荐，如Shuffle/Curriculum, BN, Early stopping, Gradient Gaussion Noise 

#### Understanding deep learning requires rethinking generalization

- ICLR 2017 Best Paper Google
- 不觉得这篇凭什么能拿best paper，都是一些很表象的东西
- 深度网络能很容易的拟合随机数据
- 正则化对模型泛化能力不起决定作用 (百分位的提升都不叫提升了？)

#### Understanding the difficulty of training deep feedforward neural networks

- AISTATS 2010 大名鼎鼎的Xavier Glorot初始化
- 对于激活函数
  - 已知sigmoid会降低学习效率（ none-zero mean that induces important singular values in the Hessian）、饱和后带来梯度消失等问题
  - 文章对比了softmax, tanh, softsign在5层NN中不同层的数值变化，论证了sigmoid一开始基本都是在做无用功，前三层一开始都在随机输出，最后一层直接陷入饱和状态，在迭代100次以后才逃出来真正开始训练。其中softsign表现最好
- 对于初始化方法
  - 近似推导了一个[初始化方法](https://blog.csdn.net/u011534057/article/details/51673458)，能够尽量保证在NN传递时，W的均值为0，方差不变。方法简单但是效果异常得好。也就是目前很多机器学习工具包集成的 Xavier initialization

#### Variational Inference: A Review for Statisticians

- CoRR 2016
- 总结性文章，一些基础知识点：Markov chain, Monte Carlo sampling, Gibbs sampling,  Kullback-Leibler divergence, mean-field theory, ELBO, CAVI

XGBoost: A Scalable Tree Boosting System

Improving Generalization Performance by Switching from Adam to SGD

ON THE CONVERGENCE OF ADAM AND BEYOND

The Marginal Value of Adaptive Gradient Methods in Machine Learning

## Recsys

DeepWalk: Online Learning of Social Representations

Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph

LINE: Large-scale Information Network Embedding

node2vec: Scalable Feature Learning for Networks 

Structural Deep Network Embedding

Learning Deep Structured Semantic Models for Web Search using Clickthrough Data

Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts

Learning Tree-based Deep Model for Recommender Systems

Joint Optimization of Tree-based Index and Deep Model for Recommender Systems

A Learning-rate Schedule for Stochastic Gradient Methods to Matrix Factorization

A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation

A User-Centered Concept Mining System for Query and Document Understanding at Tencent

Ad Click Prediction: a View from the Trenches

Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR prediction

Behavior Sequence Transformer for E-commerce Recommendation in Alibaba

Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba

Causal Embeddings for Recommendation

Deep & Cross Network for Ad Click Predictions

Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features

Deep CTR Prediction in Display Advertising

Deep Interest Evolution Network for Click-Through Rate Prediction

Deep Neural Networks for YouTube Recommendations

Deep Session Interest Network for Click-Through Rate Prediction

DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate

Factorization Machines

Field-aware Factorization Machines for CTR Prediction

Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction

FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction

FLEN: Leveraging Field for Scalable CTR Prediction

Follow-the-Regularized-Leader and Mirror Descent: Equivalence Theorems and L1 Regularization

Image Feature Learning for Cold Start Problem in Display Advertising

From RankNet to LambdaRank to LambdaMART: An Overview

Learning and Transferring IDs Representation in E-commerce

Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction

Local Item-Item Models for Top-N Recommendation

Online Learning to Rank for Sequential Music Recommendation

Personalized Re-ranking for Recommendation

Practical Lessons from Predicting Clicks on Ads at Facebook

Real-time Personalization using Embeddings for Search Ranking at Airbnb

Recommending What Video to Watch Next: A Multitask Ranking System

Representation Learning-Assisted Click-Through Rate Prediction

Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations

Sparse Linear Methods with Side Information for Top-N Recommendations

SLIM: Sparse Linear Methods for Top-N Recommender Systems

Long and Short-Term Recommendations with Recurrent Neural Networks

Wide & Deep Learning for Recommender Systems

xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

A new similarity measure for collaborative filtering to alleviate the new user cold-starting problem

A Contextual-Bandit Approach to Personalized News Article Recommendation

A Simple Multi-Armed Nearest-Neighbor Bandit for Interactive Recommendation

Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches

ExcUseMe: Asking Users to Help in Item Cold-Start Recommendations

Feature-based factorized Bilinear Similarity Model for Cold-Start Top-n Item Recommendation

From Zero-Shot Learning to Cold-Start Recommendation

Item Cold-Start Recommendations: Learning Local Collective Embeddings

Neural Collaborative Filtering

Spectral Collaborative Filtering

Variational Autoencoders for Collaborative Filtering

## NLP

Adaptive Importance Sampling to Accelerate Training of a Neural Probabilistic Language Model

Notes on Noise Contrastive Estimation and Negative Sampling

Deep contextualized word representations

Improving Language Understanding by Generative Pre-Training

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS

RoBERTa: A Robustly Optimized BERT Pretraining Approach

Single Headed Attention RNN: Stop Thinking With Your Head

Pre-Training with Whole Word Masking for Chinese BERT

A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification

Attention Is All You Need

Bag of Tricks for Efficient Text Classification

Bidirectional LSTM-CRF Models for Sequence Tagging

Convolutional Neural Networks for Sentence Classification

ERNIE: Enhanced Language Representation with Informative Entities

GloVe: Global Vectors for Word Representation

Inference Methods for Latent Dirichlet Allocation

Latent Dirichlet Allocation

Network–Efficient Distributed Word2vec Training System for Large Vocabularies

Notes on Noise Contrastive Estimation and Negative Sampling

Parameter estimation for text analysis

SpanBERT: Improving Pre-training by Representing and Predicting Spans

The Dirichlet-multinomial distribution

Distributed Representations of Words and Phrases and their Compositionality

XLNet: Generalized Autoregressive Pretraining for Language Understanding



## CV

Fully Convolutional Networks for Semantic Segmentation

Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning

U-Net: Convolutional Networks for Biomedical Image Segmentation

UNet++: A Nested U-Net Architecture for Medical Image Segmentation

YOLOv3: An Incremental Improvement

CosFace: Large Margin Cosine Loss for Deep Face Recognition

FaceNet: A Unified Embedding for Face Recognition and Clustering



## VAE

Auto-Encoding Variational Bayes

Early Visual Concept Learning with Unsupervised Deep Learning

Semi-supervised Learning with Deep Generative Models

Stochastic Gradient VB and the Variational Auto-Encoder

Tutorial on Variational Autoencoders

## GAN

Generative Adversarial Nets

Are GANs Created Equal? A Large-Scale Study

Conditional Generative Adversarial Nets

UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS

f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization

Image-to-Image Translation with Conditional Adversarial Networks

InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets

Least Squares Generative Adversarial Networks

TOWARDS PRINCIPLED METHODS FOR TRAINING GENERATIVE ADVERSARIAL NETWORKS

Autoencoding beyond pixels using a learned similarity metric

Wasserstein Generative Adversarial Networks

## Others

![image](twodog.jpg)