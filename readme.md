
# ElasticGNN

This repository includes the official implementation of ElasticGNN in the paper **"Elastic Graph Neural Networks"** [ICML 2021].

[Xiaorui Liu](http://cse.msu.edu/~xiaorui/), [Wei Jin](http://cse.msu.edu/~jinwei2/), [Yao Ma](http://cse.msu.edu/~mayao4/), [Ming Yan](https://users.math.msu.edu/users/myan/), [Jiliang Tang](http://www.cse.msu.edu/~tangjili/) at el. [**Elastic Graph Neural Networks**](http://proceedings.mlr.press/v139/liu21k/liu21k.pdf).  

Related materials: [paper](http://proceedings.mlr.press/v139/liu21k.html), [slide](http://cse.msu.edu/~xiaorui/files/Slide_ElasticGNN.pdf), [poster](http://cse.msu.edu/~xiaorui/files/Poster_ElasticGNN.pdf)

![](https://raw.githubusercontent.com/lxiaorui/ElasticGNN/master/EMP.png)


## Abstract
While many existing graph neural networks (GNNs) have been proven to perform L2-based graph smoothing that enforces smoothness globally, in this work we aim to further enhance the local smoothness adaptivity of GNNs via L1-based graph smoothing. As a result, we introduce a family of GNNs (Elastic GNNs) based on L1 and L2-based graph smoothing. In particular, we propose a novel and general message passing scheme into GNNs. This message passing algorithm is not only friendly to back-propagation training but also achieves the desired smoothing properties with a theoretical convergence guarantee. Experiments on semi-supervised learning tasks demonstrate that the proposed Elastic GNNs obtain better adaptivity on benchmark datasets and are significantly more robust to graph adversarial attacks. 


## Reference
Please cite our paper if you find the paper or code to be useful. Thank you!

```
@InProceedings{liu2021elastic,
  title = 	 {Elastic Graph Neural Networks},
  author =       {Liu, Xiaorui and Jin, Wei and Ma, Yao and Li, Yaxin and Liu Hua and Wang, Yiqi and Yan, Ming and Tang, Jiliang},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  year = 	 {2021},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
}
```


## Requirement
* PyTorch
* PyTorch Geometric
* ogb (Evaluator)
* DeepRobust (optional)

## Examples
Normal setting:

```
$ cd code
$ python3 main.py --dataset Cora --random_splits 10 --runs 1 --lr 0.01 --dropout 0.8  --weight_decay 0.0005 --K 10 --lambda1 3 --lambda2 3
```

Robustness setting
```
$ cd code
$ python3 main.py --dataset Cora-adv --random_splits 1 --runs 10 --lr 0.01 --K 10 --lambda1 9 --lambda2 3 --weight_decay 0.0005 --hidden 16 --normalize_features False --ptb_rate 0.1
```
