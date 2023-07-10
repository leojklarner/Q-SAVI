
![Q-SAVI: Drug Discovery under Covariate Shift with Domain-Informed Prior Distributions over Functions](./images/readme_header.png)

This repository contains an end-to-end pipeline to reproduce and extend the dataset curation, data shift quantification and empricial evaluation presented in the paper:

**_Drug Discovery under Covariate Shift with Domain-Informed Prior Distributions over Functions._** Leo Klarner, Tim G.J. Rudner, Michael Reutlinger, Torsten Schindler, Garrett M. Morris, Charlotte M. Deane, Yee Whye Teh **ICML 2023**.

<p align="center">
  &#151; <a href="https://proceedings.mlr.press/v202/klarner23a/klarner23a.pdf"><b>View Paper</b></a> &#151;
</p>

---

**Abstract**: Accelerating the discovery of novel and more effective therapeutics is an important pharmaceutical problem in which deep learning is playing an increasingly significant role. However, real-world drug discovery tasks are often characterized by a scarcity of labeled data and significant covariate shift—a setting that poses a challenge to standard deep learning methods. 
<img align="right" src="./images/graphical_abstract.png" width="400px"/>
In this paper, we present Q-SAVI, a probabilistic model able to address these challenges by encoding explicit prior knowledge of the data-generating process into a prior distribution over functions, presenting researchers with a transparent and probabilistically principled way to encode data-driven modeling preferences. Building on a novel, gold-standard bioactivity dataset that facilitates a meaningful comparison of models in an extrapolative regime, we explore different approaches to induce data shift and construct a challenging evaluation setup. We then demonstrate that using Q-SAVI to integrate contextualized prior knowledge of drug-like chemical space into the modeling process affords substantial gains in predictive accuracy and calibration, outperforming a broad range of state-of-the-art self-supervised pre-training and domain adaptation techniques. 

# Citation

If you found our paper or code useful for your research, please consider citing it as:

```
@InProceedings{klarner2023qsavi,
  title = {Drug Discovery under Covariate Shift with Domain-Informed Prior Distributions over Functions},
  author = {Klarner, Leo and Rudner, Tim G. J. and Reutlinger, Michael and Schindler, Torsten and Morris, Garrett M and Deane, Charlotte and Teh, Yee Whye},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {17176--17197},
  year = {2023},
  volume = {202},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
}
```