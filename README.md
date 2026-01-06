## Master’s Thesis & Scientific Publication

This repository contains the code and experiments accompanying a **master’s thesis and scientific publication** on **adversarial attacks in semantic segmentation**.  
The work proposes a **robust detection approach based on uncertainty measures**.

### Overview
- Demonstrates that **model uncertainty (entropy)** is a reliable signal for distinguishing clean inputs from adversarially perturbed images.
- Proposes and evaluates a **lightweight post-processing detector** that operates **without modifying the underlying segmentation model**.
- Provides a **comprehensive analysis of multiple adversarial attack methods** and **state-of-the-art semantic segmentation architectures**.

### Core Contributions
- Formulation of an **uncertainty-based detection method** for adversarial examples in semantic segmentation.
- Extensive **empirical evaluation across diverse attack types** and segmentation models.
- Demonstration of **efficiency and practical applicability** without requiring model retraining or architectural changes.

### Technologies & Concepts
- Python  
- PyTorch  
- MMSegmentation  
- Deep Learning  
- Semantic Segmentation  
- Uncertainty Estimation  
- Adversarial Machine Learning  
- Entropy-based Analysis

### Publication
- **Detecting Adversarial Attacks in Semantic Segmentation via Uncertainty Estimation: A Deep Analysis**  
  arXiv preprint: https://arxiv.org/abs/2408.10021

## Attribution and Prior Work
This repository contains original implementations as well as **partial adaptations** of adversarial attack algorithms
from prior research works.
In particular, **selected components of certain attack implementations** are adapted from publicly available research code
by Jérôme Rony and collaborators:

- Jérôme Rony et al., *"Proximal Splitting Adversarial Attacks for Semantic Segmentation"*
- GitHub repository: https://github.com/jeromerony

All adapted components have been modified and extended for uncertainty-based analysis
and semantic segmentation experiments. 
