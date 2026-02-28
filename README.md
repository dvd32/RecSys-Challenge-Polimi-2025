# **🏆 1st Place Solution \- Polimi RecSys Challenge 2025/26**

**Team Polenta**: [Alessio Pizzini](https://www.linkedin.com/in/alessio-pizzini-abb009252/) & [David Ravelli](https://www.linkedin.com/in/david-ravelli-2900a6252/)

**Course**: Recommender Systems 2025/26 @ Politecnico di Milano

This repository contains the winning solution for the Polimi RecSys Challenge 2025/26. We achieved **1st place** on both the Public and Private leaderboards out of 71 competing teams.

## **📊 Final Results**

**Ranked \#1** in both Public and Private Leaderboards **among 71 teams**, with the following scores:

* **Public Leaderboard (Recall@20)**: 0.53331  
* **Private Leaderboard (Recall@20)**: 0.53149

## **🎯 Challenge Overview**

The objective of the competition was to recommend the best 20 items for a target set of users based on implicit feedback data. The dataset consisted of:

* **Interactions**: \~3.04 Million  
* **Unique Users**: 27,095  
* **Unique Items**: 6,969  
* **Data Characteristics**: Strong popularity bias, cold-start users, and high sparsity.  
* **Evaluation metric:** RECALL@20

## **🏗️ System Architecture**

Our solution relies on an optimized **Two-Stage Recommender System pipeline**.

### **Stage 1: Candidate Generation (Retrieval)**

To maximize the theoretical *Recall Ceiling* while maintaining a manageable number of candidates, we utilized a diverse ensemble of base recommenders. Instead of a uniform cutoff, we applied a **Greedy Selection strategy** to find a personalized cutoff for each model based on its marginal precision.

Base models included:

* **Linear/Neighborhood**: SLIM Elastic Net (Positive & Negative), EASE\_R, ItemKNN TFIDF  
* **Graph-Based**: RP3Beta (Sharp & High Precision variants)  
* **Matrix Factorization**: iALS (Implicit), LightFM, PureSVD  
* **Deep Learning**: Mult-VAE (PyTorch)

### **Stage 2: Reranking with XGBoost**

The generated candidates were fed into an **XGBoost Ranker** using the rank:pairwise objective function.

To ensure extreme robustness and prevent overfitting to the public leaderboard, we implemented a **5-Fold Cross-Validation Bagging** strategy:

* We trained 5 distinct XGBoost models on different folds of the training data.  
* The final predictions are an ensemble (average) of these 5 models.  
* **Stability**: The Pearson correlation across the folds was an outstanding **0.998**, proving the model successfully learned generalized user behavior.

## **⚙️ Feature Engineering & High Performance**

We engineered **90 features** to help XGBoost rank the candidates effectively.

### **Advanced Features:**

* **Base Model Signals**: Normalized scores, raw scores, ranks, and normalized ranks from the Stage 1 models.  
* **Model Agreement**: A consensus feature indicating how many different algorithms recommended a specific item.  
* **Latent Factors**: Low-dimensional vector embeddings extracted from a PureSVD model to capture deep user-item affinities.  
* **User/Item Stats**: Item popularity, user profile length.  
* **Mainstreamness & Diversity**: Custom metrics to identify whether a user is a "trend-follower" (mainstream) or a "niche-seeker" (diverse).

### **🚀 High-Performance Computing with Numba**

Calculating complex **similarity statistics** (Max, Min, Mean, Standard Deviation, Skewness, Kurtosis) between candidate items and an entire user's history across millions of rows is computationally expensive in native Python.

To be able to compute these features in a reasonable time, we utilized the **Numba** library (@njit(parallel=True)). This allowed us to:

* Parallelize cycle computations across all available CPU cores.  
* Just-In-Time (JIT) compile Python code to optimized machine code.  
* Efficiently capture "Strong Match" signals via High Skewness or Max Similarity weights.

## **💻 Tech Stack**

* **Python 3.11**  
* **XGBoost**: For the pairwise learning-to-rank reranker.  
* **Numba**: For parallelized, high-speed feature extraction.  
* **Implicit & LightFM**: For fast matrix factorization algorithms.  
* **PyTorch**: For implementing the Mult-VAE neural network.  
* **Pandas, NumPy, SciPy**: For data manipulation and sparse matrix handling.  
* [**Maurizio Ferrari Dacrema’s repository**](https://github.com/remaplab/RecSys_Course_AT_PoliMi)**:** Providing the baseline framework.
