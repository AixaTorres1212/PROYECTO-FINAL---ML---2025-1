# Chemotherapy Response Prediction in Breast Cancer Using VAE and Classification Models

This repository contains the code and scripts used to develop and evaluate a **pipeline for predicting chemotherapy response** in breast cancer patients, based on **RNA-seq gene expression** data from the **TCGA-BRCA** dataset.

## Overview

The high dimensionality of transcriptomic data poses a major challenge for classification models. To address this, we implemented a **Variational Autoencoder (VAE)** that generates a lower-dimensional latent representation while preserving the most relevant features for the predictive task.

From these latent representations, four classifiers were trained and compared:

* **XGBoost**
* **Random Forest** + **Logistic Regression**
* **Multilayer Perceptron (MLP)**
* **Support Vector Machine (SVM)**

The main goal is to distinguish between two clinical outcome classes:

* **pCR** (*pathologic Complete Response*): complete pathological response after chemotherapy.
* **nopCR** (*no pathologic Complete Response*): absence of complete response.

## Summary of Methodology

1. **Data Preprocessing**

   * Curation and filtering of TCGA-BRCA samples.
   * Assignment of binary labels (*pCR* / *nopCR*).
   * Normalization and log transformation of expression values.

2. **Dimensionality Reduction with VAE**

   * Architecture optimized for transcriptomic data.
   * Training for reconstruction and KL divergence regularization.
   * Extraction of latent variables.

3. **Classifier Training and Evaluation**

   * Stratified train/test split.
   * Metrics comparison: *accuracy*, *F1-score*, *AUC-ROC*.
   * k-fold cross-validation.

## Expected Outcomes

The repository documents comparative results, highlighting which model achieves the best balance between sensitivity and specificity, and providing evidence on the feasibility of using VAE as a preprocessing stage for classification tasks in clinical prediction problems with RNA-seq data.