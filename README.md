# SilFcast

This is the official implementation for the Similarity-Based Local Forecasting System (SilFcast) paper in ******.

# ABSTRACT

Time series forecasting is a critical tool for decision-making in a variety of domains. However, certain time series, particularly those representing social phenomena, present challenges due to irregular patterns and dynamic distributions. To address these complexities, the paper presents a Similarity-Based Local Forecasting System (SilFcast) designed to improve predictive accuracy and forecast interpretability.

SilFcast leverages a local learning approach by identifying and utilizing the most relevant subsequences for each prediction based on their similarity to the test pattern. By focusing on localized data, the system adapts to changing distributions and captures nuanced temporal dependencies, enabling improved forecasting performance.

The experimental results demonstrate that SilFcast outperforms traditional inductive learning methods and state-of-the-art models in the task of crime time series forecasting, establishing yet another possibility in the treatment of real-world univariate time series. Furthermore, the proposed method is generic and can be applied to multiple domains, offering a robust and interpretable framework for complex data prediction.

# REQUIREMENTS

ipykernel==6.29.5
ipython==8.27.0
matplotlib==3.9.2
matplotlib-inline==0.1.7
numpy==1.26.4
pandas==2.2.2
requests==2.32.3
scikit-learn==1.5.1
scikit-posthocs==0.9.0
scipy==1.14.1
seaborn==0.13.2

# Usage 

You can use main.py to run our SilFcast on your dataset.

```
python main.py
```

# Reference

If you use this code, please cite our paper:

```
******
```
