Timeline Manager Documentation
==============================

Overview
--------

ScoringPy is an open-source Python library designed to streamline the development and deployment of classical credit scorecards. It simplifies the entire process from data preprocessing to scorecard scaling by providing robust tools and methods that ensure data integrity and model performance. By incorporating multiple layers of data anomaly detection, ScoringPy minimizes errors before model training, enhancing efficiency and reliability.

Features
--------

- **Data Preprocessing with Pipeline**: Automate and save every data manipulation step using a pipeline, which can be easily reapplied to new data. This ensures consistent preprocessing and reduces the likelihood of errors.
- **Feature Selection with WoE Analysis**: Generate detailed reports and visualizations for each feature based on WoE and Information Value (IV). This includes statistical summaries that help in understanding the predictive power of each feature.
- **Binning (Manual and Automatic)**: Bin continuous features for classical scoring models. Choose between manual binning or automatic suggestions provided by the library. Binning validation is included in the feature statistics report, checking if any data falls outside bin ranges.
- **Final Data Transformation**: Apply a second layer of protection against outlier data. The library alerts you if any data points fall outside the defined bin ranges during transformation.
- **Scorecard Deployment and Scaling**: Scale scores based on the model's coefficients and constants. The scaling is fully customizable, with default values optimized for most scaling scenarios.
- **Performance Testing**: Easily test the scorecard's performance on different data populations using the preprocessing pipelines.
- **Monitoring**: Track scorecard and population performance over time, leveraging the consistent preprocessing steps provided by the pipelines.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Usage

