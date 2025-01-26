ScoringPy
========================================

ScoringPy is an open-source Python library designed to streamline the development and deployment of classical credit scorecards. It simplifies the entire process from data preprocessing to scorecard scaling by providing robust tools and methods that ensure data integrity and model performance. By incorporating multiple layers of data anomaly detection, ScoringPy minimizes errors before model training, enhancing efficiency and reliability.
The library is divided into three main components:

1. **Data Preprocessing Pipeline**: Automate and save data manipulation steps using pipelines, ensuring consistency and efficiency when validating models or testing new data.

2. **Feature and Binning Selection**: Perform both automatic and manual feature selection and binning, with comprehensive reports and visualizations based on Weight of Evidence (WoE) analysis.

3. **Scorecard Deployment and Scaling**: Deploy and scale scorecards with customizable parameters, allowing for precise control over model coefficients and constants.

By using ScoringPy, you can build robust credit scoring models with ease, reduce error rates, and enhance efficiency throughout the credit scoring process.
Features
--------

- **Data Preprocessing with Pipeline**: Automate and save every data manipulation step using a pipeline, which can be easily reapplied to new data. This ensures consistent preprocessing and reduces the likelihood of errors.
- **Feature Selection with WoE Analysis**: Generate detailed reports and visualizations for each feature based on WoE analysis.
- **Binning (Manual and Automatic)**: Bin continuous features for classical scoring models.
- **Final Data Transformation**: Apply a second layer of protection against outlier data.
- **Scorecard Deployment and Scaling**: Scale scores based on the model's coefficients and constants.
- **Performance Testing**: Easily test the scorecard's performance on different data populations using the preprocessing pipelines.
- **Monitoring**: Track scorecard and population performance over time.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Usage
   Best Practices and Detailed information
   Conclusion
   Contribution and Support

