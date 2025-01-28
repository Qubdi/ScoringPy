Usage
=====

ScoringPy provides several modules, each designed for a specific part of the credit scoring process:

- **Processing**: For data preprocessing.
- **WoeAnalysis**: For feature selection and binning using WoE analysis.
- **WoeBinning**: For transforming data based on the selected features and bins.
- **CreditScoring**: For scaling scores and probabilities based on the model and scaling constants.

Below are detailed explanations and examples for each module.

Processing
----------

The **Processing** module automates data preprocessing steps using pipelines. Every transformation is saved and can be easily reapplied to new data, which is crucial for model validation and testing.

Pipeline Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

To create a processing pipeline, initialize it using the ``Processing`` class. You can enable or disable automatic data flow between steps using the ``flow`` parameter.

- **flow** (optional, default ``True``): If ``True``, the output from each function (step) will be passed as input to the next function automatically. If ``False``, you must manage data flow manually.

Type 1: Sequential Data Transformation with Automatic Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we'll create a pipeline with automatic data flow between steps:

.. code-block:: python

    from ScoringPy import Processing
    import pandas as pd
    import dill

    # Initialize the pipeline with flow control enabled
    pipeline = Processing(flow=True)

    # Define preprocessing functions
    def fill_missing_age(data):
        """Fill missing values in the 'Age' column with the mean."""
        data['Age'] = data['Age'].fillna(data['Age'].mean())
        return data

    def double_age(data):
        """Double the values in the 'Age' column."""
        data['Age'] = data['Age'] * 2
        return data

    def scale_age(data):
        """Scale the 'Age' column by dividing by 5."""
        data['Age'] = data['Age'] / 5
        return data

    # Add steps to the pipeline
    pipeline.add_step(fill_missing_age)
    pipeline.add_step(double_age)
    pipeline.add_step(scale_age)

    # Save the pipeline using dill
    with open('pipeline.pkl', 'wb') as file:
        dill.dump(pipeline, file)

    # Load your dataset
    df = pd.read_csv('data.csv')

    # Run the pipeline on the dataset
    df_processed = pipeline.run(initial_data=df)

    # Clear the pipeline if needed
    pipeline.clear()

Explanation
~~~~~~~~~~~

1. **Initialization**: We initialize the ``Processing`` pipeline with ``flow=True``, enabling automatic data flow between steps.

2. **Function Definitions**: We define three functions (``fill_missing_age``, ``double_age``, ``scale_age``) that perform specific data transformations.

3. **Adding Steps**: We add these functions to the pipeline using ``pipeline.add_step()``.

4. **Saving the Pipeline**: We use the ``dill`` library to serialize and save the pipeline for future reuse.

5. **Running the Pipeline**: We run the pipeline on the dataset using ``pipeline.run(initial_data=df)``.

6. **Clearing the Pipeline**: We clear the pipeline using ``pipeline.clear()`` if we need to reset it.



To create and use a processing pipeline, you can follow these approaches based on your requirements.

Type 1: Reusing the Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can load the saved pipeline and apply it to new data without redefining the steps:

.. code-block:: python

    import dill
    import pandas as pd

    # Load the saved pipeline
    with open('pipeline.pkl', 'rb') as file:
        pipeline = dill.load(file)

    # Load new data
    df_new = pd.read_csv('new_data.csv')

    # Run the pipeline on the new data
    df_processed_new = pipeline.run(initial_data=df_new)

    # Clear the pipeline if needed
    pipeline.clear()


Type 2: Non-Sequential Data Processing with Manual Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need more control over the data flow between steps, you can set ``flow=False`` when initializing the pipeline.

.. code-block:: python

    from ScoringPy import Processing
    import pandas as pd
    import dill

    # Initialize the pipeline without automatic flow
    pipeline = Processing(flow=False)

    # Define functions for each step
    def load_data_step1(path=None):
        """Load data from an Excel file."""
        data = pd.read_excel(path)
        return data

    def load_data_step2():
        """Load additional data from another Excel file."""
        data = pd.read_excel('Data/step2.xlsx')
        return data

    def concatenate_data():
        """Concatenate data from step 1 and step 2."""
        step1_data = pipeline.context.get('load_data_step1')
        step2_data = pipeline.context.get('load_data_step2')
        data = pd.concat([step1_data, step2_data], ignore_index=True)
        data['Age'] = data['Age'] * 2
        return data

    def finalize_data(data):
        """Finalize the data by scaling the 'Age' column."""
        data['Age'] = data['Age'] / 5
        return data

    # Add steps to the pipeline
    pipeline.add_step(load_data_step1, path='Data/step1.xlsx')
    pipeline.add_step(load_data_step2)
    pipeline.add_step(concatenate_data, flow=True)
    pipeline.add_step(finalize_data, flow=True)

    # Save the pipeline
    with open('pipeline.pkl', 'wb') as file:
        dill.dump(pipeline, file)

    # Run the pipeline
    df_processed = pipeline.run()

    # Clear the pipeline if needed
    pipeline.clear()

Explanation
~~~~~~~~~~~

1. **Initialization**:
   - We initialize the ``Processing`` pipeline with ``flow=False``, disabling automatic data flow.

2. **Function Definitions**:
   - We define functions for loading data and concatenating datasets.

3. **Using ``pipeline.context``**:
   - We use ``pipeline.context.get()`` to retrieve data from previous steps.

4. **Flow Control**:
   - We set ``flow=True`` for steps where we want the output to be passed to the next step.

Type 2: Reusing the Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can reuse a saved pipeline by loading it and applying it directly to the data:

.. code-block:: python

    import dill

    # Load the pipeline
    with open('pipeline.pkl', 'rb') as file:
        pipeline = dill.load(file)

    # Run the pipeline
    df_processed = pipeline.run()

    # Clear the pipeline if needed
    pipeline.clear()

Processing Optional Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**flow** *(bool, default True)*: Controls automatic data flow between steps. If set to ``False``, you must manage the data flow manually.

WoeAnalysis
-----------

The **WoeAnalysis** module is designed for feature selection and binning using WoE (Weight of Evidence) analysis. It provides small reports for each feature, including statistical summaries based on WoE analysis.

Methods
~~~~~~~

- **discrete**: Analyze discrete (categorical) variables.
- **continuous**: Analyze continuous variables.

Each method supports:

- **plot**: Visualizes WoE and IV analysis.
- **report**: Displays and optionally saves the report.

Analyzing Discrete Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ScoringPy import WoeAnalysis

    # Initialize WoeAnalysis
    woe_analysis = WoeAnalysis(save=False, path="Data/", type=2)

    # Analyze a discrete variable with safety checks
    woe_analysis.discrete(column="MaritalStatus", df=X_train, target=y_train, safety=True, threshold=300).report()

Explanation
~~~~~~~~~~~

1. **Initialization**: We initialize ``WoeAnalysis`` with optional parameters like ``save``, ``path``, and ``type``.

2. **Safety Parameters**:
    - **safety** (``bool``, default ``True``): Controls whether to perform safety checks on the feature before processing.
    - **threshold** (``int``, default ``300``): Specifies the maximum number of unique values allowed in a discrete feature.

3. **Analyzing the Variable**: We call the ``discrete`` method, passing the column name, DataFrame ``X_train``, target variable ``y_train``, and safety parameters.

4. **Generating the Report**: We call the ``report`` method to display the analysis.

Plotting and Saving Reports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate and save reports while analyzing discrete variables.

.. code-block:: python

    # Generate a plot and display the report
    woe_analysis.discrete(column="MaritalStatus", df=X_train, target=y_train, safety=True, threshold=300).plot(rotation=0).report()

    # Save the report
    woe_analysis.discrete(column="MaritalStatus", df=X_train, target=y_train, safety=True, threshold=300).report(save=True, type=1)

- **rotation**: Adjusts the rotation of x-axis labels in the plot.
- **save**: If True, saves the report.
- **type**: Specifies the format type when saving.

Analyzing Continuous Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For continuous variables, you need to define bins. You can use **auto** or **manual binning** methods for this purpose.

Auto Binning
^^^^^^^^^^^^

Automatically define bins for continuous variables and analyze them.

.. code-block:: python

    from ScoringPy import WoeAnalysis

    # Define bins using WoeAnalysis method
    bins = woe_analysis.auto_binning(column="RefinanceRate", n_bins=10, data=X_train, target=y_train, strategy_option=None)

    # Analyze a continuous variable
    woe_analysis.continuous(column="RefinanceRate", bins=bins, df=X_train, target=y_train).report()

    # Plot and display the report
    woe_analysis.continuous(column="RefinanceRate", bins=bins, df=X_train, target=y_train).plot(rotation=90).report()

    # Save the report
    woe_analysis.continuous(column="RefinanceRate", bins=bins, df=X_train, target=y_train).report(save=True)

Manual Binning
^^^^^^^^^^^^^^

Define custom bins for continuous variables and analyze them.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from ScoringPy import WoeAnalysis

    # Define bins using pandas IntervalIndex
    bins = pd.IntervalIndex.from_tuples([
        (-1, 0), (0, 0.2), (0.2, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, np.inf)])

    # Analyze a continuous variable
    woe_analysis.continuous(column="RefinanceRate", bins=bins, df=X_train, target=y_train).report()

    # Plot and display the report
    woe_analysis.continuous(column="RefinanceRate", bins=bins, df=X_train, target=y_train).plot(rotation=90).report()

    # Save the report
    woe_analysis.continuous(column="RefinanceRate", bins=bins, df=X_train, target=y_train).report(save=True)

Results
~~~~~~~

You can extract various attributes from the `woe_analysis` object for future use:

.. code-block:: python

    WoE_dict = woe_analysis.WoE_dict            # Dictionary of WoE values
    Variable_types = woe_analysis.Variable_types  # Types of variables analyzed
    Variable_Ranges = woe_analysis.Variable_Ranges  # Ranges or bins used
    IV_excel = woe_analysis.IV_excel            # IV values formatted for Excel
    IV_dict = woe_analysis.IV_dict              # Dictionary of IV values

WoeBinning
----------

The ``WoeBinning`` module transforms your dataset based on the WoE analysis conducted earlier. It replaces the original feature values with their corresponding WoE values.

.. code-block:: python

    from ScoringPy import WoeBinning

    # Assume WoE_dict is obtained from WoeAnalysis
    WoE_dict = woe_analysis.WoE_dict

    # Initialize WoeBinning
    woe_transform = WoeBinning(WoE_dict=WoE_dict, production=False)

    # Transform the data
    X_transformed = woe_transform.transform(X, dummy=False)

Parameters
~~~~~~~~~~

- **WoE_dict**: The dictionary containing WoE values.
- **production** (``bool``, default ``False``): Controls error handling for outliers.
- **dummy** (``bool``, default ``False``): Controls the structure of the output DataFrame.

Explanation
~~~~~~~~~~~

1. **Transformation**: The transformed data will include only the columns specified in ``WoE_dict``.

2. **Selective Transformation**: If you want to transform only specific features, remove unwanted features from ``WoE_dict`` before transformation.

CreditScoring
-------------

The **CreditScoring** module scales scores and probabilities based on your logistic regression model and specific scaling constants. It allows you to generate a scorecard and apply it to your dataset.

Steps
~~~~~

1. **Train a Logistic Regression Model**: Use the transformed data to train your model.

2. **Initialize CreditScoring**: Provide the data, model, WoE dictionary, and production mode.

3. **Apply Scoring**: Generate the scorecard and apply it to your data.

Example
~~~~~~~

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from ScoringPy import CreditScoring

    # Assume X_transformed is your WoE-transformed data
    # Assume y is your target variable

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)

    # Initialize CreditScoring
    scoring = CreditScoring(data=X_train, model=model, WoE_dict=WoE_dict, production=True)

    # Apply scoring to the data
    result = scoring.apply(X_train)

    # Access the scored data and scorecard
    df_scored = result.data
    scorecard = result.scorecard

Parameters
~~~~~~~~~~

- **data**: The dataset to score.

- **model**: The trained logistic regression model.

- **WoE_dict**: The WoE dictionary used for transformations.

- **production** *(bool, default True)*: Controls error handling for outliers during scoring.

  - If `False`: The process will raise an error if it encounters data issues, suitable for development and debugging.
  - If `True`: It will handle outliers gracefully, making it suitable for production environments.


Explanation
~~~~~~~~~~~~

1. **Scorecard Generation**: The `apply_scoring` method generates a scorecard based on the model's coefficients and constants.

2. **Scored Data**: The resulting `df_scored` DataFrame includes the calculated scores for each record.

Metrics
-------

The Metrics module provides tools for credit scoring analysis and visualization. With features like cutoff calculations, trend analysis, score binning, and detailed reporting, this module is ideal for professionals managing credit risk and decision-making processes.

Methods
~~~~~~~

cutoff
~~~~~~

Calculates metrics for a specified approval rate.

.. code-block:: python

    from ScoringPy import Metrics

    # Initialize the Metrics class
    metrics = Metrics(
      Credit_score='Scores',
      Target='Actual',
      Date_column='Date',
      Positive_Target=1,
      Negative_Target=0,
      Data_path='./',  # Adjust the path as needed
      Plot_path='./'   # Adjust the path as needed
    )

    # Count cutoff and display the results
    cutoff_metrics = metrics.cutoff(data, approved_Rate=50, display=True)

**Explanation**:

1. **Initialization**: We initialize `Metrics` with mandatory parameters.
2. **Computing Results for Cutoff**: We call the `cutoff` method, passing the dataframe and `approved_Rate` (default `display=False`).
3. **Calculating and Showing Metrics**: Calculate and display cutoff metrics across approval rates.

cutoff_report
~~~~~~~~~~~~~

Generates a report of cutoff metrics across various approval rates.

.. code-block:: python

    from ScoringPy import Metrics

    # Initialize the Metrics class
    metrics = Metrics(
      Credit_score='Scores',
      Target='Actual',
      Date_column='Date',
      Positive_Target=1,
      Negative_Target=0,
      Data_path='./',  # Adjust the path as needed
      Plot_path='./'   # Adjust the path as needed
    )

    # Generate the cutoff report and display
    cutoff_report = metrics.cutoff_report(data, step=10, save=False)

**Explanation**:

1. **Generating the Cutoff Report**: We call the `cutoff_report` method to calculate metrics like approval rate, default rate, TPR, and FPR across different thresholds. It provides a DataFrame and visual plots for analysis.
2. **Visualizing Metrics**: The `plot` method visualizes key metrics for easier interpretation and decision-making.

score_binning
~~~~~~~~~~~~~

Bins credit scores and computes statistics for each bin.

.. code-block:: python

    from ScoringPy import Metrics

    # Initialize the Metrics class
    metrics = Metrics(
      Credit_score='Scores',
      Target='Actual',
      Date_column='Date',
      Positive_Target=1,
      Negative_Target=0,
      Data_path='./',  # Adjust the path as needed
      Plot_path='./'   # Adjust the path as needed
    )

    # Perform score binning and display
    binning_result = metrics.score_binning(data, bins=10, binning_type=1, save=False)

**Explanation**:

1. **Performing Score Binning**: The `score_binning` method bins the credit scores into groups and calculates summary statistics for each bin.
2. **Summary Statistics**: The method calculates:
   - Number of samples in each bin.
   - Number of bad (negative target) and good (positive target) samples.
   - Percentage of bad/good samples in each bin.
3. **Visualizing Binned Metrics**: The method generates a line plot showing the bad rate across score bins, aiding in evaluating score distribution and risk segmentation.

approval_rate_trend
~~~~~~~~~~~~~~~~~~~

Tracks approval rates over time.

.. code-block:: python

    from ScoringPy import Metrics

    # Initialize the Metrics class
    metrics = Metrics(
      Credit_score='Scores',
      Target='Actual',
      Date_column='Date',
      Positive_Target=1,
      Negative_Target=0,
      Data_path='./',  # Adjust the path as needed
      Plot_path='./'   # Adjust the path as needed
    )

    # Analyze approval rate trends over time (weekly period)
    approval_rate_trend = metrics.approval_rate_trend(data, period='W', score_cutoff=500, save=False)

**Explanation**:

1. **Calculating Approval Rate Trends**: The `approval_rate_trend` method calculates approval rate trends over time and displays summary statistics for certain time periods.
2. **Visualizing Approval Trends**: The method generates a line plot showing the approval rate over time. This helps track performance trends and adjust policies or strategies.

risk_trend_analysis
~~~~~~~~~~~~~~~~~~~

Analyzes and visualizes risk trends over time.

.. code-block:: python

    from ScoringPy import Metrics

    # Initialize the Metrics class
    metrics = Metrics(
      Credit_score='Scores',
      Target='Actual',
      Date_column='Date',
      Positive_Target=1,
      Negative_Target=0,
      Data_path='./',  # Adjust the path as needed
      Plot_path='./'   # Adjust the path as needed
    )

    # Perform risk trend analysis
    risk_trend = metrics.risk_trend_analysis(data, period='W', score_cutoff=500, save=False)

**Explanation**:

1. **Risk Trend Analysis**: The `risk_trend_analysis` method calculates and visualizes risk (negative target rate) trends over time and displays summary statistics for certain periods.
2. **Visualizing Risk Trends**: The method generates a line plot for:
   - Total risk.
   - Risk for applications above the cutoff.
   - Risk for applications below the cutoff.

   This helps monitor trends over time and assess the effectiveness of the cutoff strategy.


Performance Testing and Monitoring
-----------------------------------

By reusing the preprocessing pipeline and WoE transformations, you can ensure consistency in data preparation. This allows for accurate performance comparisons across different data populations, facilitating performance testing and monitoring over time.
