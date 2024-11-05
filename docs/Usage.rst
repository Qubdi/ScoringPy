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

.. code-block:: python

    # Generate a plot and display the report
    woe_analysis.discrete(column="MaritalStatus", df=X_train, target=y_train, safety=True, threshold=300).plot(rotation=0).report()

    # Save the report
    woe_analysis.discrete(column="MaritalStatus", df=X_train, target=y_train, safety=True, threshold=300).report(save=True, type=1)

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
- **production** (``bool``, default ``True``): Controls error handling for outliers during scoring.
