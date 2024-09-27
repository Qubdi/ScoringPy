<h1 align="center">ScoringPy</h1>


## Overview
The ScoringPy is a Python library designed for Classical Credit ScoreCard Deployment.

This library is divided into three main parts:
data preprocessing by pipline,
feature and binning selection (automatically and manual),
scorecard deployment and scaling.

with this library data processing is more efficient, error rate is minimal 
because we use several layer of protection to find data anomaly before train model



## Features
- **Data Preprocessing with pipline**: crucial for model validation and testing with new data (every data manipulation is saved and ready to use easyly with new data)
- **Feature selection**: for every feature we have small report which can easily discribe this feature's statistic (based on Woe Analisys).
- **binning**: Classical scoring need to bin continius features, for better inmplementation, with this library we have 2 option manual and 
Automatic (sagestion) binning, aslo we have binning Validation in feature statistic report (it checks if some data is out of bin ranges).
- **Final Data Transformation**: it is second layer of protection for outlaierd data, (if some data is outlier which was not in feature bining code give us allret about it)
- **Scorcard Deploiment**: it is scoring scaling based on model it's coefitinet and constants is 100% customizable (it default values is sppet spot of most scaling cases).
- **Performace Testing**: with  all thids features it is very easy to test scorecard performace for other population (data)
- **Monitoring**: it is very easy to track scorecard/population performace becaose or piplines







## Installation

The ScoringPy library can be installed using either `pip` or `conda`. Below are the instructions for both methods:

To install the ScoringPy via `pip`, use the following command:

```bash
pip install ScoringPy
```

To install the ScoringPy via `conda`, use the following command:

```bash
conda install -c conda-forge ScoringPy
``` 





## Usage


- **`Processing`**: Designed for Data Preprocesing.
- **`WoeAnalysis`**: Designed for Feature and binning Selection.
- **`WoeBinning`**: Designed for data Transformation based on Feature and binning Selection.
- **`CreditScoring`**: Designed for scaling scoring and probabilities based on model and some constant parameters.


### Processing
Type 1:
```python
# Importing necessary modules
from ScoringPy import Processing
import pandas as pd
import dill


# Initializing the processing pipeline using the Processing class from the imported module.
# Setting 'flow=True' means that the output from each function (step) will be passed as input to the next function.
# This allows transformations to be applied sequentially, one step after another, without needing manual intervention.
# The flow won't run immediately on initialization, allowing flexibility to add steps first.
pipeline = Processing(flow=True)

# Function that defines step 1 of the pipeline
def step_1(data):
  data['Age'] = data['Age'].fillna(data['Age'].mean())  
  return data

# Function that defines step 2 of the pipeline
def step_2(data):
  data['Age'] = data['Age'] * 2  
  return data

# Function that defines step 3 of the pipeline
def step_3(data):
  data['Age'] = data['Age'] / 5  
  return data

# Adding step 1 to the processing pipeline.
# The pipeline will now execute step_1 as the first transformation when run.
pipeline.add_step(step_1)

# Adding step 2 to the processing pipeline.
# The pipeline will execute step_2 after step_1 when the pipeline is run.
pipeline.add_step(step_2)

# Adding step 3 to the processing pipeline.
# The pipeline will execute step_3 after step_2 when the pipeline is run.
pipeline.add_step(step_3)

# Saving the configured pipeline (with the steps added) to a file.
# This allows for reusability of the pipeline without reconfiguring it each time.
# The file is saved using the 'dill' library, which can serialize complex Python objects like classes and functions.
with open('Pipeline.pkl', 'wb') as file:
  dill.dump(pipeline, file)

# Running the pipeline on the initial data (the 'df' DataFrame created earlier).
df = pipeline.run(initial_data = df)

# Clearing the pipeline
# The pipeline object is cleared, meaning all the added steps will be removed.
# This can be useful if you want to reset the pipeline and add new steps for different transformations.
pipeline.clear()

# After clearing, the pipeline is now empty and ready for new steps if needed.
```

Reuse Type 1 Pipline:
```python
# Step 2: Loading the pipeline for Filling Data
# This section loads a previously saved pipeline from a file using the 'dill' library.
# The file 'Pipeline.pkl' contains the serialized version of the pipeline that was configured and saved earlier.

with open('Pipeline.pkl', 'rb') as file:
    pipeline = dill.load(file)  # Loading the previously saved pipeline from the file.

# The 'pipeline' object is now restored with all the previously added steps (step_1, step_2, and step_3).
# These steps will be executed in sequence when the pipeline is run with the new input data.

# Executing the pipeline with the DataFrame 'df', which was produced as a result of the first pipeline execution.
# The input DataFrame 'df' contains transformations done by the earlier pipeline execution.
# Here, we pass it through the same pipeline to reapply the same transformations, or to continue processing based on current data.

df = pipeline.run(initial_data=df)  # Running the pipeline on the current DataFrame.

# Clearing the pipeline
# The pipeline object is cleared, meaning all the added steps will be removed.
# This can be useful if you want to reset the pipeline and add new steps for different transformations.
pipeline.clear()

# At this point, the 'pipeline' object is empty, and no steps are available for execution until new steps are added.

```

Type 2:
```python
# Importing necessary modules
from ScoringPy import Processing
import pandas as pd
import dill

# The file path for the first Excel file.
row_path = 'Data/step1.xlsx'

# Initializing the processing pipeline without running the flow immediately.
# Setting flow=False means that the steps in the pipeline will not automatically pass data between them.
# You will control the flow manually and decide when to pass data to subsequent steps.
pipeline = Processing(flow=False)

# Step 1: Load data from an Excel file (provided as 'path').
def step_1(path=None):
  data = pd.read_excel(path)  
  return data

# Step 2: Load a different dataset from another Excel file ('Data/step2.xlsx').
def step_2():
  data = pd.read_excel('Data/step2.xlsx')  
  return data

# Step 3: Concatenate the results from step 1 and step 2.
# It retrieves the data from the context (stored results from previous steps).
def step_3():
  step_1_data = pipeline.context.get('step_1')  # Retrieving the result of step 1 from the pipeline context.
  step_2_data = pipeline.context.get('step_2')  # Retrieving the result of step 2 from the pipeline context.

  # Concatenating the two DataFrames from step 1 and step 2.
  data = pd.concat([step_1_data, step_2_data], ignore_index=True)

  data['Age'] = data['Age'] * 2  
  return data

# Step 4: Further transformation of the data by dividing the 'Age' column by 5.
def step_4(data):
  data['Age'] = data['Age'] / 5 
  return data


# Adding the steps to the pipeline
# Each step is added sequentially, and the corresponding function is passed as an argument.
# If the function requires parameters (e.g., step 1), the parameters are provided when adding the step.

# Step 1: Reading data from the Excel file located at 'row_path'.
pipeline.add_step(step_1, row_path)

# Step 2: Reading data from 'Data/step2.xlsx' file.
pipeline.add_step(step_2)

# Step 3: Concatenating the data from steps 1 and 2.
# flow=True ensures that the output from step 3 will automatically be passed to step 4.
pipeline.add_step(step_3, flow=True)

# Step 4: Applying further transformations to the data, with flow=True.
# Here, flow=True ensures that the output from step 3 will be passed to step 4 in our case saved into variable automatically.
pipeline.add_step(step_4, flow=True)

# Saving the configured pipeline to a file for reuse in the future.
# The 'dill' library is used to serialize the pipeline object.
with open('Pipeline.pkl', 'wb') as file:
  dill.dump(pipeline, file)

# Running the pipeline
# The pipeline is executed, and it flows through all the steps in sequence:
# Step 1 (data loading), Step 2 (additional data loading), Step 3 (data concatenation and transformation),
# and Step 4 (final transformation). The output from each step is automatically passed to the next step when flow=True.
df = pipeline.run()

# Clearing the pipeline
# The pipeline object is cleared, meaning all the added steps will be removed.
# This can be useful if you want to reset the pipeline and add new steps for different transformations.
pipeline.clear()

# After clearing, the pipeline is now empty and ready for new steps if needed.
```

Reuse Type 2 Pipline:
```python
# Step 2: Loading the pipeline for processing data.
# The 'Pipeline.pkl' file contains the saved pipeline object, which was serialized using the 'dill' library.
# This pipeline has multiple steps that were configured earlier for data transformation.
# We load the pipeline to reuse it without needing to redefine or reconfigure it.

with open('Pipeline.pkl', 'rb') as file:
  pipeline = dill.load(file)  # Loading the previously saved pipeline from the file using 'dill'.

# After loading, the pipeline is restored with all the previously added steps.
# These steps can now be executed in sequence as they were originally configured.

# Executing the pipeline.
# The pipeline's 'run()' method is called to execute all the steps in the pipeline.
# Since the pipeline was configured with 'flow=True' for certain steps, data will automatically be passed from one step to the next.
# The steps will process the data according to the logic defined in each function (e.g., data loading, concatenation, transformation).
df = pipeline.run()

# At this point, the 'df' DataFrame will contain the final result after all transformations are applied by the pipeline.

# Clearing the pipeline.
# The 'clear()' method removes all the steps and clears the context of the pipeline.
# This is useful if you want to reset the pipeline to a clean state before re-adding steps or reconfiguring it.
pipeline.clear()

# After clearing, the pipeline is now empty and ready for new steps if needed.

```

Processing Optional Arguments
- **`flow`**: `True` (default)
  - **Type**: Boolean
  - **Description**: Setting flow=False means that the steps in the pipeline will not automatically pass data between them,
    Setting 'flow=True' means that the output from each function (step) will be passed as input to the next function.



### WoeAnalysis

WoeAnalysis's have 2 main method `discrete` and `continuous` each of them have 
ther own method `plot` and `save`,
`plot` method stands for visualize Woe and Iv analysis and `save` stands for to save report if it nesesery

use Cases for `discrete`:
```python
from ScoringPy import WoeAnalysis
import pandas as pd
import numpy as np


# intialising WoeAnalysis class
woe_analysis = WoeAnalysis(save=False,path="Data/", type=2)


# default is: safety=True, threshold=300
# safty is built in feature which refuse to make
# feature if it is not contius and contain more then
# 300 uniqeu values it is alseo cosumaisable by user
woe_analysis.discrete(column="MaritalStatus", df=X_train, target=y_train, safety=True, threshold=300).report()
```

```python
# it will plot Woe and IV and aslo show report
# Report method must be last one in every case 
woe_analysis.discrete(column="MaritalStatus", df=X_train, target=y_train).plot(rotation=0).report()
```

```python
# have several paraeters path=path, name=name, format=file_format, type=type
# it will plot Woe and IV and aslo show report as well as show report
# Report method must be last one in every case 
woe_analysis.discrete(column="MaritalStatus", df=X_train, target=y_train).report(save=True, type=1)
```

use Cases for `continuous`:
```python
bins = pd.IntervalIndex.from_tuples([(-1,0),(0, 0.2), (0.2,0.35), (0.35, 0.45),(0.45, 0.55), (0.55, 0.65),(0.65, np.inf)])
woe_analysis.continuous(column="RefinanceRate", bins= bins,df=X_train, target=y_train).report()
```

```python
# it will plot Woe and IV and aslo show report
# Report method must be last one in every case 
bins = pd.IntervalIndex.from_tuples([(-1,0),(0, 0.2), (0.2,0.35), (0.35, 0.45),(0.45, 0.55), (0.55, 0.65),(0.65, np.inf)])
woe_analysis.continuous(column="RefinanceRate", bins= bins,df=X_train, target=y_train).plot(rotation=90).report()

```

```python
# have several paraeters path=path, name=name, format=file_format, type=type
# it will plot Woe and IV and aslo show report as well as show report
# Report method must be last one in every case 
bins = pd.IntervalIndex.from_tuples([(-1,0),(0, 0.2), (0.2,0.35), (0.35, 0.45),(0.45, 0.55), (0.55, 0.65),(0.65, np.inf)])
woe_analysis.continuous(column="RefinanceRate", bins= bins,df=X_train, target=y_train).report(save=True)
```

also we can take maximum iformation from class, for future usage
```python
WoE_dict = woe_analysis.WoE_dict
Variable_types = woe_analysis.Variable_types
Variable_Ranges = woe_analysis.Variable_Ranges
IV_excel = woe_analysis.IV_excel
IV_dict = woe_analysis.IV_dict
```


### WoeBinning

```python
# it transsform data based on WoE_dict, 
# and retun data with only column which is in WoE_dict
# if you dont want to transform data with whole WoE_dict,
# remove some features from WoE_dict and give WoeBinning 
from ScoringPy import WoeBinning

WoE_dict = woe_analysis.WoE_dict

# Production is paramter which is booleran and
# if it Is False it gives us error if some row's contain outlaier
# and if it is True it didnr rize error just remove specific 
# row from data and continiu tranforamtion
woe_transform = WoeBinning(WoE_dict= WoE_dict, Production=False)

# transform have one optional parameter dummpy 
# if it True it will return data with columns which comes from WoE_dict
# if it is False it just change values and dont change columns
X_transformed = woe_transform.transform(X,dummy=False)
```


### CreditScoring


```python
# it transsform data based on WoE_dict, 
# and retun data with only column which is in WoE_dict
# if you dont want to transform data with whole WoE_dict,
# remove some features from WoE_dict and give WoeBinning 
from sklearn.linear_model import LogisticRegression
from ScoringPy import WoeBinning
from ScoringPy import CreditScoring


WoE_dict = woe_analysis.WoE_dict

# Production is paramter which is booleran and
# if it Is False it gives us error if some row's contain outlaier
# and if it is True it didnr rize error just remove specific 
# row from data and continiu tranforamtion
woe_transform = WoeBinning(WoE_dict= WoE_dict, Production=False)

# transform have one optional parameter dummpy 
# if it True it will return data with columns which comes from WoE_dict
# if it is False it just change values and dont change columns
X_transformed = woe_transform.transform(X,dummy=False)

# creating a Logistic Regression model with specified parameters
model = LogisticRegression(max_iter=1_000, class_weight='balanced', C=0.1)


# Example usage
scoring = CreditScoring(data=df, model=model, WoE_dict=WoE_dict, production=True)

# scaling scores for df
temp_df = scoring.apply(df)

# take data from scaling result 
df = temp_df.data

# take scorecard based on model
scorecard = temp_df.scorecard
```