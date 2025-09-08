from sklearn.pipeline import Pipeline
from pathlib import Path as pth
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import math
import os
from typing import  final
from sklearn.preprocessing import KBinsDiscretizer

logging.basicConfig(format="%(message)s", level=logging.WARNING)


class Processing:
    def __init__(self, flow=True):
        """
        Initializes the Processing class with an optional flow parameter.

        Args:
            flow (bool, optional): If True, the default behavior is not to store data in context.
                                   Default is False.
        """
        self.steps = []  # List to store functions, their arguments, and the flow setting
        self.context = {}  # Dictionary to store intermediate data between steps
        self.default_flow = flow  # Set the default flow behavior for the entire pipeline

    def add_step(self, func, *args, flow=None, **kwargs):
        """
        Adds a function step to the pipeline.

        Args:
            func (callable): A function to add to the pipeline.
            *args: Positional arguments to pass to the function.
            flow (bool, optional): If True, the function's result will not be stored in the context
                                   but passed to the next step.
            **kwargs: Keyword arguments to pass to the function.
        """
        # If flow is not specified, use the default flow behavior of the pipeline
        flow = flow if flow is not None else self.default_flow
        self.steps.append((func, args, kwargs, flow))

    def update_paths(self, arg_index: int, new_value, step_index: list = None):
        """
        Updates an argument in specific or in all steps of the stored pipeline steps.

        Args:
            arg_index (int): Index of the argument to update (for positional args).
            new_value: New value to set for the argument.
            step_index (list, optional): List of specific step indices to update. If None, update all steps.
        """
        updated_steps = []
        unapplied_steps = []

        for idx, step in enumerate(self.steps):
            func, args, kwargs, flow = step
            args_list = list(args)

            if step_index is None or idx in step_index:
                if arg_index < len(args_list):
                    args_list[arg_index] = new_value
                else:
                    unapplied_steps.append(idx)

            args = tuple(args_list)
            updated_steps.append((func, args, kwargs, flow))

        step_label = "step" if len(unapplied_steps) == 1 else "steps"

        steps_str = ", ".join(map(str, unapplied_steps))

        logging.warning(
            f"\033[93m⚠️ WARNING: Position {arg_index} is out of range for available {step_label}: {steps_str}. Unable to update.\033[0m"
        )
        self.steps = updated_steps


    def run(self, initial_data=None):
        """
        Executes all steps in the pipeline in the order they were added.
        Stores results in context based on the function name unless flow is True.

        Args:
            initial_data: Optional initial data to pass to the first step.

        Returns:
            The result of the last function executed.
        """
        result = initial_data

        for func, args, kwargs, flow in self.steps:
            # Determine whether to pass the current result to the function
            if flow:
                # If flow is True, pass the current result to the function
                result = func(result, *args, **kwargs) if result is not None else func(*args, **kwargs)
            else:
                # If flow is False, call the function without the current result
                if result is not None:
                    result = func(result, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Store the result in the context
                self._update_context(func, result)
                result = None  # Reset result to prevent passing it to the next step

        return result

    def _update_context(self, func, result):
        """
        Updates the pipeline's context with the result of a function.

        Args:
            func (callable): The function whose result is being stored.
            result: The result returned by the function.
        """
        if isinstance(result, pd.DataFrame):
            # Use the function name as the key
            self.context[func.__name__] = result

    def clear(self):
        """
        Clears the pipeline's context and steps to free up memory.
        """
        self.steps.clear()
        self.context.clear()
        print("Pipeline data cleared to free up memory.")



class WoeAnalysis:
    """
    A class that performs Weight of Evidence (WoE) and Information Value (IV) analysis
    on discrete and continuous variables in a DataFrame.

    Attributes
    ----------
    WoE_dict : dict
        A dictionary to store the Weight of Evidence (WoE) values for different variables.
    IV_dict : dict
        A dictionary to store the Information Value (IV) for different variables.
    IV_excel : pandas.DataFrame
        A DataFrame to store detailed information about the variables, including WoE, IV, and other metrics.

    Methods
    -------
    plot():
        Calls the stored plot function with the given arguments.

    __safety_check(df, column, threshold=300):
        Checks if a column has a number of unique values exceeding the specified threshold.

    __discrete_dummies(df, column):
        Creates dummy variables for the specified categorical column.

    __woe(df, column_name, target_df, type=None):
        Calculates WoE, IV, and other metrics for a given column.

    __plot_woe(woe_df, rotation=0):
        Plots the WoE for a given DataFrame of WoE values.

    discrete(column, df, target, safety=True, threshold=300):
        Performs WoE and IV analysis for discrete variables.

    continuous(column, bins, df, target):
        Performs WoE and IV analysis for continuous variables by binning the data.
        """

    def __init__(self, save=False, path=None, file_format=".xlsx", type=1):
        self.WoE_dict = {}
        self.IV_dict = {}
        self.IV_excel = pd.DataFrame(columns=['Partitions', 'Total', 'Total Perc', 'Good', 'Good Rate', 'Bad', 'Bad Rate',
                                              'Good Dist', 'Bad Dist', 'Woe', 'Good Rate Difference', 'Woe Difference',
                                              'IV', 'PIV','Validation', 'Variable'])
        self.Variable_types = {}
        self.Variable_Ranges = {}

        self.save_path = path
        self.file_format = file_format
        self.type = type
        self.save = save

    def __safety_check(self, df, column, threshold=300):
        """
        Checks if the specified column in the DataFrame has a number of unique values
        exceeding the specified threshold and raises an error if it does.

        Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The name of the column to check.
        threshold (int): The threshold for the number of unique values in the column.
                         If the number of unique values is greater than or equal to this
                         threshold, an error is raised. Default is 5.

        Raises:
        ValueError: If the specified column has unique values greater than or equal to the threshold.
        """

        # validation for dataframe columns
        if column not in df.columns:
            raise KeyError(f"Column '{column}' does not exist in the DataFrame.")

        # validation for threshold
        if len(df[column].value_counts()) >= threshold:
            raise ValueError(
                f"Column '{column}' has {len(df[column].value_counts())} unique values, which exceeds the limit of {threshold}."
                f"If you want to keep tracking the data set safety parameter to False or change threshold to higher value")

    def __discrete_dummies(self,df, column):
        """
        This function creates new columns for each unique value in the specified column,
        and sets the values as True/False.

        Args:
            df (pandas.DataFrame): DataFrame that we are working on.
            column (str): The specific column in the DataFrame to process.

        Returns:
            pandas.DataFrame: DataFrame with the new dummy columns.
        """
        # validation for dataframe columns
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")


        # creating dummy variables for the specified column
        df_dummies = pd.get_dummies(df[column], prefix=column, prefix_sep=':')

        # concatenating dataframe with dummies with original dataframe
        df = pd.concat([df, df_dummies], axis=1)

        return df


    def __woe(self, df, column_name, target_df, type = None):
        """
        counting Woe, PIV, IV and other necessary values for discrete variables

        Args:
            df (dataframe): dataframe which we are working on
            column_name (str): name of variable
            target_df (dataframe): target dataframe
            type (str, optional): Type of variable (discrete or continuous). Defaults to None
        """

        length = df.shape[0]
        # concatenating feature dataframe on target dataframe vertically (rows)
        df = pd.concat([df[column_name], target_df], axis=1)

        # Perform two separate groupby operations on the DataFrame 'df':
        # 1. Group by the values in the first column and calculate the count of values in the second column for each group.
        # 2. Group by the values in the first column and calculate the mean of values in the second column for each group.
        # Concatenate the results of these groupby operations side by side along the columns axis.
        df = pd.concat([
            df.groupby(df.columns.values[0], as_index=False, observed=True)[df.columns.values[1]].count(),
            df.groupby(df.columns.values[0], as_index=False, observed=True)[df.columns.values[1]].mean()
        ], axis=1)

        # select all rows (':') and columns at positions 0, 1, and 3 (0-indexed) and
        # assign the resulting subset DataFrame back to 'df'.
        df = df.iloc[:, [0, 1, 3]]

        # replacing the name of the first column with the value from the first column name.
        # renaming the second column as 'Total'.
        # renaming the third column as 'Good Rate'.
        df.columns = [column_name, 'Total', 'Good Rate']

        # defining and counting new columns for dataframe
        df['Total Perc'] = (df['Total'] / df['Total'].sum()) * 100  # partition percentage share in relation to the total (partitions are for example: E2,E3,A1 etc..etc)
        df['Good'] = df['Good Rate'] * df['Total']  # number of good clients for each partition
        df['Bad'] = (1 - df['Good Rate']) * df['Total']  # number of bad clients for each partition
        df['Good Rate'] = df['Good Rate'] * 100  # percentage of good clients in each partition
        df['Bad Rate'] = (100 - df['Good Rate'])  # percentage of bad clients in each partition
        df['Good Dist'] = (df['Good'] / df['Good'].sum())*100   # percentage of good customers that got into specific partition out of all good customers
        df['Bad Dist'] = (df['Bad'] / df['Bad'].sum())*100   # percentage of bad customers that got into specific partition out of all good customers
        df['Woe'] = np.log(df['Good Dist'] / df['Bad Dist'])  # weight of evidence
        if type == "discrete":
            df = df.sort_values('Woe').reset_index(drop=True)
        df['Good Rate Difference'] = df['Good Rate'].diff().abs()   # difference between every next one Good Rate
        df['Woe Difference'] = df['Woe'].diff().abs()    # difference between every next one Eight of Evidence
        df['PIV'] = ((df['Good Dist'] - df['Bad Dist'])/100) * df['Woe']  # Partition Information Value
        df['IV'] = df['PIV'].sum()   # Variable Information Value
        df['Validation'] = df['Total'].sum() == length   # ensures that None values are handled properly



        # selecting relevant columns to return
        df = df[[column_name, 'Total', 'Total Perc', 'Good', 'Good Rate', 'Bad', 'Bad Rate',
                 'Good Dist', 'Bad Dist', 'Woe', 'Good Rate Difference', 'Woe Difference', 'IV', 'PIV', 'Validation']]
        return df


    def _plot_woe(self, woe_df, rotation=0):
        """
        Plotting by Woe, Woe on y-axis and subcategories (categories of variables) on x-axis.
        A bar chart is also added with the y values taken from the 'Total' column.

        Args:
            woe_df (DataFrame): DataFrame containing Woe and Total values.
            rotation (int): Rotation angle for x-axis labels, 0 by default.
        """

        # select rows where the first column's value is not equal to the string 'NaN' and 'nan'
        woe_df = woe_df[(woe_df[woe_df.columns[0]] != 'NaN') &
                        (woe_df[woe_df.columns[0]] != 'nan')]

        # extract values from the first column of DataFrame 'woe_df', convert them to strings, and make them np.array
        x = np.array(woe_df.iloc[:, 0].apply(str))

        # variables are used for y-axis plotting
        y_woe = woe_df['Woe']
        y_obs = woe_df['Total']

        # setting style and creating a figure with dual axes
        sns.set_style("darkgrid", {"grid.color": "0.8", "grid.linestyle": ":"})
        fig, ax2 = plt.subplots(figsize=(18, 6))

        # plotting the bar chart on the first y-axis
        ax2.bar(x, y_obs, color='steelblue', alpha=0.85, label='Observation Count')
        ax2.set_ylabel('Observation Count')
        ax2.tick_params(axis='x', rotation=rotation)

        # creating a second y-axis for the Woe line plot
        ax1 = ax2.twinx()
        ax1.plot(x, y_woe, marker='o', linestyle='--', color='k', label='Woe')
        ax1.set_xlabel(woe_df.columns[0])
        ax1.set_ylabel('Weight of Evidence')
        ax1.set_title(f'{woe_df.columns[0]}')
        ax1.grid(False)
        # plt.show()
        return self


    def _save_file(self, path=None, name = None ,format=None,type=1, column=None,df1=None, df2=None):
        if type not in [1,2]:
            raise ValueError("type must be 1 ot 2")

        if format not in [".xlsx",".txt",".csv",".pkl"]:
            raise ValueError('type must be .xlsx,.txt,.csv,.pkl ')


        last_element = pth(path).name.split(".")

        if len(last_element) not in (1,2):
            raise ValueError('unsupported value')


        file_name = name or last_element[0] if len(last_element) == 2 else column
        file_format = format or last_element[1] if len(last_element) == 2 else ".xlsx"
        file_path = str(pth(path).parent) if len(last_element) == 2 else path
        full_file_path = os.path.join(file_path, file_name + file_format)

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # Select the DataFrame based on the type
        df_to_save = df1 if type == 1 else df2
        if file_format == '.xlsx':
            df_to_save.to_excel(full_file_path, index=False)
        elif file_format == '.txt':
            df_to_save.to_csv(full_file_path, index=False)
        elif file_format == '.csv':
            df_to_save.to_csv(full_file_path, index=False, encoding="UTF-8")
        elif file_format == '.pkl':
            df_to_save.to_pickle(full_file_path)

        return df1


    def discrete(self, column, df, target, safety=True, threshold=300):
        """
        Determining discrete features' distributions

        Args:
            column (str): name of variable
            df (dataframe): training data
            target (dataframe): target data
            safety (bool, optional): determines unique values for column
            threshold (int, optional): threshold for number of unique values in column
        """
        # Copy of original dataframe
        df_temp = df.copy()

        # Checking if safety is on and executing safety checker function
        if safety:
            self.__safety_check(df=df_temp, column=column, threshold=threshold)

        # Converting categorical variables to dummy variables for the specified column
        df_temp = self.__discrete_dummies(df_temp, column=column)

        # Calculating WOE (Weight of Evidence) for the binned feature
        df_temp = self.__woe(df=df_temp, column_name=column, target_df=target, type="discrete")

        self.WoE_dict = {k: v for k, v in self.WoE_dict.items() if f"{column}" != k}

        # Saving the Woe values in a dictionary for each bin of the feature
        for i, row in df_temp.iterrows():
            self.WoE_dict[f'{column}:' + str(row[column])] = row['Woe']

        self.IV_dict = {k: v for k, v in self.IV_dict.items() if f"{column}" != k}

        # Calculating and storing the Information Value (IV) of the feature
        self.IV_dict[column] = df_temp['IV'].values[0]

        # Creating a copy of df_temp to modify and store in IV_excel
        df_temp2 = df_temp.copy()

        # Adding the feature name to the "Variable" column
        df_temp2['Variable'] = column

        # Renaming the original feature column to "Partitions"
        df_temp2 = df_temp2.rename(columns={column: "Partitions"})

        # Dropping rows in IV_excel where "Variable" equals column
        self.IV_excel = self.IV_excel[self.IV_excel['Variable'] != column]

        # Concatenating the modified DataFrame to IV_excel
        if self.IV_excel is not None and not self.IV_excel.empty:
            self.IV_excel = pd.concat([self.IV_excel, df_temp2], axis=0)
        else:
            self.IV_excel = df_temp2


        self.Variable_types[column] = 'discrete'



        class DiscretePlotter:
            def __init__(self, parent, df_temp):
                self._parent = parent
                self._df_temp = df_temp

            def plot_data(self, rotation=0):
                """Plot the WoE values for the discrete variable."""
                self._parent._plot_woe(self._df_temp, rotation=rotation)
                return self

            def report(self,save=self.save, path=self.save_path, name=None, file_format=self.file_format, type=self.type):
                """Return the DataFrame when called."""
                if save:
                    self._parent._save_file(path=path, name=name, format=file_format, type=type, column=column, df1=self._df_temp, df2=df_temp2)

                class _CheckChain(pd.DataFrame):
                    def __getattr__(self, item):
                        if item == "plot_data":
                            raise AttributeError("You cannot chain `.plot_data()` after `.report()`. "
                                                 "Call `.plot_data()` before `.report()`.")
                        return super().__getattribute__(item)

                return _CheckChain(self._df_temp)
                # return self._df_temp

        # Return only the DiscretePlotter object
        return DiscretePlotter(self, df_temp)



    def continuous(self,column, bins, df, target):
        """
        Determining continous features' distributions

        Args:
            column (str) : name of variable
            bins (tuple) : ranges for continuous features
            df (dataframe) : training data
            target (dataframe) : target data
            rotation_of_x_axis_labels (int) : rotation of labels on x, 0 by default
        """

        df_temp = df.copy()
        if type(bins) is list:
            bins = pd.IntervalIndex.from_tuples(bins)

        # creating a new factorized column based on binning the specified feature
        df_temp[f'{column}_factor'] = pd.cut(df_temp[column], bins)

        # calculating WOE (Weight of Evidence) for the binned feature
        df_temp = self.__woe(df=df_temp, column_name=f'{column}_factor', target_df=target, type="continuous")

        self.WoE_dict = {k: v for k, v in self.WoE_dict.items() if f"{column}" != k}

        for i, row in df_temp.iterrows():
            self.WoE_dict[f'{column}:' + str(row[f'{column}_factor'])] = row['Woe']

        self.IV_dict = {k: v for k, v in self.IV_dict.items() if f"{column}" != k}

        # calculating and storing the Information Value (IV) of the feature
        self.IV_dict[column] = df_temp['IV'].values[0]

        # creating a copy of df_temp to modify and store in IV_excel
        df_temp2 = df_temp.copy()

        # adding the feature name to the "Variable" column
        df_temp2['Variable'] = column

        # renaming the original feature column to "Partitions"
        df_temp2 = df_temp2.rename(columns={f'{column}_factor': "Partitions"})

        # dropping rows in IV_excel where "Variable" equals column
        self.IV_excel = self.IV_excel[self.IV_excel['Variable'] != f'{column}']

        # concatenating the modified DataFrame to IV_excel
        if self.IV_excel is not None and not self.IV_excel.empty:
            self.IV_excel = pd.concat([self.IV_excel, df_temp2], axis=0)
        else:
            self.IV_excel = df_temp2


        self.Variable_types[column] = 'continuous'
        self.Variable_Ranges[column] = bins

        # plotting the distribution of the binned feature based on WOE
        # Define PlottingDataFrame as a subclass of pd.DataFrame
        class DiscretePlotter:
            def __init__(self, parent, df_temp):
                self._parent = parent
                self._df_temp = df_temp

            def plot_data(self, rotation=0):
                """Plot the WoE values for the discrete variable."""
                self._parent._plot_woe(self._df_temp, rotation=rotation)
                return self

            def report(self,save=self.save, path=self.save_path, name=None, file_format=self.file_format, type=self.type):
                """Return the DataFrame when called."""
                if save:
                    self._parent._save_file(path=path, name=name, format=file_format, type=type, column=column, df1=self._df_temp, df2=df_temp2)

                class _CheckChain(pd.DataFrame):

                    def __getattr__(self, item):
                        if item == "plot_data":
                            raise AttributeError("You cannot chain `.plot_data()` after `.report()`. "
                                                 "Call `.plot_data()` before `.report()`.")
                        return super().__getattribute__(item)

                return _CheckChain(self._df_temp)

        # Return only the DiscretePlotter object
        return DiscretePlotter(self, df_temp)


    def auto_binning(self, data, column, n_bins=None, target=None, strategy_option=None, extend_min_bin=False):
        """
        Automatically bins continuous data using different strategies and performs WoE analysis.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the data.
        - column (str): The name of the column to bin.
        - n_bins (int, optional): Number of bins to create (if not provided, a range of bins is tried).
        - target (pd.Series, optional): Target variable for WoE analysis.
        - strategy_option (str, optional): "quantile" for quantile binning, or choose from "uniform", "quantile".
        - extend_min_bin (bool|float, optional): If numeric (e.g., 10), AFTER selecting the best bins,
          subtract this value from the first bin's lower bound. If False, no change.

        Returns:
        - best_result (list[tuple]): Best bin intervals as [(lower_bound, upper_bound), ...]
        """

        strategies = ["uniform", "quantile"]
        all_n_bins = range(2, 30)
        best_IV = 0.0

        if strategy_option:
            strategies = [strategy_option]

        if n_bins:
            all_n_bins = [n_bins]

        all_IV = {}  # IV(float) -> [IntervalIndex bins, list-of-tuples bins, strategy, n_bins]

        for strategy in strategies:
            for n_bins in all_n_bins:
                kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                kb.fit_transform(data[[column]])
                bin_edges = kb.bin_edges_[0]

                bins_interval = pd.IntervalIndex.from_breaks(bin_edges)
                bins_as_tuples = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]

                analysis_result = self.continuous(column=column, bins=bins_interval, df=data, target=target)
                information_value = float(analysis_result.report()["IV"][0])

                if 0 <= information_value <= 1.5:
                    all_IV[information_value] = [bins_interval, bins_as_tuples, strategy, n_bins]
                    if best_IV < information_value:
                        best_IV = information_value
                else:
                    break

        try:
            # Retrieve best bins WITHOUT any extension (so selection is unbiased)
            best_bins_interval, best_bins_tuples, _, _ = all_IV[best_IV]

            # Convert to list of tuples for downstream use
            best_result = [(float(iv.left), float(iv.right)) for iv in best_bins_interval]

            # Apply extension ONLY NOW, after best bins are selected
            if extend_min_bin and isinstance(extend_min_bin, (int, float)):
                new_lower = best_result[0][0] - float(extend_min_bin)
                # Optional safety: ensure the extended lower doesn't exceed the first upper
                if new_lower < best_result[0][1]:
                    best_result[0] = (new_lower, best_result[0][1])
                # else: if user passed a huge value making lower >= upper, keep as-is silently
                # or you could raise/log a warning if you prefer

            # Run final WoE analysis and plot with the (possibly) extended bins
            analysis_result = self.continuous(column=column, bins=best_result, df=data, target=target)
            analysis_result.plot_data()

            return best_result

        except Exception as e:
            print("Could not auto-binning.", e)
            return None

class WoeBinning:
    """
    A class that applies Weight of Evidence (WoE) binning to features in a dataset.

    This class uses a provided dictionary of WoE values to transform the input data by
    applying conditions (such as ranges or distinct values) to specified features.
    The transformed features are then returned as a new DataFrame.

    Methods:
    --------
    fit(X, y=None):
        Fits the WoeBinning model. This method is included for compatibility with
        scikit-learn's fit/transform pattern. It returns the instance itself.

    transform(X, dummy=False):
        Transforms the input data based on the conditions and WoE values specified in the WoE_dict.
        If dummy is True, it creates dummy variables without applying WoE values.
        If dummy is False (default), it applies the WoE values and aggregates the features
        by their common prefix.

    Attributes:
    -----------
    WoE_dict (dict):
        A dictionary that maps features (with optional conditions) to their corresponding
        WoE values, used for transforming the input data.
    """
    def __init__(self, WoE_dict, Production = False):
        # WoE_dict is expected to be a dictionary where keys are features (and possibly conditions)
        # and values are the Weight of Evidence (WoE) values to be applied.
        self.WoE_dict = WoE_dict
        self.Production = Production

    def fit(self, X, y=None):
        # maintains compatibility with scikit-learn's fit/transform pattern.
        # returns the instance itself.
        return self

    def transform(self, X, dummy=False):
        """
       Transform the input DataFrame X using the provided WoE_dict.

       Args:
           X (pd.DataFrame): The input DataFrame containing the features to be transformed.
           dummy (bool): If True, the method will create dummy variables without applying WoE values.
                         If False (default), WoE values will be applied.

       Returns:
           pd.DataFrame: A DataFrame with the transformed features.

       The method processes the DataFrame by applying the conditions specified in WoE_dict to the
       relevant features. If dummy is False, it applies the WoE values to the transformed features
       and aggregates them based on their common prefix.
       """


        # filtering input DataFrame X to include only the relevant columns based on WoE_dict.
        X = X[list(pd.DataFrame({"name": [i.split(":")[0] for i in self.WoE_dict]})["name"].unique())]


        # initializing new DataFrame X_new to store the transformation results.
        X_new = pd.DataFrame(index=X.index)

        # initializing DataFrame to track which rows match the conditions for each feature.
        matched_rows = pd.DataFrame(False, index=X.index, columns=X.columns)

        # Iterate over each feature in WoE_dict
        for feature, woe_value in self.WoE_dict.items():
            # Check if the feature includes a condition (range or distinct value).
            if ':' in feature:
                category, condition = feature.split(':')

                # check if the condition represents a range (e.g., "(value1,value2]" or "[value1,value2)").
                if '(' in condition or '[' in condition:
                    bot, top = condition.split(",")  # splitting the range into bottom (bot) and top (top) values.
                    bot = float(bot[1:])  # removing the leading '(' or '[' and convert to float.
                    top = float(top[:-1]) if top[:-1] != 'inf' else np.inf  # handling 'inf' for open-ended ranges.

                    # creating a mask for rows that fall within the specified range.
                    if top == np.inf:
                        mask = (X[category] > bot)
                    else:
                        mask = (X[category] > bot) & (X[category] <= top)
                else:  # handling distinct categorical values
                    mask = (X[category] == condition)

                # if no rows match the condition, raise an error
                if not mask.any() and not self.Production:
                    unmatched_value = X[category][~matched_rows[category]].iloc[0]
                    raise ValueError(
                        f"Error: No rows match the condition for feature '{feature}' with condition '{condition}'. Unmatched value in '{category}': {unmatched_value}")

                # Initializing the feature column in X_new with NaN and assign 1 to matching rows.
                X_new[feature] = 0
                X_new.loc[mask, feature] = 1

                # updating the matched_rows DataFrame for the current category.
                matched_rows.loc[mask, category] = True

        # checking if all rows have been matched for each feature
        for col in X.columns:
            unmatched_mask = ~matched_rows[col]
            if unmatched_mask.any() and not self.Production:
                # if there are unmatched rows, raise an error with details about the first unmatched value.
                unmatched_index = X.index[unmatched_mask].tolist()[0]
                unmatched_value = X.loc[unmatched_index, col]
                raise ValueError(
                    f"Error: Value '{unmatched_value}' in column '{col}' at index '{unmatched_index}' is outside the defined WoE_dict ranges.")
            if unmatched_mask.any() and self.Production:
                X = X[~unmatched_mask]

        if not dummy:
            # if dummy is False, apply the WoE values to the transformed DataFrame.
            for feature in X_new.columns:
                X_new[feature] *= self.WoE_dict[feature]

            # aggregating features based on their common prefix and sum them
            final_columns = list(
                pd.DataFrame({"name": [i.split(":")[0] for i in self.WoE_dict]}).drop_duplicates()["name"])
            for col in final_columns:
                # summing columns that start with the same prefix.
                mask = [x for x in X_new.columns if x.startswith(col)]
                X_new[col] = X_new[mask].sum(axis=1)

            # retain only the final columns in the transformed DataFrame.
            X_new = X_new[final_columns]

        return X_new



class CreditScoring:
    def __init__(self, data, WoE_dict, model, production):
        self.data = data
        self.WoE_dict = WoE_dict
        self.WoeBinning = WoeBinning
        self.model = model
        self.production = production


        # Initialize WoE transformation object with WoE_dict and production mode
        self.woe_transform = self.WoeBinning(WoE_dict=self.WoE_dict, Production=self.production)

        # Placeholder for the scorecard DataFrame (created later)
        self.scorecard = None

        # Create a pipeline to handle WoE transformation followed by logistic regression
        self.pipeline = Pipeline(steps=[('woe', self.woe_transform), ('logistic regression', self.model)])

        # Define constants for score calculations
        self.PDO = 50
        self.target_score = 800
        self.target_odds = 2
        self.factor = self.PDO / math.log(2)
        self.offset = self.target_score - self.factor * math.log(self.target_odds)

    def transform_data(self):
        # Extract the relevant columns based on the WoE dictionary
        self.X = self.data[list(pd.DataFrame({"name": [i.split(":")[0] for i in self.WoE_dict]}).name.unique())]

        # Perform the initial WoE transformation on the data
        X_dummy = self.woe_transform.transform(self.X, dummy=True)

        # Extract feature names from the transformed data
        self.features = X_dummy.columns.values

        # Perform WoE transformation again to prepare for logistic regression
        X_transformed = self.woe_transform.transform(self.X)

        # Get WoE values for each feature based on the WoE dictionary
        self.woe = [self.WoE_dict[x] for x in self.features]

        # Retrieve logistic regression coefficients for each feature
        self.coeffs = dict()
        for i, feature in enumerate(X_transformed.columns):
            self.coeffs[feature] = self.model.coef_[0][i]

        # Store the intercept (alpha) and number of features for score calculations
        self.alpha = self.model.intercept_[0]
        self.X_dummy = X_dummy
        self.n = len(X_transformed.columns)

    def calculate_score(self, feature):
        # Calculate the score for a given feature using logistic regression coefficients and WoE values
        if feature not in self.scorecard['feature'].values:
            return 0  # Return 0 if the feature is missing from the scorecard
        data = self.scorecard[self.scorecard.feature == feature]
        Bi, WoE_i = data['coef'].values[0], data['WoE'].values[0]

        # Return the contribution of the feature to the total score
        return (Bi * WoE_i + self.alpha / self.n) * self.factor + self.offset / self.n

    def logreg_coef(self, feature):
        # Retrieve the logistic regression coefficient for a given feature
        for key, value in self.coeffs.items():
            if feature.startswith(key):
                return value
        return None

    def create_scorecard(self):
        # Create a DataFrame (scorecard) that stores features, WoE values, coefficients, and calculated scores
        self.scorecard = pd.DataFrame(data=list(zip(self.features, self.woe)), columns=['feature', 'WoE'])
        self.scorecard['coef'] = self.scorecard['feature'].apply(self.logreg_coef)
        self.scorecard['score'] = self.scorecard['feature'].apply(self.calculate_score)

        negative_coefs = self.scorecard[self.scorecard['coef'] < 0]

        if not negative_coefs.empty:
            raise ValueError(f"""Negative coefficients found:
            {negative_coefs}
            """)


    def calculate_individual_scores(self):
        # Compute the individual scores by multiplying the feature matrix with the feature scores
        A_val = self.X_dummy.values
        B_val = np.asarray(self.scorecard['score'].values).T
        scores_val = np.matmul(A_val, B_val)
        return scores_val

    def assign_scores(self):
        # Use the pipeline to predict probabilities (positive and negative classes)
        Probability = self.pipeline.predict_proba(self.X)

        # Calculate the individual credit scores based on the scorecard
        scores_val = self.calculate_individual_scores()

        # Assign the scores and probabilities back to the original dataset
        self.data["Scores"] = scores_val
        self.data["Positive Probability"] = Probability[:, 1]
        self.data["Negative Probability"] = Probability[:, 0]
        return self.data

    def apply(self,data):
        self.data = data

        self.transform_data()
        self.create_scorecard()
        self.data = self.assign_scores()

        return self

class _MetricsResult:
    def __init__(self, dataframe, plot_func):
        """
        Initializes the MetricsResult object with a DataFrame and a plotting function.

        Args:
            dataframe (pd.DataFrame): Data to be analyzed and displayed.
            plot_func (function): A function for plotting the data.
        """
        self.dataframe = dataframe
        self._plot_func = plot_func

    def plot(self):
        """
        Calls the plotting function and returns the DataFrame.
        """
        self._plot_func()
        return self.dataframe  # Return DataFrame after plotting

    def __repr__(self):
        """
        Returns string representation of the DataFrame.
        """
        return self.dataframe.__repr__()

    def __getitem__(self, key):
        """
        Enables indexing into the DataFrame.
        """
        return self.dataframe[key]

    def __getattr__(self, attr):
        """
        Allows access to DataFrame attributes directly.
        """
        return getattr(self.dataframe, attr)

class Metrics:
    def __init__(self, Credit_score='Scores', Target='Actual',  Date_column=None, Positive_Target=1, Negative_Target=0,  Data_path=None, Plot_path=None):
        """
        Initializes the Metrics object with default parameters for credit scoring analysis.

        Args:
            Credit_score (str): Column name for credit scores.
            Target (str): Column name for target variable.
            Positive_Target (int): Value representing a positive target (e.g., good loan).
            Negative_Target (int): Value representing a negative target (e.g., bad loan).
            date_column (str): Column name for date values used in trend analysis.
            data_path (str): Path for saving data reports.
            plot_path (str): Path for saving plot images.
        """
        self.Credit_score = Credit_score
        self.Target = Target
        self.Positive_Target = Positive_Target
        self.Negative_Target = Negative_Target
        self.Date_column = Date_column  # New global variable for date column
        self.Data_path = Data_path
        self.Plot_path = Plot_path

    def cutoff(self, data, approved_Rate, display=False):
        """
        Calculates and returns cutoff metrics for a specified approval rate.

        Args:
            data (pd.DataFrame): Input data containing credit scores and target variable.
            approved_Rate (float): Desired approval rate as a percentage.
            display (bool): If True, prints metrics.

        Returns:
            tuple: Cutoff score, good percentage among approved, bad percentage among approved,
            approval rate, bad percentage under cutoff.

        Raises:
            ValueError: If `approved_Rate` is not between 0 and 100.
            KeyError: If `Credit_score` or `Target` column is missing in the data.
        """
        if not (0 <= approved_Rate <= 100):
            raise ValueError("approved_Rate must be between 0 and 100.")

        if self.Credit_score not in data.columns or self.Target not in data.columns:
            raise KeyError(f"Data must contain columns '{self.Credit_score}' and '{self.Target}'.")

        df_sorted = data.sort_values(by=self.Credit_score, ascending=False)
        percent_index = int((approved_Rate / 100) * len(df_sorted))
        cutoff_score = df_sorted.iloc[percent_index][self.Credit_score]

        # Split data based on cutoff score
        data_above_cutoff = data[data[self.Credit_score] >= cutoff_score]
        data_under_cutoff = data[data[self.Credit_score] < cutoff_score]

        approved_above_cutoff = data_above_cutoff.shape[0]
        good_above_cutoff = data_above_cutoff[data_above_cutoff[self.Target] == self.Positive_Target].shape[0]

        approve_rate = round((approved_above_cutoff / data.shape[0]) * 100, 2)
        good_perc_approved = round((good_above_cutoff / approved_above_cutoff) * 100, 2) if approved_above_cutoff else 0.0
        bad_perc_approved = round(100 - good_perc_approved, 2)

        bad_perc_under_cuttof = round(
            data_under_cutoff[data_under_cutoff[self.Target] == self.Negative_Target].shape[0]
            / data_under_cutoff.shape[0] * 100, 2) if data_under_cutoff.shape[0] else 0.0

        if display:
            print(f"""Cutoff Score: {round(cutoff_score, 2)}
Good Percentage Among Approved: {good_perc_approved}%
Bad Percentage Among Approved: {bad_perc_approved}%
Approval Rate: {approve_rate}%""")

        return cutoff_score, good_perc_approved, bad_perc_approved, approve_rate, bad_perc_under_cuttof


    def _Lineplot(self, data, X_value, Y_values, Y_labels, save=False, X_rotation=90, Y_rotation=0, Grid=True, Title=None, figsize=(10, 4), fontsize=12, show=False):
        """
        Creates a line plot for the given data.

        Args:
            data (pd.DataFrame): Data to plot.
            X_value (str): Column name for x-axis values.
            Y_values (list): List of column names for y-axis values.
            Y_labels (list): List of labels for y-axis data.
            save (bool): If True, saves the plot as an image.
            X_rotation (int): Rotation angle for x-axis labels.
            Y_rotation (int): Rotation angle for y-axis labels.
            Grid (bool): Whether to show grid.
            Title (str): Title of the plot.
            figsize (tuple): Figure size of the plot.
            fontsize (int): Font size for labels and title.
            show (bool): Whether to display the plot.

        Raises:
            ValueError: If `X_value` or `Y_values` columns are missing in the data.
            ValueError: If `Title` is missing and save is True.
        """
        if X_value not in data.columns:
            raise ValueError(f"X_value '{X_value}' not found in data.")
        if not all(y in data.columns for y in Y_values):
            raise ValueError(f"Y_values {Y_values} not found in data.")

        sns.set_style("darkgrid" if Grid else "dark")
        plt.figure(figsize=figsize)
        for y_value, label in zip(Y_values, Y_labels):
            sns.lineplot(x=X_value, y=y_value, data=data, label=label, marker='o')
            for x, y in zip(data[X_value], data[y_value]):
                plt.annotate(f'{round(y, 2)}', (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=fontsize * 0.6)

        plt.xlabel(X_value, fontsize=fontsize)  # Use X_value for the x-axis label
        plt.ylabel(", ".join(Y_labels), fontsize=fontsize)  # Concatenate Y_labels for the y-axis label
        if Title:
            plt.title(Title, fontsize=fontsize)
        plt.xticks(rotation=X_rotation, fontsize=fontsize)
        plt.yticks(rotation=Y_rotation, fontsize=fontsize)
        plt.legend(fontsize=fontsize)

        if save:
            if not self.Plot_path:
                raise ValueError("Plot path is not set. Please specify `plot_path` to save the plot.")
            plt.savefig(self.Plot_path + f'{Title}.png', bbox_inches='tight', format='png')

        if show:
            plt.show()


    def cutoff_report(self, data, step=5, title='Cutoff Report', save=False):
        """
        Generates a report showing various cutoff metrics across approval rates.

        Args:
            data (pd.DataFrame): Input data containing credit scores and target variable.
            step (int): Step size for approval rates (default: 5).
            title (str): Title for the saved report (if save=True).
            save (bool): If True, saves the report to an Excel file.

        Returns:
            MetricsResult: Contains the DataFrame with cutoff metrics and a plot function.

        Raises:
            ValueError: If `step` is less than or equal to 0.
            KeyError: If necessary columns are missing in `data`.
            ValueError: If `title` is not provided when `save` is True.
            ValueError: If `save=True` and `data_path` is not set.
        """
        if step <= 0:
            raise ValueError("Step must be greater than 0.")

        if self.Credit_score not in data.columns or self.Target not in data.columns:
            raise KeyError(f"Data must contain columns '{self.Credit_score}' and '{self.Target}'.")

        if save and not self.Data_path:
            raise ValueError("Data path is not set. Please specify `data_path` to save the report.")

        results = []

        for j in range(20, 100, step):
            approve_rate = self.cutoff(
                data=data[[self.Target, self.Credit_score]],
                approved_Rate=j,
                display=False
            )
            results.append({
                'Cutoff Score': approve_rate[0],
                'Good Percentage': approve_rate[1],
                'Bad Percentage': approve_rate[2],
                'Approval Rate': approve_rate[3],
                'Bad Percentage Under Cutoff': approve_rate[4]
            })

        results_df = pd.DataFrame(results)

        if save:
            results_df.to_excel(self.Data_path + f"{title}.xlsx")

        def plot(save=False):
            """
            Plots approval rate against bad rate.

            Args:
                save (bool): If True, saves the plot.

            Raises:
                ValueError: If `save=True` and `plot_path` is not set.
            """
            if save and not self.Plot_path:
                raise ValueError("Plot path is not set. Please specify `plot_path` to save the plot.")

            self._Lineplot(
                data=results_df,
                X_value='Bad Percentage',
                Y_values=['Approval Rate'],
                Y_labels=['Model Approved'],
                Title="Approve Rate Over Bad Rate",
                figsize=(12, 5),
                show=True,
                X_rotation=0,
                save=save
            )

        return _MetricsResult(results_df, plot)


    def score_binning(self, data, bins=20, binning_type=1, title='Scores Binning', save=False):
        """
        Bins the credit scores and generates summary statistics for each bin.

        Args:
            data (pd.DataFrame): Input data with scores and target.
            bins (int): Number of bins (default: 20).
            binning_type (int): 1 for quantile binning, 2 for equal-width binning.
            title (str): Title for the saved report (if save=True).
            save (bool): If True, saves the binning report to an Excel file.

        Returns:
            MetricsResult: Contains the binned DataFrame and a plot function.

        Raises:
            ValueError: If `binning_type` is not 1 or 2.
            KeyError: If `Credit_score` or `Target` column is missing in data.
            ValueError: If `bins` is less than 1.
            ValueError: If `save=True` and `data_path` is not set.
        """
        if self.Credit_score not in data.columns or self.Target not in data.columns:
            raise KeyError(f"Data must contain columns '{self.Credit_score}' and '{self.Target}'.")

        if bins < 1:
            raise ValueError("Number of bins must be at least 1.")

        if binning_type not in [1, 2]:
            raise ValueError("binning_type must be 1 (quantile) or 2 (equal-width).")

        if save and not self.Data_path:
            raise ValueError("Data path is not set. Please specify `data_path` to save the report.")

        Data = data.sort_values(by=self.Credit_score)

        if binning_type == 1:
            Data['Bin'] = pd.qcut(Data[self.Credit_score], q=bins, duplicates='drop')
        else:
            Data['Bin'] = pd.cut(Data[self.Credit_score], bins=bins, duplicates='drop')

        grouped = Data.groupby('Bin').agg(
            Count=('Scores', 'count'),
            Bad=('Actual', lambda x: (x == self.Negative_Target).sum()),
            Good=('Actual', lambda x: (x == self.Positive_Target).sum())
        ).reset_index()

        grouped['Bad Rate'] = (grouped['Bad'] / grouped['Count']) * 100
        grouped['Good Rate'] = (grouped['Good'] / grouped['Count']) * 100

        grouped.insert(0, 'RowNumber', grouped.index + 1)

        if save:
            grouped.to_excel(self.Data_path + f'{title}.xlsx')

        def plot(save=False):
            """
            Plots the bad rate across score bins.

            Args:
                save (bool): If True, saves the plot.

            Raises:
                ValueError: If `save=True` and `plot_path` is not set.
            """
            if save and not self.Plot_path:
                raise ValueError("Plot path is not set. Please specify `plot_path` to save the plot.")

            self._Lineplot(
                data=grouped,
                X_value='RowNumber',
                Y_values=['Bad Rate'],
                Y_labels=['Bad Rate'],
                Title=title,
                figsize=(12, 5),
                show=True,
                X_rotation=0,
                save=save
            )

        return _MetricsResult(grouped, plot)


    def approval_rate_trend(self, data, period='M', score_cutoff=None, save=False, title='Approval Rate Trend'):
        """
        Calculates approval rate trends over time based on a single period and provides a combined summary table.

        Args:
            data (pd.DataFrame): Input data containing a date column and credit score.
            period (str): Custom period string (e.g., '2D' for 2 days, '3W' for 3 weeks, '5M' for 5 months).
            score_cutoff (float or int, optional): Cutoff score to determine approvals. Applications with scores
                                                   above this value are "approved"; those below are "rejected."
            save (bool): If True, saves the combined report to an Excel file.
            title (str): Title for the saved report (if save=True).

        Returns:
            MetricsResult: Contains the DataFrame with approval rates and other metrics over time and a plot function.

        Raises:
            KeyError: If `date_column` or `Credit_score` is missing in data.
            ValueError: If `save=True` and `data_path` is not set.
            ValueError: If `date_column` is not set during initialization or method call.
        """
        if not self.Date_column:
            raise ValueError("date_column is not set. Please specify `date_column` during initialization or use a method with a date column.")

        if self.Date_column not in data.columns or self.Credit_score not in data.columns:
            raise KeyError(f"Data must contain columns '{self.Date_column}' and '{self.Credit_score}'.")

        if save and not self.Data_path:
            raise ValueError("Data path is not set. Please specify `data_path` to save the report.")

        # Ensure the date column is in datetime format
        data[self.Date_column] = pd.to_datetime(data[self.Date_column])

        # Group data by the specified period
        grouped = data.groupby(pd.Grouper(key=self.Date_column, freq=period))

        # Initialize the result DataFrame
        combined_data = []

        for date, group in grouped:
            if group.empty:
                continue

            total_applications = group.shape[0]

            # Determine approvals and rejections based on score cutoff
            if score_cutoff is not None:
                approved_applications = group[group[self.Credit_score] >= score_cutoff].shape[0]
                rejected_applications = group[group[self.Credit_score] < score_cutoff].shape[0]
            else:
                raise ValueError("score_cutoff must be provided to calculate approval rate.")

            approval_rate = (approved_applications / total_applications) * 100

            combined_data.append({
                'Period': date.strftime('%Y-%m-%d'),
                'Period Length': period,  # Add period type
                'Total Applications': total_applications,
                'Approved Applications': approved_applications,
                'Rejected Applications': rejected_applications,
                'Approval Rate (%)': approval_rate
            })

        # Convert to DataFrame
        combined_df = pd.DataFrame(combined_data)

        if save:
            combined_df.to_excel(self.Data_path + f'{title}_{period}_combined.xlsx', index=False)

        def plot(save=False):
            """
            Plots the approval rate trend over time.

            Args:
                save (bool): If True, saves the plot.

            Raises:
                ValueError: If `save=True` and `plot_path` is not set.
            """
            if save and not self.Plot_path:
                raise ValueError("Plot path is not set. Please specify `plot_path` to save the plot.")

            self._Lineplot(
                data=combined_df,
                X_value='Period',
                Y_values=['Approval Rate (%)'],
                Y_labels=['Approval Rate (%)'],
                Title=f'{title} ({period} Period)',
                figsize=(12, 6),
                show=True,
                save=save,
                X_rotation=45
            )

        return _MetricsResult(combined_df, plot)


    def risk_trend_analysis(self, data, period='M', score_cutoff=None, save=False, title='Risk Trend Analysis'):
        """
        Calculates and plots the risk trend (negative target rate) over time for applications above cutoff, below cutoff, and total.

        Args:
            data (pd.DataFrame): Input data containing a date column, credit score, and target variable.
            period (str): Custom period string (e.g., '2D' for 2 days, '3W' for 3 weeks, '5M' for 5 months).
            score_cutoff (float or int, optional): Cutoff score to determine approvals. Applications with scores
                                                   above this value are "approved"; those below are "rejected."
            save (bool): If True, saves the combined report and plot to Excel and image files.
            title (str): Title for the saved report and plot.

        Returns:
            MetricsResult: Contains the DataFrame with risk metrics over time and a plot function.

        Raises:
            KeyError: If `date_column` or `Credit_score` is missing in data.
            ValueError: If `save=True` and `data_path` is not set.
            ValueError: If `date_column` is not set during initialization or method call.
        """
        if not self.Date_column:
            raise ValueError("date_column is not set. Please specify `date_column` during initialization or use a method with a date column.")

        if self.Date_column not in data.columns or self.Credit_score not in data.columns or self.Target not in data.columns:
            raise KeyError(f"Data must contain columns '{self.Date_column}', '{self.Credit_score}', and '{self.Target}'.")

        if save and not self.Data_path:
            raise ValueError("Data path is not set. Please specify `data_path` to save the report.")

        # Ensure the date column is in datetime format
        data[self.Date_column] = pd.to_datetime(data[self.Date_column])

        # Group data by the specified period
        grouped = data.groupby(pd.Grouper(key=self.Date_column, freq=period))

        # Initialize the result DataFrame
        combined_data = []

        for date, group in grouped:
            if group.empty:
                continue

            total_applications = group.shape[0]
            total_negative = group[group[self.Target] == self.Negative_Target].shape[0]
            total_risk = (total_negative / total_applications) * 100 if total_applications else 0

            # Above and below cutoff analysis
            if score_cutoff is not None:
                above_cutoff = group[group[self.Credit_score] >= score_cutoff]
                below_cutoff = group[group[self.Credit_score] < score_cutoff]

                above_negative = above_cutoff[above_cutoff[self.Target] == self.Negative_Target].shape[0]
                below_negative = below_cutoff[below_cutoff[self.Target] == self.Negative_Target].shape[0]

                above_risk = (above_negative / above_cutoff.shape[0]) * 100 if above_cutoff.shape[0] else 0
                below_risk = (below_negative / below_cutoff.shape[0]) * 100 if below_cutoff.shape[0] else 0
            else:
                above_risk, below_risk = None, None

            combined_data.append({
                'Period': date.strftime('%Y-%m-%d'),
                'Total Risk (%)': total_risk,
                'Above Cutoff Risk (%)': above_risk,
                'Below Cutoff Risk (%)': below_risk,
            })

        # Convert to DataFrame
        combined_df = pd.DataFrame(combined_data)

        if save:
            combined_df.to_excel(self.Data_path + f'{title}_{period}_risk_combined.xlsx', index=False)

        def plot(save=False):
            """
            Plots the risk trend over time for above cutoff, below cutoff, and total.

            Args:
                save (bool): If True, saves the plot.

            Raises:
                ValueError: If `save=True` and `plot_path` is not set.
            """
            if save and not self.Plot_path:
                raise ValueError("Plot path is not set. Please specify `plot_path` to save the plot.")

            self._Lineplot(
                data=combined_df,
                X_value='Period',
                Y_values=['Above Cutoff Risk (%)', 'Below Cutoff Risk (%)', 'Total Risk (%)'],
                Y_labels=['Above Cutoff', 'Below Cutoff', 'Total'],
                Title=f'{title} ({period} Period)',
                figsize=(12, 6),
                show=True,
                save=save,
                X_rotation=45
            )

        return _MetricsResult(combined_df, plot)


