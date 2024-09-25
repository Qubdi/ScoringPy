from sklearn.pipeline import Pipeline
from pathlib import Path as pth
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os


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

    def __init__(self):
        self.WoE_dict = {}
        self.IV_dict = {}
        self.IV_excel = pd.DataFrame(columns=['Partitions', 'Total', 'Total Perc', 'Good', 'Good Rate', 'Bad', 'Bad Rate',
                                              'Good Dist', 'Bad Dist', 'Woe', 'Good Rate Difference', 'Woe Difference',
                                              'IV', 'PIV','Validation', 'Variable'])
        self.Variable_types = {}
        self.Variable_Ranges = {}

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


    def _save_file(path=None, name = None ,format=None,type=1, column=None,df1=None, df2=None):
        if type not in [1,2]:
            raise ValueError("type must be 1 ot 2")

        if format not in [".xlsx",".txt",".csv",".pkl"]:
            raise ValueError('type must be .xlsx,.txt,.csv,.pkl ')

        last_element = pth(path).name.split(".")

        if len(last_element) not in (1,2):
            raise ValueError('unsupported value')


        file_name = name or last_element[0] if len(last_element) == 2 else column
        file_format = format or last_element[1] if len(last_element) == 2 else ".xlsx"
        file_path = str(path.parent) if len(last_element) == 2 else path
        full_file_path = os.path.join(file_path, file_name + file_format)

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # Select the DataFrame based on the type
        df_to_save = df1 if type == 1 else df2

        if file_format == '.xslx':
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

        self.WoE_dict = {k: v for k, v in self.WoE_dict.items() if f"{column}" not in k}

        # Saving the Woe values in a dictionary for each bin of the feature
        for i, row in df_temp.iterrows():
            self.WoE_dict[f'{column}:' + str(row[column])] = row['Woe']

        self.IV_dict = {k: v for k, v in self.IV_dict.items() if f"{column}" not in k}

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

        # Define PlottingDataFrame as a subclass of pd.DataFrame
        class PlottingDataFrame(pd.DataFrame):
            _metadata = ['_parent']

            @property
            def _constructor(self):
                def _c(*args, **kwargs):
                    return PlottingDataFrame(*args, **kwargs)._set_parent(self._parent)
                return _c

            def _set_parent(self, parent):
                self._parent = parent
                return self

            def plot(self, rotation=0):
                """
                Plots the WOE values using the stored DataFrame and returns the DataFrame.

                Args:
                    rotation (int): Rotation angle for x-axis labels, 0 by default.

                Returns:
                    PlottingDataFrame: Returns the DataFrame object itself after plotting.
                """
                self._parent._plot_woe(self, rotation=rotation)
                return self

            def save(self, save=False, path=None, name=None, file_format=".xlsx", type=1, column=None):
                """
                Saves the DataFrame using the _save_file method of the WoeAnalysis class.

                Args:
                    path (str or Path, optional): Directory path where the file will be saved.
                    name (str, optional): File name. If not provided, the column name will be used.
                    file_format (str, optional): Format of the file. Default is ".xlsx".
                    type (int, optional): Type of DataFrame to save if multiple options exist.
                    column (str, optional): Column name used if the name is not provided.

                Returns:
                    PlottingDataFrame: The DataFrame object itself after saving.
                """
                if save:
                    self._parent._save_file(path=path, name=name, format=file_format, type=type, column=column, df1=self)
                return self

        # Create an instance of PlottingDataFrame and set the parent
        result = PlottingDataFrame(df_temp)._set_parent(self)

        self.Variable_types[column] = 'discrete'

        # Return the custom DataFrame with added plot and save capabilities
        return result










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

        # creating a new factorized column based on binning the specified feature
        df_temp[f'{column}_factor'] = pd.cut(df_temp[column], bins)

        # calculating WOE (Weight of Evidence) for the binned feature
        df_temp = self.__woe(df=df_temp, column_name=f'{column}_factor', target_df=target, type="continuous")

        self.WoE_dict = {k: v for k, v in self.WoE_dict.items() if f"{column}" not in k}

        for i, row in df_temp.iterrows():
            self.WoE_dict[f'{column}:' + str(row[f'{column}_factor'])] = row['Woe']

        self.IV_dict = {k: v for k, v in self.IV_dict.items() if f"{column}" not in k}

        # calculating and storing the Information Value (IV) of the feature
        self.IV_dict[column] = df_temp['IV'].values[0]

        # creating a copy of df_temp to modify and store in IV_excel
        df_temp2 = df_temp.copy()

        # adding the feature name to the "Variable" column
        df_temp2['Variable'] = column

        # renaming the original feature column to "Partitions"
        df_temp2 = df_temp2.rename(columns={f'{column}_factor': "Partitions"})

        # dropping rows in IV_excel where "Variable" equals column
        self.IV_excel = self.IV_excel[self.IV_excel['Variable'] != f'{column}_factor']

        # concatenating the modified DataFrame to IV_excel
        if self.IV_excel is not None and not self.IV_excel.empty:
            self.IV_excel = pd.concat([self.IV_excel, df_temp2], axis=0)
        else:
            self.IV_excel = df_temp2

        # plotting the distribution of the binned feature based on WOE
        class PlottingDataFrame(pd.DataFrame):
            _metadata = ['_parent']

            @property
            def _constructor(self):
                def _c(*args, **kwargs):
                    return PlottingDataFrame(*args, **kwargs)._set_parent(self._parent)
                return _c

            def _set_parent(self, parent):
                self._parent = parent
                return self

            def plot(self, rotation=0):
                """
                Plots the WOE values using the stored DataFrame and returns the DataFrame.

                Args:
                    rotation (int): Rotation angle for x-axis labels, 0 by default.

                Returns:
                    PlottingDataFrame: Returns the DataFrame object itself after plotting.
                """
                self._parent._plot_woe(self, rotation=rotation)
                return self

            def save(self, path=None, name=None, file_format=".xlsx", type=1, column=None):
                """
                Saves the DataFrame using the _save_file method of the WoeAnalysis class.

                Args:
                    path (str or Path, optional): Directory path where the file will be saved.
                    name (str, optional): File name. If not provided, the column name will be used.
                    file_format (str, optional): Format of the file. Default is ".xlsx".
                    type (int, optional): Type of DataFrame to save if multiple options exist.
                    column (str, optional): Column name used if the name is not provided.

                Returns:
                    PlottingDataFrame: The DataFrame object itself after saving.
                """
                self._parent._save_file(path=path, name=name, format=file_format, type=type, column=column, df1=self)
                return self

        # Create an instance of PlottingDataFrame and set the parent
        result = PlottingDataFrame(df_temp)._set_parent(self)

        self.Variable_types[column] = 'continuous'
        self.Variable_Ranges[column] = bins
        # Return the custom DataFrame with added plot and save capabilities
        return result


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
    def __init__(self, data, WoE_dict, WoeBinning, model, production):
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

