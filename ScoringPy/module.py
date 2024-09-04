import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path as pth
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
        df['PIV'] = (df['Good Dist'] - df['Bad Dist']) * df['Woe']  # Partition Information Value
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
        # class PlottingDataFrame(pd.DataFrame):
        #     _metadata = ['_parent']
        #
        #     @property
        #     def _constructor(self):
        #         def _c(*args, **kwargs):
        #             return PlottingDataFrame(*args, **kwargs)._set_parent(self._parent)
        #         return _c
        #
        #     def _set_parent(self, parent):
        #         self._parent = parent
        #         return self
        #
        #     def plot(self, rotation=0):
        #         """
        #         Plots the WOE values using the stored DataFrame and returns the DataFrame.
        #
        #         Args:
        #             rotation (int): Rotation angle for x-axis labels, 0 by default.
        #
        #         Returns:
        #             PlottingDataFrame: Returns the DataFrame object itself after plotting.
        #         """
        #         self._parent._plot_woe(self, rotation=rotation)
        #         return self
        #
        #     def save(self, path=None, name=None, file_format=".xlsx", type=1, column=None):
        #         """
        #         Saves the DataFrame using the _save_file method of the WoeAnalysis class.
        #
        #         Args:
        #             path (str or Path, optional): Directory path where the file will be saved.
        #             name (str, optional): File name. If not provided, the column name will be used.
        #             file_format (str, optional): Format of the file. Default is ".xlsx".
        #             type (int, optional): Type of DataFrame to save if multiple options exist.
        #             column (str, optional): Column name used if the name is not provided.
        #
        #         Returns:
        #             PlottingDataFrame: The DataFrame object itself after saving.
        #         """
        #         self._parent._save_file(path=path, name=name, format=file_format, type=type, column=column, df1=self)
        #         return self


        # Internal subclass for plotting and saving.
        class DiscretePlotter:
            def __init__(self, parent, data):
                self._parent = parent
                self._data = data

            def plot(self):
                """Plot the WoE values for the discrete variable."""
                self._parent._plot_woe(self._data)
                plt.show()
                return self

            def save(self, path=None, name=None, file_format=".xlsx", type=1):
                """Save the DataFrame to a specified format and location."""
                self._parent._save_file(path=path, name=name, format=file_format, type=type, column=column, df1=self._data)
                return self

        # Return the instance of the DiscretePlotter with the updated data.
        return DiscretePlotter(self, df_temp2)



        # Create an instance of PlottingDataFrame and set the parent
        # result = PlottingDataFrame(df_temp)._set_parent(self)

        # Return the custom DataFrame with added plot and save capabilities
        # return result










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

        # Return the custom DataFrame with added plot and save capabilities
        return result
