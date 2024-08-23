import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class CustomDataFrameWrapper:
    def __init__(self, df, plot_func, safety_func):
        self.df = df
        self.plot_func = plot_func
        self.safety_func = safety_func

    def plot(self, *args, **kwargs):
        self.plot_func(*args, **kwargs)
        return self  # Return the wrapper object itself for chaining

    def safety(self, *args, **kwargs):
        self.safety_func(*args, **kwargs)
        return self  # Return the wrapper object itself for chaining

    def __getattr__(self, name):
        return getattr(self.df, name)

    def __getitem__(self, key):
        return self.df[key]

    def __setitem__(self, key, value):
        self.df[key] = value




class WoeAnalysis:
    def __init__(self):
        self.WoE_dict = {}
        self.IV_dict = {}
        self.IV_excel = pd.DataFrame(columns=['Partitions', 'Total', 'Total Dist', 'Good', 'Good Rate', 'Bad', 'Bad Rate',
                                              'Good Dist', 'Bad Dist', 'Woe', 'Good Rate Difference', 'Woe Difference',
                                              'IV', 'PIV', 'Variable'])

    def __safety_check(self, df, column, threshold=300):
        """
        Checks if the specified column in the DataFrame has a number of unique values
        exceeding the specified threshold and raises an error if it does.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The name of the column to check.
        threshold (int): The threshold for the number of unique values in the column.
                         If the number of unique values is greater than or equal to this
                         threshold, an error is raised. Default is 5.

        Raises:
        ValueError: If the specified column has unique values greater than or equal to the threshold.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' does not exist in the DataFrame.")

        if df[column].dtype == 'object' and len(df[column].value_counts()) >= threshold:
            raise ValueError(
                f"Column '{column}' has {len(df[column].value_counts())} unique values, which exceeds the limit of {threshold}.")


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
        # Check if the specified column exists in the dataframe
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        # Check if the specified column is of type 'object'
        if df[column].dtype != 'object':
            raise ValueError(f"Column '{column}' is not of type 'object'.")

        # Create dummy variables for the specified column
        df_dummies = pd.get_dummies(df[column], prefix=column, prefix_sep=':')

        # Concatenate the original dataframe with the new dummy columns
        df = pd.concat([df, df_dummies], axis=1)

        return df


    def __woe(self, df, column_name, target_df, type = None):
        """
        counting Woe, PIV, IV and other necessary values for discrete variables

        Args:
            df (dataframe): dataframe which we are working on
            column_name (str): name of variable
            target_df (dataframe): target dataframe
        """


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
        df['Total Dist'] = df['Total'] / df['Total'].sum()
        df['Good'] = df['Good Rate'] * df['Total']
        df['Bad'] = (1 - df['Good Rate']) * df['Total']
        df['Bad Rate'] = 1 - df['Good Rate']
        df['Good Dist'] = df['Good'] / df['Good'].sum()
        df['Bad Dist'] = df['Bad'] / df['Bad'].sum()
        df['Woe'] = np.log(df['Good Dist'] / df['Bad Dist'])
        if type == "discrete":
            df = df.sort_values('Woe').reset_index(drop=True)
        df['Good Rate Difference'] = df['Good Rate'].diff().abs()
        df['Woe Difference'] = df['Woe'].diff().abs()
        df['IV'] = (df['Good Dist'] - df['Bad Dist']) * df['Woe']
        df['PIV'] = df['IV']
        df['IV'] = df['IV'].sum()



        # selecting relevant columns to return
        df = df[[column_name, 'Total', 'Total Dist', 'Good', 'Good Rate', 'Bad', 'Bad Rate',
                 'Good Dist', 'Bad Dist', 'Woe', 'Good Rate Difference', 'Woe Difference', 'IV', 'PIV']]
        return df


    def __plot_woe(self, woe_df, rotation=0):
        """
        Plotting by Woe, Woe on y-axis and subcategories (categories of variables) on x-axis.
        A bar chart is also added with the y values taken from the 'Total' column.

        Args:
            woe_df (DataFrame): DataFrame containing Woe and Total values.
            rotation (int): Rotation angle for x-axis labels, 0 by default.
        """

        # Select rows where the first column's value is not equal to the string 'NaN' and 'nan'
        woe_df = woe_df[(woe_df[woe_df.columns[0]] != 'NaN') &
                        (woe_df[woe_df.columns[0]] != 'nan')]

        # Extract values from the first column of DataFrame 'woe_df', convert them to strings, and make them np.array
        x = np.array(woe_df.iloc[:, 0].apply(str))

        # Used for y-axis plotting
        y_woe = woe_df['Woe']
        y_obs = woe_df['Total']

        # Setting style and creating a figure with dual axes
        sns.set_style("darkgrid", {"grid.color": "0.8", "grid.linestyle": ":"})
        fig, ax2 = plt.subplots(figsize=(18, 6))

        # Plotting the bar chart on the first y-axis
        ax2.bar(x, y_obs, color='steelblue', alpha=0.85, label='Observation Count')
        ax2.set_ylabel('Observation Count')
        ax2.tick_params(axis='x', rotation=rotation)

        # Creating a second y-axis for the Woe line plot
        ax1 = ax2.twinx()
        ax1.plot(x, y_woe, marker='o', linestyle='--', color='k', label='Woe')
        ax1.set_xlabel(woe_df.columns[0])
        ax1.set_ylabel('Weight of Evidence')
        ax1.set_title(f'{woe_df.columns[0]}')
        ax1.grid(False)
        # plt.show()


    def discrete(self, column, df, target):
        """
        Determining discrete features' distributions

        Args:
            column (str): name of variable
            df (dataframe): training data
            target (dataframe): target data
        """
        df_temp = df.copy()

        # Define a safety method to check for high cardinality in the specified column
        def safety(threshold=300, off=False):
            if not off:
               self.__safety_check(df=df_temp, column=column, threshold=threshold)
            return df_temp


        # plotting the distribution of the binned feature based on WOE
        def plot(rotation=0):
            self.__plot_woe(df_temp, rotation=rotation)
            return df_temp





        # converting categorical variables to dummy variables for the especific columns
        df_temp = self.__discrete_dummies(df_temp, column=column)

        # calculating WOE (Weight of Evidence) for the binned feature
        df_temp = self.__woe(df=df_temp, column_name=column, target_df=target, type="discrete")

        self.WoE_dict = {k: v for k, v in self.WoE_dict.items() if f"{column}" not in k}

        # saving the Woe values in a dictionary for each bin of the feature
        for i, row in df_temp.iterrows():
            self.WoE_dict[f'{column}:' + str(row[column])] = row['Woe']

        self.IV_dict = {k: v for k, v in self.IV_dict.items() if f"{column}" not in k}

        # calculating and storing the Information Value (IV) of the feature
        self.IV_dict[column] = df_temp['IV'].values[0]

        # creating a copy of df_temp to modify and store in IV_excel
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

        print(554)
        # Return the custom DataFrame wrapper that includes the plot and safety methods
        return CustomDataFrameWrapper(df_temp,  plot, safety)




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

        # Adding the feature name to the "Variable" column
        df_temp2['Variable'] = column

        # Renaming the original feature column to "Partitions"
        df_temp2 = df_temp2.rename(columns={f'{column}_factor': "Partitions"})

        # Dropping rows in IV_excel where "Variable" equals column
        self.IV_excel = self.IV_excel[self.IV_excel['Variable'] != f'{column}_factor']

        # Concatenating the modified DataFrame to IV_excel
        if self.IV_excel is not None and not self.IV_excel.empty:
            self.IV_excel = pd.concat([self.IV_excel, df_temp2], axis=0)
        else:
            self.IV_excel = df_temp2

        # plotting the distribution of the binned feature based on WOE
        def plot(rotation=0):
            self.__plot_woe(df_temp, rotation=rotation)
            return df_temp

        df_temp.plot = plot  # Attach the plot method to the DataFrame

        return df_temp
