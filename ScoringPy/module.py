import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class WoeAnalysis:
    def __init__(self):
        self.WoE_dict = {}
        self.IV_dict = {}
        self.IV_excel = pd.DataFrame(columns=['Partitions', 'Total', 'Total Dist', 'Good', 'Good Rate', 'Bad', 'Bad Rate',
                                              'Good Dist', 'Bad Dist', 'Woe', 'Good Rate Difference', 'Woe Difference',
                                              'IV', 'PIV', 'Variable'])

    def woe_discrete(self, df, cat_variable_name, y_df, type = None):
        """
        counting Woe, PIV, IV and other necessary values for discrete variables

        Args:
            df (dataframe): dataframe which we are working on
            cat_variable_name (str): name of variable
            y_df (dataframe): target dataframe
        """


        # concatenating feature dataframe on target dataframe vertically (rows)
        df = pd.concat([df[cat_variable_name], y_df], axis=1)

        # Perform two separate groupby operations on the DataFrame 'df':
        # 1. Group by the values in the first column and calculate the count of values in the second column for each group.
        # 2. Group by the values in the first column and calculate the mean of values in the second column for each group.
        # Concatenate the results of these groupby operations side by side along the columns axis.
        df = pd.concat([
            df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
            df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()
        ], axis=1)

        # select all rows (':') and columns at positions 0, 1, and 3 (0-indexed) and
        # assign the resulting subset DataFrame back to 'df'.
        df = df.iloc[:, [0, 1, 3]]

        # replacing the name of the first column with the value from the first column name.
        # renaming the second column as 'Total'.
        # renaming the third column as 'Good Rate'.
        df.columns = [cat_variable_name, 'Total', 'Good Rate']

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
        df = df[[cat_variable_name, 'Total', 'Total Dist', 'Good', 'Good Rate', 'Bad', 'Bad Rate',
                 'Good Dist', 'Bad Dist', 'Woe', 'Good Rate Difference', 'Woe Difference', 'IV', 'PIV']]
        return df


    def plot_woe(self, df_WoE, rotation=0):
        """
        Plotting by Woe, Woe on y-axis and subcategories (categories of variables) on x-axis.
        A bar chart is also added with the y values taken from the 'Total' column.

        Args:
            df_WoE (DataFrame): DataFrame containing Woe and Total values.
            rotation (int): Rotation angle for x-axis labels, 0 by default.
        """

        # Select rows where the first column's value is not equal to the string 'NaN' and 'nan'
        df_WoE = df_WoE[(df_WoE[df_WoE.columns[0]] != 'NaN') &
                        (df_WoE[df_WoE.columns[0]] != 'nan')]

        # Extract values from the first column of DataFrame 'df_WoE', convert them to strings, and make them np.array
        x = np.array(df_WoE.iloc[:, 0].apply(str))

        # Used for y-axis plotting
        y_woe = df_WoE['Woe']
        y_obs = df_WoE['Total']

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
        ax1.set_xlabel(df_WoE.columns[0])
        ax1.set_ylabel('Weight of Evidence')
        ax1.set_title(f'{df_WoE.columns[0]}')
        ax1.grid(False)
        # plt.show()

    def discrete(self, feature_name, X_train, y_train):
        """
        Determining discrete features' distributions

        Args:
            feature_name (str): name of variable
            X_train (dataframe): training data
            y_train (dataframe): target data
        """

        # calculating WOE (Weight of Evidence) for the binned feature
        df_temp = self.woe_discrete(X_train, feature_name, y_train, type="discrete")

        self.WoE_dict = {k: v for k, v in self.WoE_dict.items() if f"{feature_name}" not in k}

        # saving the Woe values in a dictionary for each bin of the feature
        for i, row in df_temp.iterrows():
            self.WoE_dict[f'{feature_name}:' + str(row[feature_name])] = row['Woe']

        self.IV_dict = {k: v for k, v in self.IV_dict.items() if f"{feature_name}" not in k}

        # calculating and storing the Information Value (IV) of the feature
        self.IV_dict[feature_name] = df_temp['IV'].values[0]

        # creating a copy of df_temp to modify and store in IV_excel
        df_temp2 = df_temp.copy()

        # Adding the feature name to the "Variable" column
        df_temp2['Variable'] = feature_name

        # Renaming the original feature column to "Partitions"
        df_temp2 = df_temp2.rename(columns={feature_name: "Partitions"})

        # Dropping rows in IV_excel where "Variable" equals feature_name
        self.IV_excel = self.IV_excel[self.IV_excel['Variable'] != feature_name]

        # Concatenating the modified DataFrame to IV_excel
        self.IV_excel = pd.concat([self.IV_excel, df_temp2], axis=0)

        # plotting the distribution of the binned feature based on WOE
        def plot(rotation=0):
            self.plot_woe(df_temp, rotation=rotation)
            return df_temp

        df_temp.plot = plot  # Attach the plot method to the DataFrame

        return df_temp

    def continuous(self,feature_name, bins, X_train,y_train):
        """
        Determining continous features' distributions

        Args:
            feature_name (str) : name of variable
            bins (tuple) : ranges for continuous features
            X_train (dataframe) : training data
            y_train (dataframe) : target data
            rotation_of_x_axis_labels (int) : rotation of labels on x, 0 by default
        """

        # creating a new factorized column based on binning the specified feature
        X_train[f'{feature_name}_factor'] = pd.cut(X_train[feature_name], bins)

        # calculating WOE (Weight of Evidence) for the binned feature
        df_temp = self.woe_discrete(X_train, f'{feature_name}_factor', y_train, type="continuous")

        self.WoE_dict = {k: v for k, v in self.WoE_dict.items() if f"{feature_name}" not in k}

        for i, row in df_temp.iterrows():
            self.WoE_dict[f'{feature_name}:' + str(row[f'{feature_name}_factor'])] = row['Woe']

        self.IV_dict = {k: v for k, v in self.IV_dict.items() if f"{feature_name}" not in k}

        # calculating and storing the Information Value (IV) of the feature
        self.IV_dict[feature_name] = df_temp['IV'].values[0]

        # creating a copy of df_temp to modify and store in IV_excel
        df_temp2 = df_temp.copy()

        # Adding the feature name to the "Variable" column
        df_temp2['Variable'] = feature_name

        # Renaming the original feature column to "Partitions"
        df_temp2 = df_temp2.rename(columns={f'{feature_name}_factor': "Partitions"})

        # Dropping rows in IV_excel where "Variable" equals feature_name
        self.IV_excel = self.IV_excel[self.IV_excel['Variable'] != f'{feature_name}_factor']

        # Concatenating the modified DataFrame to IV_excel
        self.IV_excel = pd.concat([self.IV_excel, df_temp2], axis=0)

        # plotting the distribution of the binned feature based on WOE
        def plot(rotation=0):
            self.plot_woe(df_temp, rotation=rotation)
            return df_temp

        df_temp.plot = plot  # Attach the plot method to the DataFrame

        return df_temp