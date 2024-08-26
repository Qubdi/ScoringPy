import pandas as pd
import numpy as np

class Preprocessing:
    """
    A class that initializes a preprocessing pipeline for data transformations.

    This class allows you to sequentially apply multiple transformation functions
    to a given dataset.

    Methods:
    --------
    add_step(func, *args, **kwargs):
        Adds a transformation step to the pipeline. The step is stored as a
        tuple containing the function, its positional arguments, and keyword arguments.

    apply(data):
        Applies all the stored transformation steps to the input data sequentially.
        Returns the transformed data.
    """

    def __init__(self):
        # initialize an empty list to hold the transformation steps that are stored as a tuple containing a function,
        # its arguments, and keyword arguments.
        self.steps = []

    def add_step(self, func, *args, **kwargs):
        """
        Add a transformation step to the pipeline.

        Args:
            func (callable): The function to apply as a transformation.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        This method adds a transformation function along with its arguments to the list of steps.
        Each step is a tuple containing the function, its positional arguments, and keyword arguments.
        """
        # adding transformation step  to the steps list.
        self.steps.append((func, args, kwargs))

    def apply(self, data):
        """
        Apply all steps to the data.

        Args:
            data: The input data that will be transformed by the pipeline.

        Returns:
            Transformed data after applying all the steps in the pipeline.

        This method iterates through all the steps in the pipeline, applying each function to the data
        with the corresponding arguments. The data is updated with each transformation,
        and the final transformed data is returned.
        """
        # iterating through each step in the pipeline.
        for func, args, kwargs in self.steps:
            # applying function to the data
            data = func(data, *args, **kwargs)
        return data


class WoE_Binning:
    """
    A class that applies Weight of Evidence (WoE) binning to features in a dataset.

    This class uses a provided dictionary of WoE values to transform the input data by
    applying conditions (such as ranges or distinct values) to specified features.
    The transformed features are then returned as a new DataFrame.

    Methods:
    --------
    fit(X, y=None):
        Fits the WoE_Binning model. This method is included for compatibility with
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
    def __init__(self, WoE_dict):
        # WoE_dict is expected to be a dictionary where keys are features (and possibly conditions)
        # and values are the Weight of Evidence (WoE) values to be applied.
        self.WoE_dict = WoE_dict

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
                if not mask.any():
                    unmatched_value = X[category][~matched_rows[category]].iloc[0]
                    raise ValueError(
                        f"Error: No rows match the condition for feature '{feature}' with condition '{condition}'. Unmatched value in '{category}': {unmatched_value}")

                # Initializing the feature column in X_new with NaN and assign 1 to matching rows.
                X_new[feature] = np.nan
                X_new.loc[mask, feature] = 1

                # updating the matched_rows DataFrame for the current category.
                matched_rows.loc[mask, category] = True

        # checking if all rows have been matched for each feature
        for col in X.columns:
            unmatched_mask = ~matched_rows[col]
            if unmatched_mask.any():
                # if there are unmatched rows, raise an error with details about the first unmatched value.
                unmatched_index = X.index[unmatched_mask].tolist()[0]
                unmatched_value = X.loc[unmatched_index, col]
                raise ValueError(
                    f"Error: Value '{unmatched_value}' in column '{col}' at index '{unmatched_index}' is outside the defined WoE_dict ranges.")

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