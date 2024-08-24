import pandas as pd
import numpy as np

class Preprocessing:
    def __init__(self):
        self.steps = []

    def add_step(self, func, *args, **kwargs):
        """Add a transformation step to the pipeline."""
        self.steps.append((func, args, kwargs))

    def apply(self, data):
        """Apply all steps to the data."""
        for func, args, kwargs in self.steps:
            data = func(data, *args, **kwargs)
        return data


class WoE_Binning:
    def __init__(self, WoE_dict):
        self.WoE_dict = WoE_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, dummy=False):
        X = X[list(pd.DataFrame({"name": [i.split(":")[0] for i in self.WoE_dict]})["name"].unique())]
        # Initialize a new DataFrame
        X_new = pd.DataFrame(index=X.index)
        matched_rows = pd.DataFrame(False, index=X.index, columns=X.columns)  # Track matched rows for each feature

        # Iterate over each feature in WoE_dict
        for feature, woe_value in self.WoE_dict.items():
            if ':' in feature:  # Handling ranges or distinct values
                category, condition = feature.split(':')

                # Check if it's a range
                if '(' in condition or '[' in condition:
                    bot, top = condition.split(",")
                    bot = float(bot[1:])
                    top = float(top[:-1]) if top[:-1] != 'inf' else np.inf

                    if top == np.inf:
                        mask = (X[category] > bot)
                    else:
                        mask = (X[category] > bot) & (X[category] <= top)
                else:  # Handling distinct categorical values
                    mask = (X[category] == condition)

                # If no rows match the condition, raise an error
                if not mask.any():
                    unmatched_value = X[category][~matched_rows[category]].iloc[0]
                    raise ValueError(
                        f"Error: No rows match the condition for feature '{feature}' with condition '{condition}'. Unmatched value in '{category}': {unmatched_value}")

                # Assign 1 to matching rows and keep NaN for non-matching rows
                X_new[feature] = np.nan
                X_new.loc[mask, feature] = 1

                # Update the matched rows for this feature
                matched_rows.loc[mask, category] = True

        # Explicitly check if all rows have been matched for each feature
        for col in X.columns:
            unmatched_mask = ~matched_rows[col]
            if unmatched_mask.any():
                unmatched_index = X.index[unmatched_mask].tolist()[0]
                unmatched_value = X.loc[unmatched_index, col]
                raise ValueError(
                    f"Error: Value '{unmatched_value}' in column '{col}' at index '{unmatched_index}' is outside the defined WoE_dict ranges.")

        if not dummy:
            # Apply WoE values
            for feature in X_new.columns:
                X_new[feature] *= self.WoE_dict[feature]

            # Aggregate features based on their common prefix and sum them
            final_columns = list(
                pd.DataFrame({"name": [i.split(":")[0] for i in self.WoE_dict]}).drop_duplicates()["name"])
            for col in final_columns:
                mask = [x for x in X_new.columns if x.startswith(col)]
                X_new[col] = X_new[mask].sum(axis=1)

            X_new = X_new[final_columns]

        return X_new