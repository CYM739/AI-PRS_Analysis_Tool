# src/logic/helpers.py
import pandas as pd

def _add_polynomial_terms(dataframe, all_alphabet_vars):
    """
    Augments a dataframe with polynomial terms for response surface methodology.
    """
    # Add squared terms
    for header in all_alphabet_vars:
        squared_header = header + '_sq'
        dataframe[header] = pd.to_numeric(dataframe[header], errors='coerce')
        dataframe[squared_header] = dataframe[header] ** 2

    # Add interaction terms if there is more than one independent variable
    if len(all_alphabet_vars) > 1:
        for i in range(len(all_alphabet_vars)):
            for j in range(i + 1, len(all_alphabet_vars)):
                interaction_header = all_alphabet_vars[i] + '*' + all_alphabet_vars[j]
                dataframe[interaction_header] = dataframe[all_alphabet_vars[i]] * dataframe[all_alphabet_vars[j]]