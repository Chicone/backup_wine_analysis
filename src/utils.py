import numpy as np
import re

def collapse_lists(d):
    for key, value in d.items():
        if isinstance(value, list) and all(isinstance(sublist, list) for sublist in value):
            d[key] = [item for sublist in value for item in sublist]
        elif not isinstance(value, list):
            raise ValueError(f"Value for key '{key}' is not a list or a list of lists.")
    return d

def concatenate_dict_values(d1, d2):
    result = {}
    all_keys = set(d1.keys()).union(set(d2.keys()))
    for key in all_keys:
        result[key] = d1.get(key, []) + d2.get(key, [])
    return result


def find_first_and_last_position(s):
    # Define the regex pattern to find numbers
    pattern = r'\d+'

    # Find the first occurrence
    first_match = re.search(pattern, s)
    first_pos = first_match.start() if first_match else -1

    # Find the last occurrence
    last_match = None
    for match in re.finditer(pattern, s):
        last_match = match
    last_pos = last_match.start() if last_match else -1

    return first_pos, last_pos