import numpy as np
import json
import os
import time

"""
functions to save and load data for analysis
"""

def find_and_prep_file(filename):
    """
    Ensures filename is available, else change to current directory
    :param filename:
    :return:
    """

    dir_path = os.path.dirname(filename)

    if not os.path.exists(dir_path):
        print(f"cannot locate path {dir_path}, create file in current directory")
        filename = "./" + os.path.basename(filename)
    else:
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

    return filename


def save_array_to_file(array, filename):
    """
    Save a numerical array to a JSON file

    :param array: The array to save
    :param filename: The path to save the file to
    """

    # ensure file is available
    filename = find_and_prep_file(filename)

    # Convert to list if it's a numpy array
    if isinstance(array, np.ndarray):
        array = array.tolist()

    # Save to file
    with open(filename, 'w') as f:
        json.dump(array, f)

    print(f"Array saved to {filename}")


def append_values_to_file(new_values, filename, create_if_missing=True):
    """
    Append new values to an existing array file or create one if it doesn't exist

    :param new_values: Single value or list of values to append
    :param filename: The path to the file
    :param create_if_missing: If True, create the file if it doesn't exist
    :return: True if successful, False otherwise
    """

    # ensure file is available
    filename = find_and_prep_file(filename)

    try:
        # Ensure new_values is a list (convert single value if needed)
        if not isinstance(new_values, (list, np.ndarray)):
            new_values = [new_values]

        # Convert numpy arrays to lists
        if isinstance(new_values, np.ndarray):
            new_values = new_values.tolist()


        # Load existing data or create new file
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"File {filename} is corrupted")
                    return False
        else:
            existing_data = []

        # Append new values and save
        existing_data.extend(new_values)

        with open(filename, 'w') as f:
            json.dump(existing_data, f)

        print(f"Successfully appended {len(new_values)} value(s) to {filename}")
        return True

    except Exception as e:
        print(f"Error appending values to {filename}: {e}")
        return False


def load_array_from_file(filename):
    """
    Load a numerical array from a JSON file

    :param filename: The path to the file to load
    :return: The loaded array
    """

    # ensure file is available
    filename = find_and_prep_file(filename)

    try:
        with open(filename, 'r') as f:
            array = json.load(f)
        print(f"Array loaded from {filename}")
        return array
    except Exception as e:
        print(f"Error loading array from {filename}: {e}")
        return []


if __name__=="__main__":
    arr = [0,1,2,3,4,5,6,]
    save_array_to_file(arr, "./test.json")
    append_values_to_file(1000, "./test.json")
    print(load_array_from_file("./test.json"))
