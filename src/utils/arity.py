# ------------------------------------------------------
# This code is licensed under the MIT License.
# Copyright (c) 2025 Martina Plumari, Daniel Bologna.
# Developed for the course "Computational Intelligence" 
# at Politecnico di Torino.
# ------------------------------------------------------

import re
import numpy as np

def get_function_signature(func) -> str:
    """
    Retrieve the signature of a given numpy function.
    Tries using inspect.signature, falls back to parsing docstrings or np.info.
    """
    
    # Attempt to parse the first line of the docstring
    if func.__doc__:
        first_line = func.__doc__.splitlines()[0]
        return first_line.strip()
    
    # Fallback to using np.info
    try:
        return np.info(func)
    except Exception as e:
        return f"Unable to retrieve signature: {e}"


def arity(func):
    """
    Parse the signature string of the function to count the number of input arguments, without considering optional args.
    WARNING: only tested on numpy functions, may not work for all cases.
    
    Args:
        signature_str (str): The string representation of a function's signature.
    
    Returns:
        int: Number of arguments the function takes, or None if not parseable.
    """
    signature_str = get_function_signature(func)
    # Use a regex to find the argument list inside parentheses
    match = re.search(r"\((.*?)\)", signature_str)
    count = 0
    if not match:
        return None  # Signature string is not in expected format

    args = match.group(1).split(",")  # Split by commas
    # Filter out special markers like `/` (positional only) and `*` (keyword only marker)
    for arg in args:
        if arg.strip() in ("*", "/") or '=' in arg:
            break
        count += 1

    return count