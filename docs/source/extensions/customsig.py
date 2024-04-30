# This file is a Sphinx extension that modifies the signature of functions/methods/classes
# Pybind11 generates signatures in the first line of the docstring, it includes type hints and other redundant information that we put in the docstring itself.

from sphinx.application import Sphinx
import logging
import re


def setup(app: Sphinx):
    app.connect("autodoc-process-signature", process_signature)


def modify_signature(signature):
    # Remove the class method references (e.g., self: ...)
    modified = re.sub(r"\s*self:\s*[\w\.]+\s*,?\s*", "", signature)

    # Remove the type hints from the parameters
    modified = re.sub(r":\s*[\w\.\[\]]+", "", modified)

    # Simplify numpy array default values to be just empty
    modified = re.sub(r"array\(\[\],\s*dtype=[\w]+\)", "", modified)

    # Remove any trailing equals signs that now lead nowhere
    modified = re.sub(r"=\s*,", ",", modified)
    modified = re.sub(r"=\s*\)", ")", modified)

    # Remove extra spaces around parameters
    modified = re.sub(r"\s*,\s*", ", ", modified)
    modified = re.sub(r"\s*\(", "(", modified)
    modified = re.sub(r"\s*\)", ")", modified)

    return modified


def process_signature(app, what, name, obj, options, signature, return_annotation):
    if what in ["function", "method", "class"] and signature:
        original_signature = signature
        signature = modify_signature(signature)
        return_annotation = None  # Clear the return annotation if needed
        logging.info(
            f"Modified {name} signature from {original_signature} to {signature}"
        )
        return (signature, return_annotation)
