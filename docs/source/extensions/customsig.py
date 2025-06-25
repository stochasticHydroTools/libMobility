# This file is a Sphinx extension that modifies the signature of functions/methods/classes
# Pybind11 generates signatures in the first line of the docstring, it includes type hints and other redundant information that we put in the docstring itself.

from sphinx.application import Sphinx
import logging
import re


def setup(app: Sphinx):
    app.connect("autodoc-process-signature", process_signature)


def strip_type_hints(signature: str) -> str:
    # This pattern matches parameters with type hints and optional default values
    param_pattern = re.compile(
        r"""
        (\b\w+\b)                              # parameter name
        \s*:\s*                                  # colon and optional whitespace
        (?:[^=,\[\]()]+(?:\[[^\[\]]*\])?)    # base type with optional brackets
        (?:\s*\|\s*[^=,()\[\]]+(?:\[[^\[\]]*\])?)*  # optional union types
        (\s*=\s*[^,()]+)?                        # optional default (capturing group)
    """,
        re.VERBOSE,
    )

    def replacer(match):
        name = match.group(1)
        default = match.group(2) if match.lastindex and match.lastindex >= 2 else None
        if default and default.strip() != "= None":
            return f"{name}{default}"
        else:
            return name

    # First replace all type-hinted parameters with clean ones
    stripped = param_pattern.sub(replacer, signature)
    # Normalize spacing around equals signs
    return re.sub(r"\s*=\s*", " = ", stripped)


def modify_signature(signature):
    # Remove the class method references (e.g., self: ...)
    modified = re.sub(r"\s*self:\s*[\w\.]+\s*,?\s*", "", signature)
    # Remove the type hints from the parameters
    # modified = re.sub(r":\s*[\w\.\[\]]+", "", modified)

    modified = strip_type_hints(modified)
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
