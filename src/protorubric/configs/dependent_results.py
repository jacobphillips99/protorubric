"""
Helper for formatting dependent results into a string. AggregatedQueryConfig objects can be nested, so we need to format them recursively.
For a cleaner architecture, each AggregatedQueryConfig should have a `to_explanation` method that returns a dict with 'explanation' and 'internals' keys.
This module is used to format the dependent results into one string with proper indentation.
"""

from protorubric.configs.aggregating import AggregatedQueryConfig


def format_dependent_results(
    dependent_results: dict[str, AggregatedQueryConfig], include_internals: bool = False
) -> str:
    """
    Format the dependent results into a string.
    """
    if not dependent_results:
        return ""

    formatted_sections = []
    for key, aggregated_query in dependent_results.items():
        explanation_dict = aggregated_query.to_explanation(include_internals=include_internals)
        formatted_explanation = _format_explanation_dict(explanation_dict, key)
        formatted_sections.append(formatted_explanation)
    return "\n\n".join(formatted_sections)


def _format_explanation_dict(
    explanation_dict: dict, section_name: str = "", indent_level: int = 0
) -> str:
    """
    Format a structured explanation dictionary with hierarchical indentation.

    Args:
        explanation_dict: Dict with 'explanation' and 'internals' keys
        section_name: Optional section name to display
        indent_level: Current indentation level

    Returns:
        Formatted string with proper hierarchical indentation
    """
    indent = "  " * indent_level

    # Start with section name if provided
    lines = []
    if section_name and indent_level == 0:
        lines.append(f"Dependency {section_name}:")
        lines.append("")

    # Add the main explanation
    main_explanation = explanation_dict.get("explanation", "")
    if main_explanation:
        lines.append(f"{indent}{main_explanation}")

    # Process internals if they exist
    internals = explanation_dict.get("internals", [])
    if internals:
        lines.append(f"{indent}Computed over the following results:")
        for internal in internals:
            # Format each internal item as a bullet point
            internal_text = _format_explanation_dict(internal, "", indent_level + 1)
            # Add bullet point to first line of internal explanation
            internal_lines = internal_text.split("\n")
            if internal_lines:
                lines.append(f"{indent}- {internal_lines[0].strip()}")
                # Add remaining lines with proper alignment
                for line in internal_lines[1:]:
                    if line.strip():  # Skip empty lines
                        lines.append(f"{indent}  {line.strip()}")

    return "\n".join(lines)
