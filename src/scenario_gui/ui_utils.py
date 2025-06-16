"""UI utility functions for the FleetPy scenario GUI."""
import streamlit as st
from typing import Any, Optional

def render_parameter_input(param: str, param_obj: Any, key_prefix: str = "") -> Optional[str]:
    """Render the appropriate input widget for a parameter based on its type and metadata.
    
    Args:
        param: The parameter name to render
        param_obj: The parameter object containing metadata
        key_prefix: Prefix for the Streamlit widget key
        
    Returns:
        The value from the input widget, or None if no value was entered
    """
    help_text = param_obj.doc_string if hasattr(param_obj, 'doc_string') else ""
    if hasattr(param_obj, 'type') and param_obj.type:
        type_info = f" (Expected type: {param_obj.type})"
        help_text = f"{help_text}{type_info}" if help_text else type_info
    default_value = param_obj.default_value if hasattr(param_obj, 'default_value') else None
    param_type = param_obj.type if hasattr(param_obj, 'type') else "str"

    if hasattr(param_obj, 'options') and param_obj.options:
        options = ["None"] if key_prefix == "optional_" else ["Choose..."]
        options.extend(param_obj.options)
        value = st.selectbox(
            f"{param}",
            options=options,
            key=f"param_{key_prefix}{param}",
            help=help_text
        )
        if value not in ["None", "Choose..."]:
            return value
    elif param_type == "int":
        try:
            default = int(default_value) if default_value and str(default_value).strip() else 0
        except (ValueError, TypeError):
            default = 0
        value = st.number_input(
            f"{param}",
            value=default,
            key=f"param_{key_prefix}{param}",
            help=help_text
        )
        return str(value)
    elif param_type == "float":
        try:
            default = float(default_value) if default_value and str(default_value).strip() else 0.0
        except (ValueError, TypeError):
            default = 0.0
        value = st.number_input(
            f"{param}",
            value=default,
            key=f"param_{key_prefix}{param}",
            help=help_text
        )
        return str(value)
    elif param_type == "bool":
        value = st.checkbox(
            f"{param}",
            value=bool(default_value) if default_value is not None else False,
            key=f"param_{key_prefix}{param}",
            help=help_text
        )
        return str(value)
    else:
        value = st.text_input(
            f"{param}",
            value=str(default_value) if default_value is not None else "",
            key=f"param_{key_prefix}{param}",
            help=help_text
        )
        return value if value else None

def apply_sidebar_styles() -> None:
    """Apply custom CSS styles to the Streamlit sidebar."""
    st.sidebar.markdown("""
        <style>
        section[data-testid="stSidebar"] > div {
            padding-top: 0;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
        section[data-testid="stSidebar"] button[kind="secondary"] {
            background: none;
            text-align: left;
            font-weight: normal;
            padding: 0.5rem 0.75rem;
            color: #262730;
            width: 100%;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border: none;
            border-left: 2px solid transparent;
            border-radius: 0;
            margin: 0;
            justify-content: flex-start;
            min-height: 40px;
            transition: border-left-color 0.2s ease;
        }
        section[data-testid="stSidebar"] button[kind="secondary"]:hover {
            border-left-color: rgba(255, 75, 75, 0.3);
        }
        section[data-testid="stSidebar"] button[kind="secondary"][data-active="true"] {
            color: rgb(255, 75, 75);
            border-left-color: rgb(255, 75, 75);
            font-weight: 600;
        }
        section[data-testid="stSidebar"] div.stTitle:first-child {
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
