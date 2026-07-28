from langchain.tools import tool
import pandas as pd

@tool("CSV Data Loader")
def load_csv_data(file_path: str) -> str:
    """
    Reads a local CSV file and returns the column names, data types, 
    and the first 5 rows of data as a formatted string for structural inspection.
    """
    try:
        df = pd.read_csv(file_path)
        info_str = f"Columns and Types:\n{df.dtypes.to_string()}\n\n"
        info_str += f"Shape: {df.shape}\n\n"
        info_str += f"First 5 Rows Preview:\n{df.head().to_string()}"
        return info_str
    except Exception as e:
        return f"Error reading the CSV file: {str(e)}"