import pandas as pd
from typing import Dict, Any, Optional, List

class CSVDataExtractor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self._load_data()

    def _load_data(self) -> Optional[pd.DataFrame]:
        """Loads the CSV data into a pandas DataFrame."""
        try:
            df = pd.read_csv(self.file_path)
            return df
        except FileNotFoundError:
            print(f"Error: CSV file not found at {self.file_path}")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: CSV file at {self.file_path} is empty.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            return None

    def get_data(self) -> Optional[pd.DataFrame]:
        """Returns the loaded DataFrame."""
        return self.df

    def extract_specific_data(self, column_name: str, value: Any) -> Optional[pd.DataFrame]:
        """Filters the DataFrame by column name and value, and returns the filtered data."""
        if self.df is not None:
            try:
                filtered_df = self.df[self.df[column_name] == value]
                return filtered_df
            except KeyError:
                print(f"Error: Column '{column_name}' not found in the DataFrame.")
                return None
            except Exception as e:
                print(f"An error occurred during filtering: {e}")
                return None
        return None

    def get_column_unique_values(self, column_name: str) -> Optional[List[Any]]:
        """Returns unique values from a specified column."""
        if self.df is not None:
            try:
                return self.df[column_name].unique().tolist()
            except KeyError:
                print(f"Error: Column '{column_name}' not found in the DataFrame.")
                return None
            except Exception as e:
                print(f"An error occurred while getting unique values: {e}")
                return None
        return None 