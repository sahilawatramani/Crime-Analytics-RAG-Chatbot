import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from utils import log_error
import json

# Define feedback log file path
FEEDBACK_LOG_FILE = "data/feedback.log"

def get_processed_results(results: Union[str, pd.DataFrame, Dict], query_id: Optional[str] = None) -> Dict:
    """Process results into a format suitable for frontend rendering."""
    processed_output = {}
    if isinstance(results, str):
        processed_output['type'] = 'text'
        processed_output['content'] = results
    elif isinstance(results, pd.DataFrame):
        processed_output['type'] = 'dataframe'
        processed_output['content'] = results.to_dict(orient='records')
        processed_output['columns'] = results.columns.tolist()
    elif isinstance(results, dict):
        processed_output['type'] = 'analysis_results'
        processed_output['content'] = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                processed_output['content'][key] = {
                    'data': value.to_dict(orient='records'),
                    'columns': value.columns.tolist()
                }
            else:
                processed_output['content'][key] = value
    return processed_output

def get_source_documents_data(sources: List[Dict]) -> List[Dict]:
    """Extract source document data for frontend display."""
    if not sources:
        return []
    return [{'content': source.get('content', 'N/A'), 'metadata': source.get('metadata', {})} for source in sources]

def generate_visualizations_json(query: str, df: pd.DataFrame, results: Dict) -> List[Dict]:
    """Generate visualizations and return them as JSON for Plotly.js."""
    charts_json = []
    try:
        if 'data' in results and isinstance(results['data'], pd.DataFrame):
            data_df = results['data']
            if len(data_df) > 0 and 'Year' in data_df.columns:
                numeric_cols = data_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    y_col = numeric_cols[0]
                    fig = px.line(data_df, x='Year', y=y_col, title=f'Trend: {y_col}')
                    charts_json.append(json.loads(fig.to_json()))
    except Exception as e:
        logging.warning(f"Could not generate visualizations: {str(e)}")
    return charts_json

def get_query_suggestions(query: str, df: pd.DataFrame) -> List[str]:
    """Generate query suggestions based on current input"""
    suggestions = []
    try:
        query_lower = query.lower()
        crime_types = [col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]
        for crime in crime_types:
            if crime.lower() in query_lower:
                suggestions.append(f"Show me {crime} statistics")
        suggestions = list(set(suggestions))[:3]
    except Exception as e:
        logging.error(f"Error generating query suggestions: {e}")
        pass
    return suggestions

# Placeholder for PDF generation
def generate_pdf(conversation_data: Dict[str, Any]) -> bytes:
    try:
        from fpdf import FPDF
        import re

        def clean_text_for_pdf(text):
            """Remove emojis and other Unicode characters that can't be encoded in latin-1"""
            if not text:
                return ""
            # Remove emojis and other Unicode characters
            text = re.sub(r'[^\x00-\x7F]+', '', str(text))
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.cell(200, 10, txt="Conversation Export", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        pdf.ln(10)

        for i, conv in enumerate(conversation_data.get('conversations', []), 1):
            pdf.set_font("Arial", "B", size=10)
            timestamp = clean_text_for_pdf(conv.get('timestamp', 'N/A'))
            pdf.cell(0, 10, txt=f"Exchange {i} - {timestamp}", ln=True)
            pdf.set_font("Arial", size=10)
            
            user_text = clean_text_for_pdf(conv.get('user', ''))
            bot_text = clean_text_for_pdf(conv.get('bot', ''))
            confidence = clean_text_for_pdf(conv.get('confidence', ''))
            
            pdf.multi_cell(0, 5, txt=f"User: {user_text}")
            pdf.multi_cell(0, 5, txt=f"Bot: {bot_text}")
            pdf.cell(0, 10, txt=f"Confidence: {confidence}", ln=True)
            pdf.ln(5)
        
        return pdf.output(dest='S').encode('latin1') # Return bytes
    except ImportError:
        logging.error("PDF export requires the 'fpdf' library. Please install it with `pip install fpdf`")
        return b""
    except Exception as e:
        logging.error(f"Error generating PDF: {str(e)}")
        return b""

def get_analytics_dashboard_data(analytics_data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare analytics data for frontend visualization."""
    # This is a placeholder. Real implementation would aggregate and format data
    # to be consumed by a JS charting library.
    return {
        "total_queries": len(analytics_data.get('queries', [])),
        "avg_response_time": np.mean(analytics_data.get('response_times', [0])),
        "satisfaction_scores": analytics_data.get('satisfaction_scores', []),
        "error_rates": analytics_data.get('error_rates', []),
        "feature_usage": analytics_data.get('feature_usage', {})
    }

def generate_automatic_chart_json(results: Dict, df: pd.DataFrame, analysis_type: str) -> Optional[Dict]:
    """Generate an automatic chart based on analysis results and return it as Plotly JSON."""
    if not results or not results.get('data') or not isinstance(results['data'], pd.DataFrame):
        return None

    data_df = results['data']
    chart_figure = None

    try:
        if 'Year' in data_df.columns and len(data_df.columns) > 1:
            # Try to identify a suitable numeric column for Y-axis, excluding 'Year' and 'STATE/UT'
            numeric_cols = [col for col in data_df.columns if pd.api.types.is_numeric_dtype(data_df[col]) and col not in ['Year', 'STATE/UT', 'DISTRICT']]

            if analysis_type == 'trend' and 'Year' in data_df.columns and numeric_cols:
                # Assuming the first numeric column is the primary value
                y_col = numeric_cols[0]
                chart_figure = px.line(data_df, x='Year', y=y_col, title=f'Trend of {y_col} Over Years', markers=True)
                chart_figure.update_layout(xaxis_title='Year', yaxis_title=y_col)
            
            elif analysis_type == 'ranking' and 'STATE/UT' in data_df.columns and numeric_cols:
                y_col = numeric_cols[0]
                chart_figure = px.bar(data_df.sort_values(by=y_col, ascending=False), x='STATE/UT', y=y_col, 
                                      title=f'Ranking of {y_col} by State/UT', text=y_col)
                chart_figure.update_layout(xaxis_title='State/UT', yaxis_title=y_col)
                chart_figure.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            
            elif analysis_type == 'comparison' and 'STATE/UT' in data_df.columns and numeric_cols:
                y_col = numeric_cols[0]
                chart_figure = px.bar(data_df, x='STATE/UT', y=y_col, 
                                      title=f'Comparison of {y_col} Across States/UTs', text=y_col)
                chart_figure.update_layout(xaxis_title='State/UT', yaxis_title=y_col)
                chart_figure.update_traces(texttemplate='%{text:.2s}', textposition='outside')

            elif analysis_type == 'breakdown' and 'STATE/UT' in data_df.columns and numeric_cols:
                # For breakdowns, a pie chart might be useful if breaking down by a categorical variable and a value
                # This example assumes breakdown by STATE/UT with a numeric value
                y_col = numeric_cols[0]
                chart_figure = px.pie(data_df, names='STATE/UT', values=y_col, title=f'Distribution of {y_col} by State/UT')
                chart_figure.update_traces(textposition='inside', textinfo='percent+label')

            elif numeric_cols:
                # Default to bar chart for general statistical queries if a numerical column is available
                if 'STATE/UT' in data_df.columns and len(data_df['STATE/UT'].unique()) > 1:
                    y_col = numeric_cols[0]
                    chart_figure = px.bar(data_df, x='STATE/UT', y=y_col, title=f'{y_col} by State/UT')
                elif 'Year' in data_df.columns and len(data_df['Year'].unique()) > 1:
                    y_col = numeric_cols[0]
                    chart_figure = px.line(data_df, x='Year', y=y_col, title=f'{y_col} by Year')
                else:
                    # Fallback for single value or other structures
                    if len(data_df) == 1 and numeric_cols:
                        value_name = numeric_cols[0]
                        value = data_df[value_name].iloc[0]
                        chart_figure = go.Figure(go.Indicator(mode = "number", value = value, title = { 'text': value_name}))
                    elif numeric_cols:
                        # Create a simple bar chart if no obvious X-axis for categorical data
                        y_col = numeric_cols[0]
                        chart_figure = px.bar(data_df, y=y_col, title=f'{y_col} Distribution')

        if chart_figure:
            return json.loads(chart_figure.to_json())
        
    except Exception as e:
        logging.error(f"Error generating automatic chart ({analysis_type}): {e}")
        return None
    return None 