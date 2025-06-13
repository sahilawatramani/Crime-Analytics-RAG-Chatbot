"""
Query Processor for Crime Analytics RAG Chatbot
"""

import logging
from typing import Dict, Any, List
import pandas as pd
from analytical_processor import AnalyticalProcessor
from query_analyzer import QueryAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class QueryProcessor:
    """Processes natural language queries for crime analytics"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize the query processor with a DataFrame."""
        self.df = df
        self.analytical_processor = AnalyticalProcessor(df)
        self.query_analyzer = QueryAnalyzer()
        logger.info("QueryProcessor initialized")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query and return the results."""
        try:
            logger.info(f"Processing query: {query}")
            
            # Analyze the query to determine its type and extract parameters
            analysis = self.query_analyzer.analyze_query(query)
            logger.debug(f"Query analysis: {analysis}")
            
            # Process the query using the analytical processor
            result = self.analytical_processor.process_analytical_query(query)
            logger.debug(f"Analytical result: {result}")
            
            # Format the response
            if isinstance(result, dict):
                if result.get('query_type') == 'error':
                    return {
                        'status': 'error',
                        'message': result.get('error_message', 'An error occurred'),
                        'suggestions': result.get('suggestions', [])
                    }
                else:
                    return {
                        'status': 'success',
                        'data': result,
                        'summary': result.get('summary', ''),
                        'insights': result.get('insights', [])
                    }
            else:
                return {
                    'status': 'error',
                    'message': 'Invalid result format from analytical processor'
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """Format the result into a human-readable response."""
        try:
            if result['status'] == 'error':
                error_msg = f"😔 I'm sorry, but I ran into an issue: {result['message']}"
                if 'suggestions' in result and result['suggestions']:
                    error_msg += "\n\n💡 **Here are some suggestions that might help:**\n" + "\n".join([f"• {s}" for s in result['suggestions']])
                return error_msg
            
            # Format successful response
            response_parts = []
            
            # Add summary if available
            if 'summary' in result:
                response_parts.append(result['summary'])
            
            # Add insights if available
            if 'insights' in result and result['insights']:
                response_parts.append("\n💡 **Key Insights:**")
                for insight in result['insights']:
                    response_parts.append(f"• {insight}")
            
            # Add data quality information if available
            if 'data' in result and 'data_quality' in result['data']:
                quality = result['data']['data_quality']
                response_parts.append(f"\n📊 **Data Quality:**")
                response_parts.append(f"• Records analyzed: {quality.get('total_records', 'N/A')}")
                response_parts.append(f"• Data completeness: {quality.get('completeness', 'N/A')}%")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}", exc_info=True)
            return f"Error formatting response: {str(e)}"