"""
Enhanced Query Processor with CSV-based accurate data extraction
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional
import torch
import numpy as np
import faiss
from datetime import datetime
import uuid
from csv_data_extractor import CSVDataExtractor
from analytical_processor import AnalyticalProcessor
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

logger = logging.getLogger(__name__)

class EnhancedQueryProcessor:
    """Processes natural language queries using RAG with CSV-based accurate data extraction"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        embedding_model: Any,
        nlp_model: Any,
        qa_model: Any,
        generator_model: Any,
        documents: List[str],
        metadata: List[Dict],
        embedding_matrix: torch.Tensor,
        csv_file_path: str = "data/crime_data.csv"
    ):
        self.df = df
        self.embedding_model = embedding_model
        self.nlp_model = nlp_model
        self.qa_model = qa_model
        self.generator_model = generator_model
        self.documents = documents
        self.metadata = metadata
        self.embedding_matrix = embedding_matrix
        self.csv_file_path = csv_file_path
        
        # Initialize analytical processor
        self.analytical_processor = AnalyticalProcessor(df)
        
        # Initialize CSV data extractor for accurate results
        self.csv_extractor = CSVDataExtractor(csv_file_path)
        
        # Comprehensive query patterns for all possible user questions
        self.query_patterns = {
            # Ranking queries
            'ranking': [
                r'top\s+(\d+)\s+(?:states?|districts?|cities?|regions?)',
                r'bottom\s+(\d+)\s+(?:states?|districts?|cities?|regions?)',
                r'highest\s+(?:states?|districts?|cities?|regions?|.*)',
                r'lowest\s+(?:states?|districts?|cities?|regions?|.*)',
                r'rank\s+(?:states?|districts?|cities?|regions?)',
                r'most\s+(?:states?|districts?|cities?|regions?|.*)',
                r'least\s+(?:states?|districts?|cities?|regions?|.*)',
                r'worst\s+(?:states?|districts?|cities?|regions?|.*)',
                r'best\s+(?:states?|districts?|cities?|regions?|.*)',
                r'which\s+(?:state|district|city|region)\s+(?:had|has)\s+(?:the\s+)?(?:highest|lowest|most|least|worst|best)',
                r'(?:highest|lowest|most|least|worst|best)\s+(?:.*?)\s+(?:in|for|during)',
                r'(?:highest|lowest|most|least|worst|best)\s+(?:.*?)\s+(?:cases?|crimes?|incidents?)'
            ],
            
            # Comparison queries
            'comparison': [
                r'compare\s+(?:between\s+)?(.+?)\s+and\s+(.+)',
                r'vs\s+(.+)',
                r'versus\s+(.+)',
                r'difference\s+between\s+(.+?)\s+and\s+(.+)',
                r'which\s+(?:state|district|city|region)\s+(?:has|had)\s+(?:more|less|higher|lower)',
                r'better\s+than',
                r'worse\s+than'
            ],
            
            # Trend queries
            'trend': [
                r'trend\s+(?:of|over|in)',
                r'over\s+time',
                r'year\s+wise',
                r'yearly\s+(?:breakdown|analysis|trend)',
                r'how\s+(?:has|did)\s+(?:it|crime)\s+(?:changed|increased|decreased)',
                r'change\s+(?:over|in)\s+time',
                r'progression',
                r'evolution'
            ],
            
            # Statistical queries
            'statistical': [
                r'total\s+(?:number|count|cases)',
                r'average\s+(?:number|count|cases)',
                r'mean\s+(?:number|count|cases)',
                r'median\s+(?:number|count|cases)',
                r'sum\s+of',
                r'how\s+many',
                r'what\s+is\s+the\s+(?:total|average|mean|median)',
                r'statistics',
                r'statistical\s+summary'
            ],
            
            # Breakdown queries
            'breakdown': [
                r'breakdown\s+(?:by|of)',
                r'state\s+wise',
                r'district\s+wise',
                r'year\s+wise',
                r'category\s+wise',
                r'detailed\s+(?:analysis|breakdown)',
                r'by\s+(?:state|district|year|crime)',
                r'grouped\s+by',
                r'categorized\s+by'
            ],
            
            # Specific queries
            'specific': [
                r'in\s+(.+)',
                r'for\s+(.+)',
                r'during\s+(.+)',
                r'between\s+(.+?)\s+and\s+(.+)',
                r'from\s+(.+?)\s+to\s+(.+)',
                r'specific\s+(?:state|district|year|crime)',
                r'particular\s+(?:state|district|year|crime)'
            ],
            
            # Analysis queries
            'analysis': [
                r'analyze',
                r'analysis\s+of',
                r'study\s+of',
                r'investigate',
                r'examine',
                r'explore',
                r'research',
                r'find\s+out',
                r'discover'
            ],
            
            # Pattern queries
            'pattern': [
                r'pattern',
                r'correlation',
                r'relationship',
                r'connection',
                r'association',
                r'link\s+between',
                r'related\s+to',
                r'connected\s+with'
            ],
            
            # Forecast queries
            'forecast': [
                r'predict',
                r'forecast',
                r'future',
                r'projection',
                r'estimate',
                r'expected',
                r'likely',
                r'prediction'
            ],
            
            # Impact queries
            'impact': [
                r'impact',
                r'effect',
                r'influence',
                r'consequence',
                r'result\s+of',
                r'because\s+of',
                r'due\s+to',
                r'caused\s+by'
            ]
        }
        
        # Crime type mappings
        self.crime_mappings = {
            'dowry': 'Dowry Deaths',
            'dowry deaths': 'Dowry Deaths',
            'dowry death': 'Dowry Deaths',
            'rape': 'Rape',
            'kidnapping': 'Kidnapping and Abduction',
            'kidnap': 'Kidnapping and Abduction',
            'abduction': 'Kidnapping and Abduction',
            'assault': 'Assault on women with intent to outrage her modesty',
            'assault on women': 'Assault on women with intent to outrage her modesty',
            'insult': 'Insult to modesty of Women',
            'insult to modesty': 'Insult to modesty of Women',
            'cruelty': 'Cruelty by Husband or his Relatives',
            'cruelty by husband': 'Cruelty by Husband or his Relatives',
            'importation': 'Importation of Girls',
            'importation of girls': 'Importation of Girls',
            'all crimes': 'all',
            'total crimes': 'all',
            'crime': 'all'
        }
        
        # State mappings
        self.state_mappings = {
            'delhi': 'DELHI',
            'mumbai': 'MAHARASHTRA',
            'maharashtra': 'MAHARASHTRA',
            'andhra pradesh': 'ANDHRA PRADESH',
            'bihar': 'BIHAR',
            'karnataka': 'KARNATAKA',
            'tamil nadu': 'TAMIL NADU',
            'west bengal': 'WEST BENGAL',
            'uttar pradesh': 'UTTAR PRADESH',
            'rajasthan': 'RAJASTHAN',
            'gujarat': 'GUJARAT',
            'kerala': 'KERALA',
            'punjab': 'PUNJAB',
            'haryana': 'HARYANA',
            'assam': 'ASSAM',
            'odisha': 'ODISHA',
            'jharkhand': 'JHARKHAND',
            'chhattisgarh': 'CHHATTISGARH',
            'telangana': 'TELANGANA',
            'goa': 'GOA',
            'manipur': 'MANIPUR',
            'meghalaya': 'MEGHALAYA',
            'mizoram': 'MIZORAM',
            'nagaland': 'NAGALAND',
            'sikkim': 'SIKKIM',
            'tripura': 'TRIPURA',
            'uttarakhand': 'UTTARAKHAND',
            'chandigarh': 'CHANDIGARH',
            'an islands': 'A & N ISLANDS',
            'andaman': 'A & N ISLANDS',
            'nicobar': 'A & N ISLANDS',
            'andaman and nicobar': 'A & N ISLANDS',
            'lakshadweep': 'LAKSHADWEEP',
            'jammu': 'JAMMU & KASHMIR',
            'kashmir': 'JAMMU & KASHMIR',
            'jammu kashmir': 'JAMMU & KASHMIR',
            'd n haveli': 'D & N HAVELI',
            'dadra': 'D & N HAVELI',
            'nagar haveli': 'D & N HAVELI'
        }
        
        # Initialize FAISS index with better error handling
        self.dimension = embedding_matrix.shape[1]
        logger.info(f"Initializing FAISS index with dimension {self.dimension} and {len(documents)} documents")
        
        # Ensure embedding matrix is on CPU and in numpy format
        if isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = embedding_matrix.cpu().numpy()
        
        # Create and populate index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embedding_matrix.astype('float32'))
        
        logger.info(f"Successfully initialized FAISS index and CSV extractor")
    
    def process_with_context(self, query: str) -> Dict[str, Any]:
        """Process query with comprehensive analysis and visualizations"""
        try:
            # Extract query parameters
            params = self._extract_query_parameters(query)
            
            # Determine query type and analysis needed
            analysis_type = self._determine_analysis_type(query, params)
            
            # Process the query based on type
            if analysis_type == 'ranking':
                result = self._process_ranking_query(query, params)
            elif analysis_type == 'comparison':
                result = self._process_comparison_query(query, params)
            elif analysis_type == 'trend':
                result = self._process_trend_query(query, params)
            elif analysis_type == 'statistical':
                result = self._process_statistical_query(query, params)
            elif analysis_type == 'breakdown':
                result = self._process_breakdown_query(query, params)
            elif analysis_type == 'specific':
                result = self._process_specific_query(query, params)
            elif analysis_type == 'analysis':
                result = self._process_analysis_query(query, params)
            elif analysis_type == 'pattern':
                result = self._process_pattern_query(query, params)
            elif analysis_type == 'forecast':
                result = self._process_forecast_query(query, params)
            elif analysis_type == 'impact':
                result = self._process_impact_query(query, params)
            else:
                result = self._process_general_query(query, params)
            
            # Generate visualizations (result already contains 'results' key if analytical)
            visualizations = self._generate_visualizations(query, params, result)
            
            # Format the response
            formatted_answer = self._format_response(result, query)

            print(f"DEBUG: Result before returning from process_with_context: {result}")
            
            return {
                'answer': formatted_answer,
                'confidence': result.get('confidence', 0.85),
                'sources': result.get('sources', []),
                'query_id': result.get('query_id', ''),
                'visualizations': visualizations,
                'analysis_type': result.get('analysis_type', analysis_type), # Use analysis_type from result if available
                'raw_data': result # The full analytical result is here
            }
            
        except Exception as e:
            return {
                'answer': f"I apologize, but I encountered an error processing your query: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'query_id': '',
                'visualizations': [],
                'analysis_type': 'error',
                'raw_data': {}
            }
    
    def _extract_query_parameters(self, query: str) -> Dict[str, Any]:
        """Extract all possible parameters from the query"""
        query_lower = query.lower()
        params = {
            'query': query,
            'crimes': [],
            'states': [],
            'years': [],
            'districts': [],
            'ranking_limit': 5,
            'comparison_type': 'general',
            'analysis_type': 'general'
        }
        
        # Extract crime types
        for crime_keyword, crime_name in self.crime_mappings.items():
            if crime_keyword in query_lower:
                if crime_name == 'all':
                    crime_columns = [col for col in self.df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]
                    params['crimes'].extend(crime_columns)
                else:
                    params['crimes'].append(crime_name)
        
        # Extract states
        for state_keyword, state_name in self.state_mappings.items():
            if state_keyword in query_lower:
                params['states'].append(state_name)
        
        # Extract years
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, query)
        params['years'] = [int(year) for year in years]
        
        # Extract ranking limit
        ranking_pattern = r'(?:top|bottom|rank)\s+(\d+)'
        ranking_match = re.search(ranking_pattern, query_lower)
        if ranking_match:
            params['ranking_limit'] = int(ranking_match.group(1))
        
        # Detect rank type (highest, lowest, both)
        if re.search(r'\b(highest|most|top|best|upper)\b', query_lower) and re.search(r'\b(lowest|least|bottom|worst|lower)\b', query_lower):
            params['rank_type'] = 'both'
            # If asking for "the highest and lowest" (singular), set limit to 1
            if re.search(r'the\s+(highest|lowest)', query_lower):
                params['ranking_limit'] = 1
        elif re.search(r'\b(lowest|least|bottom|worst|lower)\b', query_lower):
            params['rank_type'] = 'lowest'
        elif re.search(r'\b(highest|most|top|best|upper)\b', query_lower):
            params['rank_type'] = 'highest'
        else:
            params['rank_type'] = 'highest' # Default to highest if not specified

        # Adjust ranking_limit if singular highest/lowest is detected but no specific limit
        if (params['rank_type'] == 'highest' or params['rank_type'] == 'lowest') and not ranking_match:
             if re.search(r'the\s+(highest|lowest)', query_lower):
                params['ranking_limit'] = 1
        
        # Extract districts
        district_pattern = r'in\s+([A-Za-z\s]+?)\s+(?:district|city)'
        district_matches = re.findall(district_pattern, query_lower)
        params['districts'] = [d.strip() for d in district_matches]
        
        return params

    def _determine_analysis_type(self, query: str, params: Dict[str, Any]) -> str:
        """Determine the type of analysis needed"""
        query_lower = query.lower()
        
        for analysis_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return analysis_type
        
        return 'general'

    def _process_ranking_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process ranking queries using comprehensive analytical processor"""
        try:
            # Use the analytical processor for comprehensive ranking analysis
            analysis = {
                'analysis_type': 'ranking',
                'extracted_params': {'query': query},
                'ranking_limit': params.get('ranking_limit', 5)
            }
            
            # Use the analytical processor
            result = self.analytical_processor._perform_ranking_analysis(self.df, params, analysis)
            
            if not result or result.get('query_type') == 'error':
                return {
                    'answer': "I couldn't find data for ranking analysis.",
                    'confidence': 0.0,
                    'sources': [],
                    'query_id': 'ranking_no_data'
                }
            
            return {
                'answer': result.get('summary', 'Ranking analysis completed.'),
                'confidence': 0.95,
                'sources': [f"Data from {self.csv_file_path}"],
                'query_id': 'ranking_analysis',
                'results': result.get('results', {}),
                'analysis_type': 'ranking',
                'query_params': params
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing ranking query: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'query_id': 'ranking_error'
            }

    def _process_comparison_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process comparison queries using comprehensive analytical processor"""
        try:
            # Use the analytical processor
            result = self.analytical_processor.process_analytical_query(query)
            
            if not result or result.get('query_type') == 'error':
                return {
                    'answer': "I couldn't find data for comparison analysis.",
                    'confidence': 0.0,
                    'sources': [],
                    'query_id': 'comparison_no_data'
                }
            
            return {
                'answer': result.get('summary', 'Comparison analysis completed.'),
                'confidence': 0.95,
                'sources': [f"Data from {self.csv_file_path}"],
                'query_id': 'comparison_analysis',
                'results': result.get('results', {}),
                'analysis_type': result.get('analysis_type', 'comparison'),
                'query_params': params
            }
                
        except Exception as e:
            return {
                'answer': f"Error processing comparison query: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'query_id': 'comparison_error'
            }

    def _process_trend_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process trend analysis queries using comprehensive analytical processor"""
        try:
            # Use the analytical processor for comprehensive trend analysis
            analysis = {
                'analysis_type': 'trend',
                'extracted_params': {'query': query}
            }
            
            # Use the analytical processor
            result = self.analytical_processor.process_analytical_query(query)
            
            if not result or result.get('query_type') == 'error':
                return {
                    'answer': "I couldn't find data for trend analysis.",
                    'confidence': 0.0,
                    'sources': [],
                    'query_id': 'trend_no_data'
                }
            
            return {
                'answer': result.get('summary', 'Trend analysis completed.'),
                'confidence': 0.95,
                'sources': [f"Data from {self.csv_file_path}"],
                'query_id': 'trend_analysis',
                'results': result.get('results', {}),
                'analysis_type': 'trend',
                'query_params': params
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing trend query: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'query_id': 'trend_error'
            }

    def _process_statistical_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process statistical queries using analytical processor"""
        try:
            # Use the analytical processor for statistical analysis
            analysis = {
                'analysis_type': 'statistical',
                'aggregation_type': 'total',  # Default to total, can be overridden
                'extracted_params': {'query': query}
            }
            
            # Use the analytical processor
            result = self.analytical_processor.process_analytical_query(query)
            
            if not result or result.get('query_type') == 'error':
                return {
                    'answer': "I couldn't find data for statistical analysis.",
                    'confidence': 0.0,
                    'sources': [],
                    'query_id': 'statistical_no_data'
                }
            
            return {
                'answer': result.get('summary', 'Statistical analysis completed.'),
                'confidence': 0.95,
                'sources': [f"Data from {self.csv_file_path}"],
                'query_id': 'statistical_analysis',
                'results': result.get('results', {}),
                'analysis_type': result.get('analysis_type', 'statistical'),
                'query_params': params
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing statistical query: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'query_id': 'statistical_error'
            }

    def _process_breakdown_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process breakdown queries using comprehensive analytical processor"""
        try:
            # Use the analytical processor for comprehensive breakdown analysis
            analysis = {
                'analysis_type': 'breakdown',
                'extracted_params': {'query': query}
            }
            
            # Use the analytical processor
            result = self.analytical_processor.process_analytical_query(query)
            
            if not result or result.get('query_type') == 'error':
                return {
                    'answer': "I couldn't find data for breakdown analysis.",
                    'confidence': 0.0,
                    'sources': [],
                    'query_id': 'breakdown_no_data'
                }
            
            return {
                'answer': result.get('summary', 'Breakdown analysis completed.'),
                'confidence': 0.95,
                'sources': [f"Data from {self.csv_file_path}"],
                'query_id': 'breakdown_analysis',
                'results': result.get('results', {}),
                'analysis_type': 'breakdown',
                'query_params': params
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing breakdown query: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'query_id': 'breakdown_error'
            }

    def _process_specific_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process specific queries"""
        return self._process_general_query(query, params)

    def _process_analysis_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process analysis queries"""
        return self._process_general_query(query, params)

    def _process_pattern_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process pattern queries"""
        return self._process_general_query(query, params)

    def _process_forecast_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process forecast queries"""
        return {
            'answer': "I can analyze historical trends, but I cannot make predictions about future crime rates as this would require additional data and modeling capabilities.",
            'confidence': 0.8,
            'sources': [],
            'query_id': 'forecast_limitation',
            'analysis_type': 'forecast'
        }

    def _process_impact_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process impact queries"""
        return {
            'answer': "I can provide statistical analysis of crime data, but I cannot determine causal relationships or impacts as this would require additional research and analysis.",
            'confidence': 0.8,
            'sources': [],
            'query_id': 'impact_limitation',
            'analysis_type': 'impact'
        }

    def _process_general_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process general queries"""
        try:
            filtered_df = self._filter_data(params)
            
            if filtered_df.empty:
                return {
                    'answer': "I couldn't find data matching your criteria.",
                    'confidence': 0.0,
                    'sources': [],
                    'query_id': 'general_no_data'
                }
            
            # If no crimes specified, use all available
            if not params['crimes']:
                crime_columns = [col for col in self.df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]
                params['crimes'] = crime_columns
            
            results = {}
            summary_parts = []
            
            for crime in params['crimes']:
                if crime in filtered_df.columns:
                    total = filtered_df[crime].sum()
                    results[crime] = {'total': int(total)}
                    summary_parts.append(f"{crime}: {int(total):,} cases")
            
            if not results:
                return {
                    'answer': "I couldn't find the specific crime data you requested.",
                    'confidence': 0.0,
                    'sources': [],
                    'query_id': 'general_no_results'
                }
            
            return {
                'answer': '\n'.join(summary_parts),
                'confidence': 0.8,
                'sources': [f"Data from {self.csv_file_path}"],
                'query_id': 'general_analysis',
                'results': results,
                'analysis_type': 'general'
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing query: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'query_id': 'general_error'
            }

    def _filter_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Filter data based on parameters"""
        filtered_df = self.df.copy()
        
        # Filter by states
        if params.get('states'):
            filtered_df = filtered_df[filtered_df['STATE/UT'].isin(params['states'])]
        
        # Filter by years
        if params.get('years'):
            filtered_df = filtered_df[filtered_df['Year'].isin(params['years'])]
        
        # Filter by districts
        if params.get('districts') and 'DISTRICT' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['DISTRICT'].isin(params['districts'])]
        
        return filtered_df

    def _extract_comparison_entities(self, query: str) -> List[str]:
        """Extract entities to compare from query"""
        entities = []
        
        # Look for "compare X and Y" pattern
        compare_pattern = r'compare\s+(?:between\s+)?(.+?)\s+and\s+(.+)'
        compare_match = re.search(compare_pattern, query.lower())
        if compare_match:
            entities.extend([compare_match.group(1).strip(), compare_match.group(2).strip()])
        
        # Look for "X vs Y" pattern
        vs_pattern = r'(.+?)\s+vs\s+(.+)'
        vs_match = re.search(vs_pattern, query.lower())
        if vs_match:
            entities.extend([vs_match.group(1).strip(), vs_match.group(2).strip()])
        
        # Map entities to state names
        mapped_entities = []
        for entity in entities:
            for keyword, state_name in self.state_mappings.items():
                if keyword in entity.lower():
                    mapped_entities.append(state_name)
                    break
            else:
                # If no mapping found, try to match directly
                if entity.upper() in self.state_mappings.values():
                    mapped_entities.append(entity.upper())
        
        return mapped_entities

    def _generate_visualizations(self, query: str, params: Dict[str, Any], result: Dict[str, Any]) -> List[Any]:
        """Generate visualizations based on query and results"""
        visualizations = []
        
        try:
            analysis_type = result.get('analysis_type', 'general')
            results_data = result.get('results', {})
            
            if analysis_type == 'ranking':
                visualizations.extend(self._create_ranking_charts(results_data))
            elif analysis_type == 'comparison':
                visualizations.extend(self._create_comparison_charts(results_data))
            elif analysis_type == 'trend':
                visualizations.extend(self._create_trend_charts(results_data))
            elif analysis_type == 'statistical':
                visualizations.extend(self._create_statistical_charts(results_data))
            elif analysis_type == 'breakdown':
                visualizations.extend(self._create_breakdown_charts(results_data))
            else:
                visualizations.extend(self._create_general_charts(results_data))
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
        
        return visualizations

    def _create_ranking_charts(self, results_data: Dict[str, Any]) -> List[Any]:
        """Create ranking visualization charts"""
        charts = []
        
        for crime, data in results_data.items():
            if 'top' in data:
                # Create bar chart for top rankings
                top_data = data['top']
                if isinstance(top_data, dict):
                    entities = list(top_data.keys())
                    values = list(top_data.values())
                    
                    # Truncate long entity names
                    short_entities = [str(entity)[:20] + '...' if len(str(entity)) > 20 else str(entity) for entity in entities]
                    
                    fig = px.bar(
                        x=short_entities,
                        y=values,
                        title=f"Top Rankings for {crime}",
                        labels={'x': 'Entity', 'y': 'Cases'},
                        color=values,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        height=400,
                        showlegend=False
                    )
                    charts.append(fig)
        
        return charts

    def _create_comparison_charts(self, results_data: Dict[str, Any]) -> List[Any]:
        """Create comparison visualization charts"""
        charts = []
        
        for crime, data in results_data.items():
            if isinstance(data, dict):
                entities = list(data.keys())
                values = list(data.values())
                
                # Create bar chart
                fig = px.bar(
                    x=entities,
                    y=values,
                    title=f"Comparison for {crime}",
                    labels={'x': 'Entity', 'y': 'Cases'},
                    color=values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=400,
                    showlegend=False
                )
                charts.append(fig)
                
                # Create pie chart if only 2-5 entities
                if 2 <= len(entities) <= 5:
                    fig_pie = px.pie(
                        values=values,
                        names=entities,
                        title=f"Distribution for {crime}"
                    )
                    fig_pie.update_layout(height=400)
                    charts.append(fig_pie)
        
        return charts

    def _create_trend_charts(self, results_data: Dict[str, Any]) -> List[Any]:
        """Create trend visualization charts"""
        charts = []
        
        for crime, data in results_data.items():
            if isinstance(data, dict):
                years = sorted(data.keys())
                values = [data[year] for year in years]
                
                # Create line chart
                fig = px.line(
                    x=years,
                    y=values,
                    title=f"Trend for {crime}",
                    labels={'x': 'Year', 'y': 'Cases'},
                    markers=True
                )
                fig.update_layout(
                    height=400,
                    xaxis=dict(tickmode='linear', tick0=min(years), dtick=1)
                )
                charts.append(fig)
        
        return charts

    def _create_statistical_charts(self, results_data: Dict[str, Any]) -> List[Any]:
        """Create statistical visualization charts"""
        charts = []
        
        for crime, data in results_data.items():
            if isinstance(data, dict) and 'total' in data:
                # Create summary statistics chart
                stats = ['Total', 'Average', 'Median', 'Max', 'Min']
                values = [data.get('total', 0), data.get('average', 0), data.get('median', 0), data.get('max', 0), data.get('min', 0)]
                
                fig = px.bar(
                    x=stats,
                    y=values,
                    title=f"Statistics for {crime}",
                    labels={'x': 'Statistic', 'y': 'Value'},
                    color=values,
                    color_continuous_scale='Greens'
                )
                fig.update_layout(height=400, showlegend=False)
                charts.append(fig)
        
        return charts

    def _create_breakdown_charts(self, results_data: Dict[str, Any]) -> List[Any]:
        """Create breakdown visualization charts"""
        charts = []
        
        for crime, data in results_data.items():
            if isinstance(data, dict) and 'data' in data:
                breakdown_data = data['data']
                if isinstance(breakdown_data, dict):
                    entities = list(breakdown_data.keys())[:10]  # Top 10
                    values = [breakdown_data[entity] for entity in entities]
                    
                    # Truncate long entity names
                    short_entities = [str(entity)[:15] + '...' if len(str(entity)) > 15 else str(entity) for entity in entities]
                    
                    fig = px.bar(
                        x=short_entities,
                        y=values,
                        title=f"Breakdown for {crime}",
                        labels={'x': 'Entity', 'y': 'Cases'},
                        color=values,
                        color_continuous_scale='Purples'
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        height=400,
                        showlegend=False
                    )
                    charts.append(fig)
        
        return charts

    def _create_general_charts(self, results_data: Dict[str, Any]) -> List[Any]:
        """Create general visualization charts"""
        charts = []
        
        for crime, data in results_data.items():
            if isinstance(data, dict) and 'total' in data:
                # Create simple bar chart
                fig = px.bar(
                    x=[crime],
                    y=[data['total']],
                    title=f"Total Cases for {crime}",
                    labels={'x': 'Crime Type', 'y': 'Cases'},
                    color=[data['total']],
                    color_continuous_scale='Oranges'
                )
                fig.update_layout(height=400, showlegend=False)
                charts.append(fig)
        
        return charts

    def _format_response(self, result: Dict[str, Any], query: str) -> str:
        """Format the response in a user-friendly way"""
        # Check if we have analytical processor results
        if 'results' in result and result.get('analysis_type') in ['ranking', 'comparison', 'trend', 'statistical', 'breakdown']:
            # Use the comprehensive analytical processor output
            answer = result.get('answer', '')
            
            # Add confidence indicator
            confidence = result.get('confidence', 0.0)
            if confidence > 0.8:
                answer += "\n\n✅ *High confidence analysis*"
            elif confidence > 0.6:
                answer += "\n\n⚠️ *Moderate confidence analysis*"
            else:
                answer += "\n\n❓ *Low confidence analysis*"
            
            return answer
        else:
            # Fallback to basic answer for other query types
            answer = result.get('answer', '')
            
            # Add confidence indicator
            confidence = result.get('confidence', 0.0)
            if confidence > 0.8:
                answer += "\n\n✅ *High confidence analysis*"
            elif confidence > 0.6:
                answer += "\n\n⚠️ *Moderate confidence analysis*"
            else:
                answer += "\n\n❓ *Low confidence analysis*"
            
            return answer