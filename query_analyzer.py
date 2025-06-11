"""
Query Analyzer for detecting different types of analytical questions
"""

import re
from typing import Dict, List, Any, Tuple

class QueryAnalyzer:
    """Analyzes queries to determine the type of analysis needed"""
    
    def __init__(self):
        # Define patterns for different types of analytical queries
        self.analysis_patterns = {
            'comparison': {
                'keywords': ['compare', 'vs', 'versus', 'difference', 'higher', 'lower', 'more', 'less', 'highest', 'lowest', 'most', 'least', 'maximum', 'minimum', 'district', 'city', 'region'],
                'patterns': [
                    r'which.*(?:state|year|district|city|region).*(?:highest|most|maximum)',
                    r'which.*(?:state|year|district|city|region).*(?:lowest|least|minimum)',
                    r'(?:highest|most|maximum).*(?:state|year|district|city|region)',
                    r'(?:lowest|least|minimum).*(?:state|year|district|city|region)',
                    r'compare.*(?:between|with)',
                    r'(?:more|less).*(?:than)',
                    r'which.*(?:state|year|district|city|region).*(?:had|has).*(?:highest|lowest)'
                ]
            },
            'ranking': {
                'keywords': ['top', 'bottom', 'rank', 'ranking', 'best', 'worst', 'district', 'city', 'region'],
                'patterns': [
                    r'top\s+\d+',
                    r'bottom\s+\d+',
                    r'rank.*(?:state|year|district|city|region)',
                    r'(?:highest|lowest).*\d+',
                    r'list.*(?:top|bottom)',
                    r'(?:district|city|region).*top',
                    r'top.*(?:district|city|region)'
                ]
            },
            'trend': {
                'keywords': ['trend', 'over time', 'from', 'to', 'between', 'across years', 'yearly', 'annual', 'increase', 'decrease', 'change'],
                'patterns': [
                    r'trend.*(?:from|to|between)',
                    r'(?:increase|decrease).*(?:over|from|to)',
                    r'over time',
                    r'across years',
                    r'year.*(?:trend|pattern)'
                ]
            },
            'breakdown': {
                'keywords': ['breakdown', 'distribution', 'spread', 'range', 'variation', 'pattern', 'year-wise', 'state-wise', 'district-wise', 'category-wise', 'detailed', 'by year', 'by state', 'by district', 'district', 'city', 'region'],
                'patterns': [
                    r'breakdown.*(?:by|of|in)',
                    r'distribution.*(?:across|in|of)',
                    r'spread.*(?:across|in)',
                    r'pattern.*(?:in|across)',
                    r'year-wise.*(?:breakdown|distribution)',
                    r'state-wise.*(?:breakdown|distribution)',
                    r'district-wise.*(?:breakdown|distribution)',
                    r'detailed.*(?:breakdown|analysis)',
                    r'by year',
                    r'by state',
                    r'by district',
                    r'year.*(?:breakdown|distribution)',
                    r'state.*(?:breakdown|distribution)',
                    r'district.*(?:breakdown|distribution)',
                    r'(?:district|city|region).*breakdown',
                    r'breakdown.*(?:district|city|region)'
                ]
            },
            'statistical': {
                'keywords': ['average', 'mean', 'median', 'total', 'sum', 'percentage', '%', 'rate', 'count', 'number'],
                'patterns': [
                    r'average.*(?:per|rate)',
                    r'mean.*(?:per|rate)',
                    r'percentage.*(?:of|in)',
                    r'rate.*(?:per|of)',
                    r'total.*(?:number|count)',
                    r'how many',
                    r'what.*(?:total|number|count)'
                ]
            }
        }
        
        # Crime type mappings
        self.crime_mapping = {
            'rape': 'Rape',
            'dowry deaths': 'Dowry Deaths',
            'cruelty by husband or his relatives': 'Cruelty by Husband or his Relatives',
            'kidnapping and abduction': 'Kidnapping and Abduction',
            'assault on women with intent to outrage her modesty': 'Assault on women with intent to outrage her modesty',
            'insult to modesty of women': 'Insult to modesty of Women',
            'importation of girls': 'Importation of Girls'
        }
        
        # State keywords
        self.state_keywords = [
            'andhra pradesh', 'andhra', 'arunachal pradesh', 'arunachal',
            'assam', 'bihar', 'chhattisgarh', 'goa', 'gujarat', 'haryana',
            'himachal pradesh', 'himachal', 'jharkhand', 'karnataka',
            'kerala', 'madhya pradesh', 'maharashtra', 'manipur',
            'meghalaya', 'mizoram', 'nagaland', 'odisha', 'punjab',
            'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura',
            'uttar pradesh', 'uttarakhand', 'west bengal', 'delhi', 'chandigarh',
            'an islands', 'andaman', 'nicobar', 'andaman and nicobar',
            'lakshadweep', 'jammu', 'kashmir', 'jammu kashmir'
        ]
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query to determine what type of analysis is needed"""
        query_lower = query.lower()
        
        # Extract basic parameters first
        basic_params = self._extract_basic_params(query)
        
        # Check for multi-state and multi-year queries - these should be breakdown queries
        num_states = len(basic_params.get('states', []))
        num_years = len(basic_params.get('years', []))
        
        # If we have multiple states and/or multiple years, prioritize breakdown analysis
        if (num_states > 1 or num_years > 1) and (num_states > 0 or num_years > 0):
            analysis_result = {
                'query_type': 'analytical',
                'analysis_type': 'breakdown',
                'comparison_type': None,
                'ranking_limit': None,
                'aggregation_type': None,
                'extracted_params': basic_params,
                'confidence': 0.9
            }
            return analysis_result
        
        # Prioritize 'total' and 'sum' as statistical queries
        if 'total' in query_lower or 'sum' in query_lower:
            analysis_result = {
                'query_type': 'analytical',
                'analysis_type': 'statistical',
                'comparison_type': None,
                'ranking_limit': None,
                'aggregation_type': 'total',
                'extracted_params': basic_params,
                'confidence': 1.0
            }
            return analysis_result
        
        # Force ranking for 'top' or 'most' and 'district'
        if (('top' in query_lower or 'most' in query_lower) and 'district' in query_lower):
            # If no crime is specified but 'kidnapping' is in the query, set it
            if not basic_params['crimes'] and 'kidnapping' in query_lower:
                basic_params['crimes'] = ['Kidnapping and Abduction']
            analysis_result = {
                'query_type': 'analytical',
                'analysis_type': 'ranking',
                'comparison_type': None,
                'ranking_limit': 5,
                'aggregation_type': None,
                'extracted_params': basic_params,
                'confidence': 1.0
            }
            return analysis_result
        
        analysis_result = {
            'query_type': 'direct_extraction',  # default
            'analysis_type': None,
            'comparison_type': None,
            'ranking_limit': None,
            'aggregation_type': None,
            'extracted_params': basic_params,
            'confidence': 0.0
        }
        
        # Check for different analysis types with priority order
        # Comparison queries should take priority over ranking for "highest/lowest" queries
        
        # First check for comparison patterns (highest, lowest, etc.)
        comparison_confidence = 0.0
        for keyword in self.analysis_patterns['comparison']['keywords']:
            if keyword in query_lower:
                comparison_confidence += 0.3
        
        for pattern in self.analysis_patterns['comparison']['patterns']:
            if re.search(pattern, query_lower):
                comparison_confidence += 0.7
        
        # Check for breakdown patterns
        breakdown_confidence = 0.0
        for keyword in self.analysis_patterns['breakdown']['keywords']:
            if keyword in query_lower:
                breakdown_confidence += 0.3
        
        for pattern in self.analysis_patterns['breakdown']['patterns']:
            if re.search(pattern, query_lower):
                breakdown_confidence += 0.7
        
        # Check for ranking patterns
        ranking_confidence = 0.0
        for keyword in self.analysis_patterns['ranking']['keywords']:
            if keyword in query_lower:
                ranking_confidence += 0.3
        
        for pattern in self.analysis_patterns['ranking']['patterns']:
            if re.search(pattern, query_lower):
                ranking_confidence += 0.7
        
        # Check for trend patterns
        trend_confidence = 0.0
        for keyword in self.analysis_patterns['trend']['keywords']:
            if keyword in query_lower:
                trend_confidence += 0.3
        
        for pattern in self.analysis_patterns['trend']['patterns']:
            if re.search(pattern, query_lower):
                trend_confidence += 0.7
        
        # Check for statistical patterns
        statistical_confidence = 0.0
        for keyword in self.analysis_patterns['statistical']['keywords']:
            if keyword in query_lower:
                statistical_confidence += 0.3
        
        for pattern in self.analysis_patterns['statistical']['patterns']:
            if re.search(pattern, query_lower):
                statistical_confidence += 0.7
        
        # Determine the best analysis type with priority
        if comparison_confidence > 0:
            analysis_result['confidence'] = comparison_confidence
            analysis_result['query_type'] = 'analytical'
            analysis_result['analysis_type'] = 'comparison'
        elif breakdown_confidence > 0:
            analysis_result['confidence'] = breakdown_confidence
            analysis_result['query_type'] = 'analytical'
            analysis_result['analysis_type'] = 'breakdown'
        elif ranking_confidence > 0:
            analysis_result['confidence'] = ranking_confidence
            analysis_result['query_type'] = 'analytical'
            analysis_result['analysis_type'] = 'ranking'
        elif trend_confidence > 0:
            analysis_result['confidence'] = trend_confidence
            analysis_result['query_type'] = 'analytical'
            analysis_result['analysis_type'] = 'trend'
        elif statistical_confidence > 0:
            analysis_result['confidence'] = statistical_confidence
            analysis_result['query_type'] = 'analytical'
            analysis_result['analysis_type'] = 'statistical'
        
        # Determine specific analysis details
        if analysis_result['analysis_type'] == 'comparison':
            analysis_result.update(self._analyze_comparison(query))
        elif analysis_result['analysis_type'] == 'ranking':
            analysis_result.update(self._analyze_ranking(query))
        elif analysis_result['analysis_type'] == 'statistical':
            analysis_result.update(self._analyze_statistical(query))
        
        return analysis_result
    
    def _extract_basic_params(self, query: str) -> Dict[str, Any]:
        """Extract basic parameters from query"""
        query_lower = query.lower()
        
        # Extract years
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        years = [int(year) for year in year_matches]
        
        # Extract crime types
        crimes = []
        sorted_crime_keys = sorted(self.crime_mapping.keys(), key=lambda x: -len(x))
        for crime_key in sorted_crime_keys:
            if crime_key in query_lower:
                crimes.append(self.crime_mapping[crime_key])
        
        # Extract states
        states = []
        for state in self.state_keywords:
            if state in query_lower:
                states.append(state)
        
        return {
            'years': years,
            'crimes': crimes,
            'states': states
        }
    
    def _analyze_comparison(self, query: str) -> Dict[str, Any]:
        """Analyze comparison queries"""
        query_lower = query.lower()
        
        comparison_type = 'highest'
        if any(word in query_lower for word in ['lowest', 'least', 'minimum', 'worst']):
            comparison_type = 'lowest'
        elif any(word in query_lower for word in ['difference', 'compare', 'vs', 'versus']):
            comparison_type = 'difference'
        
        return {
            'comparison_type': comparison_type
        }
    
    def _analyze_ranking(self, query: str) -> Dict[str, Any]:
        """Analyze ranking queries"""
        query_lower = query.lower()
        
        # Extract ranking limit
        ranking_match = re.search(r'(?:top|bottom|rank)\s+(\d+)', query_lower)
        ranking_limit = int(ranking_match.group(1)) if ranking_match else 5
        
        return {
            'ranking_limit': ranking_limit
        }
    
    def _analyze_statistical(self, query: str) -> Dict[str, Any]:
        """Analyze statistical queries"""
        query_lower = query.lower()
        
        aggregation_type = 'sum'
        if 'average' in query_lower or 'mean' in query_lower:
            aggregation_type = 'average'
        elif 'percentage' in query_lower or '%' in query_lower:
            aggregation_type = 'percentage'
        elif 'rate' in query_lower:
            aggregation_type = 'rate'
        
        return {
            'aggregation_type': aggregation_type
        } 