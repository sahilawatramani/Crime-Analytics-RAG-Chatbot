import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import re
from datetime import datetime

class AnalyticalProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.crime_mappings = {
            'rape': 'Rape',
            'dowry deaths': 'Dowry Deaths',
            'dowry': 'Dowry Deaths',
            'dowry death': 'Dowry Deaths',
            'kidnapping': 'Kidnapping and Abduction',
            'kidnapping and abduction': 'Kidnapping and Abduction',
            'abduction': 'Kidnapping and Abduction',
            'assault': 'Assault on women with intent to outrage her modesty',
            'assault on women': 'Assault on women with intent to outrage her modesty',
            'insult': 'Insult to modesty of Women',
            'insult to modesty': 'Insult to modesty of Women',
            'cruelty': 'Cruelty by Husband or his Relatives',
            'cruelty by husband': 'Cruelty by Husband or his Relatives',
            'cruelty by husband or his relatives': 'Cruelty by Husband or his Relatives',
            'importation': 'Importation of Girls',
            'importation of girls': 'Importation of Girls'
        }
        
        self.state_mappings = {
            'chandigarh': 'CHANDIGARH',
            'chhattisgarh': 'CHHATTISGARH',
            'assam': 'ASSAM',
            'arunachal pradesh': 'ARUNACHAL PRADESH',
            'arunachal': 'ARUNACHAL PRADESH',
            'bihar': 'BIHAR',
            'andhra pradesh': 'ANDHRA PRADESH',
            'andhra': 'ANDHRA PRADESH',
            'daman & diu': 'DAMAN & DIU',
            'daman and diu': 'DAMAN & DIU',
            'daman': 'DAMAN & DIU',
            'diu': 'DAMAN & DIU',
            'goa': 'GOA',
            'delhi': 'DELHI',
            'mumbai': 'MAHARASHTRA',
            'maharashtra': 'MAHARASHTRA',
            'karnataka': 'KARNATAKA',
            'tamil nadu': 'TAMIL NADU',
            'tamilnadu': 'TAMIL NADU',
            'west bengal': 'WEST BENGAL',
            'uttar pradesh': 'UTTAR PRADESH',
            'rajasthan': 'RAJASTHAN',
            'gujarat': 'GUJARAT',
            'kerala': 'KERALA',
            'punjab': 'PUNJAB',
            'haryana': 'HARYANA',
            'odisha': 'ODISHA',
            'jharkhand': 'JHARKHAND',
            'telangana': 'TELANGANA',
            'manipur': 'MANIPUR',
            'meghalaya': 'MEGHALAYA',
            'mizoram': 'MIZORAM',
            'nagaland': 'NAGALAND',
            'sikkim': 'SIKKIM',
            'tripura': 'TRIPURA',
            'uttarakhand': 'UTTARAKHAND',
            'an islands': 'A & N ISLANDS',
            'a&n islands': 'A & N ISLANDS',
            'andaman': 'A & N ISLANDS',
            'nicobar': 'A & N ISLANDS',
            'andaman and nicobar': 'A & N ISLANDS',
            'andaman & nicobar': 'A & N ISLANDS',
            'andaman nicobar': 'A & N ISLANDS',
            'andaman and nicobar islands': 'A & N ISLANDS',
            'andaman & nicobar islands': 'A & N ISLANDS',
            'lakshadweep': 'LAKSHADWEEP',
            'jammu': 'JAMMU & KASHMIR',
            'kashmir': 'JAMMU & KASHMIR',
            'jammu kashmir': 'JAMMU & KASHMIR',
            'jammu & kashmir': 'JAMMU & KASHMIR',
            'd n haveli': 'D & N HAVELI',
            'd&n haveli': 'D & N HAVELI',
            'd & n haveli': 'D & N HAVELI',
            'dadra': 'D & N HAVELI',
            'nagar haveli': 'D & N HAVELI',
            'dadra nagar haveli': 'D & N HAVELI',
            'dadra & nagar haveli': 'D & N HAVELI',
            'dadra and nagar haveli': 'D & N HAVELI'
        }
    
    def process_analytical_query(self, query: str) -> Dict[str, Any]:
        """Process analytical queries with comprehensive parameter extraction and analysis"""
        try:
            print(f"DEBUG: Processing query: {query}")
            
            # Extract parameters from query
            params = self._extract_parameters(query)
            print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Extracted params: {params}")
            
            # Always include the original query for downstream formatting logic
            params['extracted_params'] = {'query': query}
            
            # Determine analysis type based on query content
            analysis_type = self._determine_analysis_type(query, params)
            print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Determined analysis_type: {analysis_type}")
        
            # Filter data based on parameters
            filtered_df = self._filter_data(params)
            print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Filtered data shape: {filtered_df.shape}")
            print(f"DEBUG: Unique states in filtered_df: {filtered_df['STATE/UT'].unique().tolist() if 'STATE/UT' in filtered_df.columns else 'N/A'}")
            
            if filtered_df.empty:
                print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Filtered DF is empty.")
                return {
                    'query_type': 'error',
                    'error_message': 'No data found for the specified criteria.',
                    'suggestions': self._generate_suggestions(params)
                }
        
            # Perform analysis based on type
            print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Performing analysis for type: {analysis_type}")
            if analysis_type == 'comparison':
                result = self._perform_comparison_analysis(filtered_df, params, {
                    'analysis_type': analysis_type,
                    'comparison_type': params.get('comparison_type', 'general'),
                    'extracted_params': {'query': query}
                })
                print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Comparison analysis result: {result.get('results', 'No results key in comparison result').keys() if isinstance(result.get('results'), dict) else result.get('results')}")
            elif analysis_type == 'ranking':
                result = self._perform_ranking_analysis(filtered_df, params, {
                    'analysis_type': analysis_type,
                    'ranking_limit': params.get('ranking_limit', 5),
                    'extracted_params': {'query': query}
                })
                print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Ranking analysis result: {result.get('results', 'No results key in ranking result').keys() if isinstance(result.get('results'), dict) else result.get('results')}")
            elif analysis_type == 'trend':
                result = self._perform_trend_analysis(filtered_df, params, {
                    'analysis_type': analysis_type,
                    'extracted_params': {'query': query}
                })
                print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Trend analysis result: {result.get('results', 'No results key in trend result').keys() if isinstance(result.get('results'), dict) else result.get('results')}")
            elif analysis_type == 'statistical':
                result = self._perform_statistical_analysis(filtered_df, params, {
                    'analysis_type': analysis_type,
                    'aggregation_type': params.get('aggregation_type', 'total'),
                    'extracted_params': {'query': query}
                })
                print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Statistical analysis result: {result.get('results', 'No results key in statistical result').keys() if isinstance(result.get('results'), dict) else result.get('results')}")
            elif analysis_type == 'breakdown':
                result = self._perform_breakdown_analysis(filtered_df, params, {
                    'analysis_type': analysis_type,
                    'extracted_params': {'query': query}
                })
                print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Breakdown analysis result: {result.get('results', 'No results key in breakdown result').keys() if isinstance(result.get('results'), dict) else result.get('results')}")
            else:
                # Default to comprehensive analysis for specific queries
                result = self._perform_comprehensive_analysis(filtered_df, params)
                print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Comprehensive analysis result: {result.get('results', 'No results key in comprehensive result').keys() if isinstance(result.get('results'), dict) else result.get('results')}")
            
            # Add query parameters to result for formatting decisions
            result['query_params'] = params
            print(f"DEBUG: AnalyticalProcessor.process_analytical_query - After adding query_params: {result.get('query_params', 'No query_params key')}")
            result['data_quality'] = self._assess_data_quality(filtered_df, params)
            print(f"DEBUG: AnalyticalProcessor.process_analytical_query - After adding data_quality: {result.get('data_quality', 'No data_quality key')}")
            result['insights'] = self._generate_insights(filtered_df, params, result)
            print(f"DEBUG: AnalyticalProcessor.process_analytical_query - After adding insights: {result.get('insights', 'No insights key')}")
            
            print(f"DEBUG: AnalyticalProcessor.process_analytical_query - Final result before return: {result.get('results', 'No results key').keys() if isinstance(result.get('results'), dict) else result.get('results')}")
            
            return result
            
        except Exception as e:
            print(f"DEBUG: Error in process_analytical_query: {str(e)}")
            return {
                'query_type': 'error',
                'error_message': str(e),
                'suggestions': self._generate_suggestions(params)
            }
    
    def _filter_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Filter data based on parameters with comprehensive mapping"""
        filtered_df = self.df.copy()
        
        # Filter by years
        if params.get('years'):
            filtered_df = filtered_df[filtered_df['Year'].isin(params['years'])]
            print(f"DEBUG: After year filtering: {filtered_df.shape}")
        
        # Filter by states with comprehensive mapping
        if params.get('states'):
            mapped_states = []
            for state in params['states']:
                # First try exact match (case-insensitive)
                state_upper = state.upper()
                if state_upper in filtered_df['STATE/UT'].values:
                    mapped_states.append(state_upper)
                elif state in filtered_df['STATE/UT'].values:
                    mapped_states.append(state)
                else:
                    # Try mapping with exact match first
                    mapped_state = self.state_mappings.get(state.lower(), None)
                    if mapped_state and mapped_state in filtered_df['STATE/UT'].values:
                        mapped_states.append(mapped_state)
                    else:
                        # Try partial matching only if no exact match found
                        for available_state in filtered_df['STATE/UT'].unique():
                            if (state.lower() == available_state.lower() or
                                state.lower().replace(' ', '') == available_state.lower().replace(' ', '') or
                                (state.lower() in available_state.lower() and len(state) > 3) or
                                (available_state.lower() in state.lower() and len(available_state) > 3)):
                                mapped_states.append(available_state)
                                break
            
            print(f"DEBUG: Original states: {params['states']}")
            print(f"DEBUG: Mapped states: {mapped_states}")
            print(f"DEBUG: Available states in data: {sorted(filtered_df['STATE/UT'].unique())}")
            
            if mapped_states:
                filtered_df = filtered_df[filtered_df['STATE/UT'].isin(mapped_states)]
                print(f"DEBUG: After state filtering: {filtered_df.shape}")
                print(f"DEBUG: States in filtered data: {sorted(filtered_df['STATE/UT'].unique())}")
        
        return filtered_df
    
    def _perform_comprehensive_analysis(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis for any type of query"""
        try:
            results = {
                'analysis_type': 'comprehensive',
                'results': {},
                'summary': ''
            }
            
            # Get crime columns with mapping
            crime_columns = [col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]
            if params.get('crimes'):
                mapped_crimes = self._map_crime_names(params['crimes'], crime_columns)
                crime_columns = mapped_crimes
            
            print(f"DEBUG: Crime columns to analyze: {crime_columns}")
            
            # Determine the type of analysis needed based on parameters
            if len(params.get('states', [])) > 1:
                # Multiple states - comparison analysis
                return self._perform_comparison_analysis(df, params, {
                    'analysis_type': 'comparison',
                    'comparison_type': 'difference',
                    'extracted_params': {'query': 'comparison'}
                })
            elif len(params.get('years', [])) > 1:
                # Multiple years - trend analysis
                return self._perform_trend_analysis(df, params, {
                    'analysis_type': 'trend',
                    'extracted_params': {'query': 'trend'}
                })
            elif len(params.get('crimes', [])) > 1:
                # Multiple crimes - statistical analysis
                return self._perform_statistical_analysis(df, params, {
                    'analysis_type': 'statistical',
                    'aggregation_type': 'total',
                    'extracted_params': {'query': 'statistical'}
                })
            else:
                # Single criteria - detailed breakdown
                return self._perform_statistical_analysis(df, params, {
                    'analysis_type': 'statistical',
                    'aggregation_type': 'detailed',
                    'extracted_params': {'query': 'detailed'}
                })
            
        except Exception as e:
            return {
                'query_type': 'error',
                'error_message': str(e)
            }
    
    def _perform_statistical_analysis(self, df: pd.DataFrame, params: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis with individual state-year breakdown"""
        try:
            results = {
                'analysis_type': 'statistical',
                'results': {},
                'summary': ''
            }
            
            # Get crime columns with mapping
            crime_columns = [col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]
            if params.get('crimes'):
                mapped_crimes = self._map_crime_names(params['crimes'], crime_columns)
                crime_columns = mapped_crimes
            
            print(f"DEBUG: Crime columns to analyze: {crime_columns}")
            print(f"DEBUG: Available states in data: {sorted(df['STATE/UT'].unique())}")
            print(f"DEBUG: Available years in data: {sorted(df['Year'].unique())}")
            
            summary_parts = []
            
            # Fix period display to show actual years
            actual_years = sorted(df['Year'].unique())
            if len(actual_years) == 1:
                period_display = f"Year {actual_years[0]}"
            else:
                period_display = f"Years {', '.join(map(str, actual_years))}"
            
            summary_parts.append(f"## 📊 **Crime Statistics Analysis**")
            summary_parts.append(f"**Period:** {period_display} | **States:** {len(df['STATE/UT'].unique())} | **Crimes:** {len(crime_columns)}")
            summary_parts.append("")
            
            # Individual Crime Analysis with State-Year breakdown
            for crime in crime_columns:
                if crime in df.columns:
                    summary_parts.append(f"### 🚨 **{crime}**")
                    
                    # Create pivot table for state-year breakdown
                    pivot_table = df.pivot_table(values=crime, index='STATE/UT', columns='Year', aggfunc='sum', fill_value=0)
                    
                    # Show individual state-year breakdown
                    summary_parts.append(f"**State-wise breakdown by year:**")
                    for state in sorted(pivot_table.index):
                        state_data = []
                        for year in sorted(df['Year'].unique()):
                            value = pivot_table.loc[state, year] if year in pivot_table.columns else 0
                            if value > 0:  # Only show non-zero values
                                state_data.append(f"{year}: {value:,}")
                        if state_data:  # Only show states with data
                            summary_parts.append(f"• **{state}:** {', '.join(state_data)}")
                    
                    summary_parts.append("")
                    
                    # Year-wise totals (for trend analysis)
                    year_totals = df.groupby('Year')[crime].sum().sort_index()
                    summary_parts.append(f"**Year-wise totals:** {', '.join([f'{year} ({total:,})' for year, total in year_totals.items()])}")
                    
                    # Key statistics
                    total_cases = df[crime].sum()
                    avg_per_year = total_cases / len(df['Year'].unique())
                    trend = self._calculate_trend(year_totals) if len(year_totals) > 1 else "Insufficient data"
                    
                    summary_parts.append(f"**Stats:** Total {total_cases:,} | Avg/year {avg_per_year:,.0f} | {trend}")
                    
                    # Growth rate if applicable
                    if len(year_totals) > 1:
                        first_year = year_totals.iloc[0]
                        last_year = year_totals.iloc[-1]
                        if first_year > 0:
                            growth_rate = ((last_year - first_year) / first_year) * 100
                            summary_parts.append(f"**Growth:** {growth_rate:+.1f}% ({year_totals.index[0]} to {year_totals.index[-1]})")
                    
                    summary_parts.append("")
            
            # Executive Summary - Compact format
            summary_parts.append(f"## 📋 **Executive Summary**")
            
            # Overall totals
            total_cases_overall = df[crime_columns].sum().sum()
            total_states = len(df['STATE/UT'].unique())
            total_years = len(df['Year'].unique())

            # Store overall totals for visualization
            results['results']['overall_totals'] = {
                'total_cases': total_cases_overall,
                'total_states': total_states,
                'total_years': total_years
            }
            
            summary_parts.append(f"**Overall:** {total_cases_overall:,} total cases | {total_states} states | {total_years} years | Avg {total_cases_overall / (total_states * total_years):,.0f}/state/year")
            summary_parts.append("")
            
            # Crime-wise totals in compact format
            crime_totals = df[crime_columns].sum().sort_values(ascending=False)
            # Store crime-wise totals for visualization
            results['results']['crime_totals'] = crime_totals.to_dict()
            summary_parts.append(f"**Crime-wise:** {', '.join([f'{crime} ({total:,}, {total/total_cases_overall*100:.1f}%)' for crime, total in crime_totals.items()])}")
            
            # State-wise totals in compact format
            state_totals = df.groupby('STATE/UT')[crime_columns].sum().sum(axis=1).sort_values(ascending=False)
            # Store state-wise totals for visualization
            results['results']['state_totals'] = state_totals.to_dict()
            summary_parts.append(f"**State-wise:** {', '.join([f'{state} ({total:,}, {total/total_cases_overall*100:.1f}%)' for state, total in state_totals.items()])}")
            
            # Year-wise totals in compact format
            year_totals_overall = df.groupby('Year')[crime_columns].sum().sum(axis=1).sort_index()
            # Store year-wise totals for visualization
            results['results']['year_totals'] = year_totals_overall.to_dict()
            summary_parts.append(f"**Year-wise:** {', '.join([f'{year} ({total:,}, {total/total_cases_overall*100:.1f}%)' for year, total in year_totals_overall.items()])}")
            summary_parts.append("")
            
            # Key insights in compact format
            most_affected_state = state_totals.index[0]
            least_affected_state = state_totals.index[-1]
            most_common_crime = crime_totals.index[0]
            overall_trend = self._calculate_trend(year_totals) if len(year_totals) > 1 else "Insufficient data"
            
            # Check if this is a "top" or "highest" query to avoid showing lowest state
            query_lower = str(params.get('extracted_params', {}).get('query', '')).lower()
            if any(word in query_lower for word in ['top', 'highest', 'most', 'best']):
                summary_parts.append(f"**Key Insights:** Highest {most_affected_state} ({state_totals.iloc[0]:,}) | Most common {most_common_crime} ({crime_totals.iloc[0]:,}) | {overall_trend}")
            else:
                summary_parts.append(f"**Key Insights:** Highest {most_affected_state} ({state_totals.iloc[0]:,}) | Lowest {least_affected_state} ({state_totals.iloc[-1]:,}) | Most common {most_common_crime} ({crime_totals.iloc[0]:,}) | {overall_trend}")
            
            results['summary'] = "\n".join(summary_parts)
            return results
            
        except Exception as e:
            return {
                'query_type': 'error',
                'error_message': str(e)
            }
    
    def _perform_comparison_analysis(self, df: pd.DataFrame, params: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive comparison analysis with friendly, humanized insights"""
        try:
            comparison_type = analysis.get('comparison_type', 'highest')
            
            results = {
                'analysis_type': 'comparison',
                'comparison_type': comparison_type,
                'results': {},
                'summary': ''
            }
            
            # Get crime columns with mapping
            crime_columns = [col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]
            if params.get('crimes'):
                mapped_crimes = self._map_crime_names(params['crimes'], crime_columns)
                crime_columns = mapped_crimes
            
            summary_parts = []
            summary_parts.append(f"## ⚖️ **Comparing Crime Statistics Across States**")
            summary_parts.append(f"Hi! 👋 I've compared crime data from **{df['Year'].min()} to {df['Year'].max()}** across **{len(df['STATE/UT'].unique())} states and union territories**.")
            summary_parts.append(f"I'm looking for the **{comparison_type}** numbers to help you understand the patterns.")
            summary_parts.append("")
            
            # Initialize data structures for storing visualization data
            all_years = sorted(df['Year'].unique())
            comparison_data = pd.DataFrame(index=all_years, columns=crime_columns).fillna(0)
            overall_crime_totals = pd.Series(dtype=float)

            for crime in crime_columns:
                if crime in df.columns:
                    print(f"DEBUG: _perform_comparison_analysis - Processing crime: {crime}")
                    state_totals = df.groupby('STATE/UT')[crime].sum().sort_values(ascending=False)
                    
                    if len(state_totals) > 0:
                        summary_parts.append(f"### 🚨 **{crime}**")
                        
                        if comparison_type == 'highest':
                            top_state = state_totals.index[0]
                            top_value = state_totals.iloc[0]
                            summary_parts.append(f"**🏆 State with the most cases:** {top_state} reported {top_value:,} cases")
                            
                            # Show top 5 for context
                            summary_parts.append(f"**📊 Here are the top 5 states:**")
                            for i, (state, value) in enumerate(state_totals.head(5).items(), 1):
                                percentage = (value / state_totals.sum() * 100) if state_totals.sum() > 0 else 0
                                summary_parts.append(f"{i}. **{state}:** {value:,} cases ({percentage:.1f}% of all cases)")
                            
                        elif comparison_type == 'lowest':
                            bottom_state = state_totals.index[-1]
                            bottom_value = state_totals.iloc[-1]
                            summary_parts.append(f"**🔻 State with the fewest cases:** {bottom_state} reported {bottom_value:,} cases")
                            
                            # Show bottom 5 for context
                            summary_parts.append(f"**📊 Here are the 5 states with the lowest numbers:**")
                            for i, (state, value) in enumerate(state_totals.tail(5).items(), 1):
                                percentage = (value / state_totals.sum() * 100) if state_totals.sum() > 0 else 0
                                summary_parts.append(f"{i}. **{state}:** {value:,} cases ({percentage:.1f}% of all cases)")
                        
                        elif comparison_type == 'difference':
                            top_state = state_totals.index[0]
                            bottom_state = state_totals.index[-1]
                            top_value = state_totals.iloc[0]
                            bottom_value = state_totals.iloc[-1]
                            difference = top_value - bottom_value
                            
                            summary_parts.append(f"**⚖️ Here's how the states compare:**")
                            summary_parts.append(f"• **Highest:** {top_state} with {top_value:,} cases")
                            summary_parts.append(f"• **Lowest:** {bottom_state} with {bottom_value:,} cases")
                            summary_parts.append(f"• **Difference:** {difference:,} cases between the highest and lowest")
                            
                            if bottom_value > 0:
                                ratio = top_value / bottom_value
                                summary_parts.append(f"• **Scale of difference:** The highest state has {ratio:.1f} times more cases than the lowest")
                        
                        # Statistical insights with friendly language
                        total_cases = state_totals.sum()
                        if total_cases > 0:
                            summary_parts.append(f"**📈 Some interesting numbers:**")
                            summary_parts.append(f"• **Total cases across all states:** {total_cases:,}")
                            summary_parts.append(f"• **Range:** From {state_totals.iloc[-1]:,} to {state_totals.iloc[0]:,} cases")
                            summary_parts.append(f"• **Variation:** States differ by about {state_totals.std():,.0f} cases on average")
                        
                        summary_parts.append("")

                        # Calculate yearly totals for the current crime
                        yearly_crime_totals = df.groupby('Year')[crime].sum()
                        print(f"DEBUG: _perform_comparison_analysis - Yearly totals for {crime}: {yearly_crime_totals.head()}")
                        for year, total in yearly_crime_totals.items():
                            if year in comparison_data.index:
                                comparison_data.loc[year, crime] = total
                        
                        # Calculate overall total for the current crime across all filtered data
                        overall_crime_totals[crime] = df[crime].sum()
                        print(f"DEBUG: _perform_comparison_analysis - Overall total for {crime}: {overall_crime_totals[crime]}")
            
            # Store dataframes for visualization
            print(f"DEBUG: _perform_comparison_analysis - comparison_data before to_dict: {comparison_data.head()}")
            print(f"DEBUG: _perform_comparison_analysis - overall_crime_totals before to_dict: {overall_crime_totals.head()}")
            results['results']['comparison_data'] = comparison_data.to_dict(orient='index')
            results['results']['overall_crime_totals'] = overall_crime_totals.to_dict()
            
            print(f"DEBUG: _perform_comparison_analysis - Results before return: {results['results'].keys()}")

            results['summary'] = "\n".join(summary_parts)
            return results
            
        except Exception as e:
            print(f"DEBUG: _perform_comparison_analysis - Error: {e}")
            return {
                'query_type': 'error',
                'error_message': str(e)
            }
    
    def _perform_ranking_analysis(self, df: pd.DataFrame, params: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive ranking analysis with friendly, humanized insights"""
        try:
            ranking_limit = analysis.get('ranking_limit', 5)
            
            results = {
                'analysis_type': 'ranking',
                'ranking_limit': ranking_limit,
                'results': {},
                'summary': ''
            }
            
            # Check if this is a year-based query
            query_lower = str(params.get('extracted_params', {}).get('query', '')).lower()
            is_year_based_query = any(word in query_lower for word in ['which year', 'what year', 'year with', 'year had', 'year showing', 'year recorded'])
            
            # Get crime columns with mapping
            crime_columns = [col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]
            if params.get('crimes'):
                mapped_crimes = self._map_crime_names(params['crimes'], crime_columns)
                crime_columns = [c for c in mapped_crimes if c in df.columns] # Ensure mapped crimes exist in df
            
            print(f"DEBUG: _perform_ranking_analysis - crime_columns: {crime_columns}")
            
            summary_parts = []
            
            rank_type = params.get('rank_type', 'highest') # Get rank_type, default to 'highest'
            
            if is_year_based_query:
                # Year-based ranking
                if rank_type == 'highest' or rank_type == 'both':
                    summary_parts.append(f"## 📅 **Year with Highest Crime Rates**")
                    summary_parts.append(f"Hello! 👋 I've analyzed which year had the highest crime rates from **{df['Year'].min()} to {df['Year'].max()}**.")
                    summary_parts.append(f"Here are the **years with the highest crime rates** for each crime type:")
                    summary_parts.append("")
                    
                    for crime in crime_columns:
                        if crime in df.columns:
                            year_totals = df.groupby('Year')[crime].sum().sort_values(ascending=False)
                            total_cases = year_totals.sum()
                            
                            summary_parts.append(f"### 🚨 **{crime}**")
                            summary_parts.append(f"**Total cases across all years:** {total_cases:,}")
                            summary_parts.append("")
                            
                            summary_parts.append(f"**🏆 Top {ranking_limit} years with the most cases:**")
                            for i, (year, value) in enumerate(year_totals.head(ranking_limit).items(), 1):
                                percentage = (value / total_cases * 100) if total_cases > 0 else 0
                                summary_parts.append(f"{i}. **{year}:** {value:,} cases ({percentage:.1f}% of all cases)")
                            
                            summary_parts.append("")

                            # Store the results for visualization
                            if crime not in results['results']:
                                results['results'][crime] = {}
                            results['results'][crime]['year_ranking'] = {
                                'year_ranking': year_totals.head(ranking_limit).to_dict(),
                                'rank_type': 'highest'
                            }
                            print(f"DEBUG: Added year_ranking for {crime}: {results['results'][crime]['year_ranking']}")
                
                if rank_type == 'lowest' or rank_type == 'both':
                    summary_parts.append(f"## 📅 **Year with Lowest Crime Rates**")
                    summary_parts.append(f"Hello! 👋 I've analyzed which year had the lowest crime rates from **{df['Year'].min()} to {df['Year'].max()}**.")
                    summary_parts.append(f"Here are the **years with the lowest crime rates** for each crime type:")
                    summary_parts.append("")
                    
                    for crime in crime_columns:
                        if crime in df.columns:
                            year_totals = df.groupby('Year')[crime].sum().sort_values(ascending=True)
                            total_cases = year_totals.sum()
                            
                            summary_parts.append(f"### 🚨 **{crime}**")
                            summary_parts.append(f"**Total cases across all years:** {total_cases:,}")
                            summary_parts.append("")
                            
                            summary_parts.append(f"**📉 Bottom {ranking_limit} years with the least cases:**")
                            for i, (year, value) in enumerate(year_totals.head(ranking_limit).items(), 1):
                                percentage = (value / total_cases * 100) if total_cases > 0 else 0
                                summary_parts.append(f"{i}. **{year}:** {value:,} cases ({percentage:.1f}% of all cases)")
                            
                            summary_parts.append("")

                            # Store the results for visualization
                            if crime not in results['results']:
                                results['results'][crime] = {}
                            results['results'][crime]['year_ranking_lowest'] = {
                                'year_ranking': year_totals.head(ranking_limit).to_dict(),
                                'rank_type': 'lowest'
                            }
                            print(f"DEBUG: Added year_ranking_lowest for {crime}: {results['results'][crime]['year_ranking_lowest']}")

            else:
                # State-based ranking (original logic)
                if rank_type == 'highest' or rank_type == 'both':
                    summary_parts.append(f"## 🏆 **Top {ranking_limit} Rankings by Crime Type**")
                    summary_parts.append(f"Hello! 👋 I've ranked the states based on crime statistics from **{df['Year'].min()} to {df['Year'].max()}**.")
                    summary_parts.append(f"Here are the **top {ranking_limit} states** for each crime type:")
                    summary_parts.append("")
                    
                    for crime in crime_columns:
                        if crime in df.columns:
                            state_totals = df.groupby('STATE/UT')[crime].sum().sort_values(ascending=False)
                            total_cases = state_totals.sum()
                            
                            summary_parts.append(f"### 🚨 **{crime}**")
                            summary_parts.append(f"**Total cases across all states:** {total_cases:,}")
                            summary_parts.append("")
                            
                            summary_parts.append(f"**🏆 Top {ranking_limit} states with the most cases:**")
                            for i, (state, value) in enumerate(state_totals.head(ranking_limit).items(), 1):
                                percentage = (value / total_cases * 100) if total_cases > 0 else 0
                                summary_parts.append(f"{i}. **{state}:** {value:,} cases ({percentage:.1f}% of all cases)")
                            
                            summary_parts.append("")

                            # Store the results for visualization
                            results['results'][crime] = {
                                'state_ranking': state_totals.head(ranking_limit).to_dict(),
                                'rank_type': 'highest'
                            }
                            print(f"DEBUG: Added state_ranking for {crime}: {results['results'][crime]['state_ranking']}")

                if rank_type == 'lowest' or rank_type == 'both':
                    summary_parts.append(f"## 📉 **Bottom {ranking_limit} Rankings by Crime Type**")
                    if rank_type != 'both': # Avoid repeating the intro if already said for highest
                        summary_parts.append(f"Hello! 👋 I've ranked the states based on crime statistics from **{df['Year'].min()} to {df['Year'].max()}**.")
                    summary_parts.append(f"Here are the **bottom {ranking_limit} states** for each crime type:")
                    summary_parts.append("")
                    
                    for crime in crime_columns:
                        if crime in df.columns:
                            state_totals = df.groupby('STATE/UT')[crime].sum().sort_values(ascending=True)
                            total_cases = state_totals.sum()
                            
                            summary_parts.append(f"### 🚨 **{crime}**")
                            summary_parts.append(f"**Total cases across all states:** {total_cases:,}")
                            summary_parts.append("")
                            
                            summary_parts.append(f"**📉 Bottom {ranking_limit} states with the least cases:**")
                            for i, (state, value) in enumerate(state_totals.head(ranking_limit).items(), 1):
                                percentage = (value / total_cases * 100) if total_cases > 0 else 0
                                summary_parts.append(f"{i}. **{state}:** {value:,} cases ({percentage:.1f}% of all cases)")
                            
                            summary_parts.append("")

                            # Store the results for visualization
                            if crime not in results['results']:
                                results['results'][crime] = {}
                            results['results'][crime]['state_ranking_lowest'] = {
                                'state_ranking': state_totals.head(ranking_limit).to_dict(),
                                'rank_type': 'lowest'
                            }
                            print(f"DEBUG: Added state_ranking_lowest for {crime}: {results['results'][crime]['state_ranking_lowest']}")
            
            results['summary'] = "\n".join(summary_parts)
            print(f"DEBUG: Final results['results'] keys: {results['results'].keys()}")
            print(f"DEBUG: Final results dictionary: {results}") # Print the whole results dict
            return results
            
        except Exception as e:
            return {
                'query_type': 'error',
                'error_message': str(e)
            }
    
    def _perform_trend_analysis(self, df: pd.DataFrame, params: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive trend analysis with friendly, humanized insights"""
        try:
            results = {
                'analysis_type': 'trend',
                'results': {},
                'summary': ''
            }
            
            # Get crime columns with mapping
            crime_columns = [col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]
            if params.get('crimes'):
                mapped_crimes = self._map_crime_names(params['crimes'], crime_columns)
                crime_columns = mapped_crimes
            
            summary_parts = []
            summary_parts.append(f"## 📈 **How Crime Trends Changed Over Time**")
            summary_parts.append(f"Hi there! 👋 I've analyzed how crime patterns changed from **{df['Year'].min()} to {df['Year'].max()}** across **{len(df['STATE/UT'].unique())} states and union territories**.")
            summary_parts.append(f"Let me show you how the numbers evolved over the years:")
            summary_parts.append("")
            
            for crime in crime_columns:
                if crime in df.columns:
                    yearly_totals = df.groupby('Year')[crime].sum().sort_index()
                    
                    if len(yearly_totals) > 1:
                        summary_parts.append(f"### 🚨 **{crime}**")
                        
                        # Year-by-year breakdown with friendly language
                        summary_parts.append(f"**📅 Here's how the numbers changed each year:**")
                        for year, value in yearly_totals.items():
                            summary_parts.append(f"• **{year}:** {value:,} cases")
                        
                        # Trend analysis with friendly language
                        trend = self._calculate_trend(yearly_totals)
                        summary_parts.append(f"**📈 Overall pattern:** {trend}")
                        
                        # Peak and lowest years with friendly language
                        peak_year = yearly_totals.idxmax()
                        lowest_year = yearly_totals.idxmin()
                        summary_parts.append(f"• **Peak year:** {peak_year} with {yearly_totals[peak_year]:,} cases")
                        summary_parts.append(f"• **Lowest year:** {lowest_year} with {yearly_totals[lowest_year]:,} cases")
                        
                        # Growth rate with friendly language
                        if len(yearly_totals) >= 2:
                            first_year = yearly_totals.iloc[0]
                            last_year = yearly_totals.iloc[-1]
                            if first_year > 0:
                                growth_rate = ((last_year - first_year) / first_year) * 100
                                if growth_rate > 0:
                                    summary_parts.append(f"• **Overall change:** Cases increased by {growth_rate:+.1f}% from {yearly_totals.index[0]} to {yearly_totals.index[-1]}")
                                else:
                                    summary_parts.append(f"• **Overall change:** Cases decreased by {abs(growth_rate):.1f}% from {yearly_totals.index[0]} to {yearly_totals.index[-1]}")
                        
                        summary_parts.append("")
            
            results['summary'] = "\n".join(summary_parts)
            return results
        
        except Exception as e:
            return {
                'query_type': 'error',
                'error_message': str(e)
            }
    
    def _perform_breakdown_analysis(self, df: pd.DataFrame, params: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive breakdown analysis with friendly, humanized insights"""
        try:
            results = {
                'analysis_type': 'breakdown',
                'results': {},
                'summary': ''
            }
            
            # Get crime columns with mapping
            crime_columns = [col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]
            if params.get('crimes'):
                mapped_crimes = self._map_crime_names(params['crimes'], crime_columns)
                crime_columns = mapped_crimes
            
            summary_parts = []
            summary_parts.append(f"## 🔍 **Detailed Breakdown of Crime Statistics**")
            summary_parts.append(f"Hello! 👋 I've created a detailed breakdown of crime data from **{df['Year'].min()} to {df['Year'].max()}** across **{len(df['STATE/UT'].unique())} states and union territories**.")
            summary_parts.append(f"Let me break this down for you by year and by state:")
            summary_parts.append("")
            
            # Year-wise breakdown with friendly language
            yearly_breakdown = df.groupby('Year')[crime_columns].sum()
            summary_parts.append(f"### 📅 **Year-by-Year Breakdown**")
            summary_parts.append(f"Here's how the numbers looked each year:")
            for year in sorted(yearly_breakdown.index):
                year_total = yearly_breakdown.loc[year].sum()
                summary_parts.append(f"**{year}:** {year_total:,} total cases")
                
                # Breakdown by crime type for this year
                year_data = yearly_breakdown.loc[year]
                # Store crime type breakdown for the specific year if requested
                if params.get('years') and len(params['years']) == 1 and year == params['years'][0]:
                    # Ensure only crime columns are included and sum them up for the given year
                    crime_data_for_year = year_data[crime_columns].drop('Total', errors='ignore').dropna()
                    # Filter out zero values for cleaner visualization
                    crime_data_for_year = crime_data_for_year[crime_data_for_year > 0]
                    results['results']['crime_type_breakdown'] = crime_data_for_year.to_dict()
                    print(f"DEBUG: Stored crime_type_breakdown for year {year}: {crime_data_for_year.to_dict()}")

                for crime in crime_columns:
                    if crime in year_data and year_data[crime] > 0:
                        summary_parts.append(f"• {crime}: {year_data[crime]:,} cases")
                summary_parts.append("")
            
            # State-wise breakdown with friendly language
            state_breakdown = df.groupby('STATE/UT')[crime_columns].sum()
            summary_parts.append(f"### 🏛️ **State-by-State Breakdown**")
            summary_parts.append(f"Here's how each state performed:")
            for state in sorted(state_breakdown.index):
                state_total = state_breakdown.loc[state].sum()
                summary_parts.append(f"**{state}:** {state_total:,} total cases")
                
                # Breakdown by crime type for this state
                state_data = state_breakdown.loc[state]
                for crime in crime_columns:
                    if crime in state_data and state_data[crime] > 0:
                        summary_parts.append(f"• {crime}: {state_data[crime]:,} cases")
                summary_parts.append("")
            
            results['summary'] = "\n".join(summary_parts)
            return results
            
        except Exception as e:
            return {
                'query_type': 'error',
                'error_message': str(e)
            }
    
    def _map_crime_names(self, crimes: List[str], available_crimes: List[str]) -> List[str]:
        """Map crime names to actual column names"""
        mapped_crimes = []
        for crime in crimes:
            if crime in available_crimes:
                mapped_crimes.append(crime)
            else:
                mapped_crime = self.crime_mappings.get(crime.lower(), crime)
                if mapped_crime in available_crimes:
                    mapped_crimes.append(mapped_crime)
                else:
                    # Try partial matching
                    for available_crime in available_crimes:
                        if (crime.lower() in available_crime.lower() or 
                            available_crime.lower() in crime.lower()):
                            mapped_crimes.append(available_crime)
                            break
        
        return list(set(mapped_crimes))  # Remove duplicates
    
    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from query with comprehensive pattern matching"""
        params = {
            'years': [],
            'states': [],
            'crimes': [],
            'comparison_type': 'highest',
            'ranking_limit': 5,
            'aggregation_type': 'total'
        }
        
        query_lower = query.lower()
        
        # Extract years with multiple patterns
        year_patterns = [
            r'\b(20\d{2})\b',
            r'\b(2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011|2012|2013|2014)\b'
        ]
        
        for pattern in year_patterns:
            years = re.findall(pattern, query)
            if years:
                params['years'] = [int(year) for year in years]
                break
        
        # Extract states with improved pattern matching
        # First, try to match multi-word state names
        multi_word_states = [
            'andaman and nicobar islands', 'andaman & nicobar islands', 'andaman and nicobar',
            'arunachal pradesh', 'andhra pradesh', 'west bengal', 'uttar pradesh', 'tamil nadu',
            'dadra and nagar haveli', 'dadra & nagar haveli', 'daman and diu', 'daman & diu',
            'jammu and kashmir', 'jammu & kashmir'
        ]
        
        for state in multi_word_states:
            if state in query_lower:
                params['states'].append(state)
        
        # Then extract single-word states
        single_word_states = [
            'delhi', 'mumbai', 'maharashtra', 'karnataka', 'tamilnadu', 'rajasthan', 'gujarat',
            'kerala', 'punjab', 'haryana', 'odisha', 'jharkhand', 'telangana', 'manipur',
            'meghalaya', 'mizoram', 'nagaland', 'sikkim', 'tripura', 'uttarakhand',
            'chandigarh', 'chhattisgarh', 'assam', 'arunachal', 'bihar', 'andhra', 'goa',
            'jammu', 'kashmir', 'dadra', 'nagar haveli', 'daman', 'diu', 'andaman',
            'nicobar', 'lakshadweep'
        ]
        
        for state in single_word_states:
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + re.escape(state) + r'\b', query_lower):
                if state not in [s.split()[0] for s in params['states']]:  # Avoid duplicates
                    params['states'].append(state)
        
        # Extract crimes with comprehensive patterns
        crime_patterns = [
            r'\b(rape|dowry deaths?|dowry death|dowry|kidnapping|kidnapping and abduction|abduction|assault|assault on women|insult|insult to modesty|cruelty|cruelty by husband|cruelty by husband or his relatives|importation|importation of girls)\b'
        ]
        
        for pattern in crime_patterns:
            crimes = re.findall(pattern, query_lower)
            if crimes:
                params['crimes'].extend(crimes)
        
        # Determine comparison type
        if any(word in query_lower for word in ['lowest', 'least', 'minimum', 'worst']):
            params['comparison_type'] = 'lowest'
        elif any(word in query_lower for word in ['difference', 'compare', 'vs', 'versus', 'between']):
            params['comparison_type'] = 'difference'
        
        # Extract ranking limit
        ranking_match = re.search(r'\b(top|first|best|worst)\s+(\d+)\b', query_lower)
        if ranking_match:
            params['ranking_limit'] = int(ranking_match.group(2))
        
        return params
    
    def _determine_analysis_type(self, query: str, params: Dict[str, Any]) -> str:
        """Determine the type of analysis needed with enhanced logic"""
        query_lower = query.lower()
        
        # Check for year-based queries first (highest priority)
        if any(word in query_lower for word in ['which year', 'what year', 'year with', 'year had', 'year showing', 'year recorded']):
            return 'ranking'  # Year-based ranking
        
        # Check for specific keywords
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'between']):
            return 'comparison'
        elif any(word in query_lower for word in ['top', 'first', 'best', 'worst', 'ranking', 'highest', 'lowest']):
            return 'ranking'
        elif any(word in query_lower for word in ['trend', 'over time', 'yearly', 'annual', 'growth', 'decline', 'pattern', 'change']):
            return 'trend'
        elif any(word in query_lower for word in ['statistics', 'total', 'sum', 'average', 'breakdown']):
            return 'statistical'
        elif any(word in query_lower for word in ['breakdown', 'year-wise', 'state-wise', 'detailed']):
            return 'breakdown'
        
        # Check parameter-based logic
        if len(params.get('states', [])) > 1:
            return 'comparison'
        elif len(params.get('years', [])) > 1:
            return 'trend'
        elif len(params.get('crimes', [])) > 1:
            return 'statistical'
        else:
            return 'statistical'  # Default to comprehensive statistical analysis
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction with friendly, humanized descriptions"""
        if len(series) < 2:
            return "Not enough data to see a clear pattern"
        
        first_half = series.iloc[:len(series)//2].mean()
        second_half = series.iloc[len(series)//2:].mean()
        
        if second_half > first_half * 1.1:
            return "A strong upward trend - cases increased significantly over time"
        elif second_half > first_half * 1.05:
            return "A moderate upward trend - cases increased gradually over time"
        elif second_half < first_half * 0.9:
            return "A strong downward trend - cases decreased significantly over time"
        elif second_half < first_half * 0.95:
            return "A moderate downward trend - cases decreased gradually over time"
        else:
            return "A stable pattern - cases remained relatively consistent over time"
    
    def _assess_data_quality(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality and completeness of the data"""
        return {
            'total_records': len(df),
            'missing_values': df.isnull().sum().sum(),
            'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'coverage': {
                'states': len(df['STATE/UT'].unique()),
                'years': len(df['Year'].unique()),
                'crimes': len([col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']])
            }
        }
    
    def _generate_insights(self, df: pd.DataFrame, params: Dict[str, Any], results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights with friendly language from the analysis"""
        insights = []
        
        # Add insights based on the analysis type and data
        if results.get('analysis_type') == 'statistical':
            total_cases = df[[col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]].sum().sum()
            if total_cases > 0:
                insights.append(f"The data shows {total_cases:,} total cases across the criteria you specified")
        
        return insights
    
    def _generate_suggestions(self, params: Dict[str, Any]) -> List[str]:
        """Generate helpful suggestions with friendly language when no data is found"""
        suggestions = []
        
        if not params.get('states'):
            suggestions.append("Try mentioning specific states like 'BIHAR', 'MAHARASHTRA', or 'DELHI'")
        
        if not params.get('years'):
            suggestions.append("Try specifying years like '2005', '2010', or a range like '2005-2010'")
        
        if not params.get('crimes'):
            suggestions.append("Try mentioning specific crimes like 'rape', 'kidnapping', or 'cruelty'")
        
        suggestions.append("Try using broader criteria or different combinations of states, years, and crimes")
        
        return suggestions
    
    def format_answer(self, results: Dict[str, Any], original_query: str) -> str:
        """Format the analysis results into a friendly, humanized answer"""
        if results.get('query_type') == 'error':
            error_msg = f"😔 I'm sorry, but I ran into a little issue while processing your request: {results.get('error_message', 'Something unexpected happened')}"
            if results.get('suggestions'):
                error_msg += "\n\n💡 **Here are some suggestions that might help:**\n" + "\n".join([f"• {s}" for s in results['suggestions']])
            return error_msg
        
        answer = results.get('summary', 'I couldn\'t find any analysis results for your query. Let me know if you\'d like to try something else!')
        
        # Add friendly data quality information
        if results.get('data_quality'):
            quality = results['data_quality']
            answer += f"\n\n📊 **Quick Note:** I analyzed {quality['total_records']} records with {quality['completeness']:.1f}% data completeness"
        
        # Add friendly insights
        if results.get('insights'):
            answer += f"\n\n💡 **Here's what I found interesting:**\n" + "\n".join([f"• {insight}" for insight in results['insights']])
        
        return answer 