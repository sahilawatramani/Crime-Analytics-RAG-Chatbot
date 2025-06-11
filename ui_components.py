import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime
import logging
from utils import log_error
import json

# Define feedback log file path
FEEDBACK_LOG_FILE = "data/feedback.log"

def inject_custom_css():
    """Inject custom CSS for enhanced styling"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def render_chat_message(message: Dict[str, Any], is_user: bool = False) -> None:
    """Render a chat message with enhanced styling and formatting"""
    try:
        if is_user:
            # User message styling
            st.markdown(f"""
            <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px 20px;
                    border-radius: 10px;
                    margin: 15px 0;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    border: none;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                ">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 18px;">👤</span>
                        <span style="font-size: 1.1em; font-weight: 500;">{message.get('user', '')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Bot message styling
            bot_message = message.get('bot', '')
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                color: #2c3e50;
                padding: 15px 20px;
                border-radius: 10px;
                margin: 15px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border: none;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
                <div style="display: flex; align-items: flex-start; gap: 10px;">
                    <span style="font-size: 18px; margin-top: 2px;">🤖</span>
                    <div style="flex: 1;">
                        <div style="font-size: 1.1em; line-height: 1.6; white-space: pre-wrap;">{bot_message}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add timestamp if available
        if 'timestamp' in message:
            st.caption(f"🕐 {message['timestamp']}")

    except Exception as e:
        st.error(f"Error rendering message: {str(e)}")
        st.write(message.get('bot', message.get('user', 'Error displaying message')))

def render_typing_indicator():
    """Render an animated typing indicator"""
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 5px; padding: 10px;">
        <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #667eea; animation: typing 1.4s infinite ease-in-out;"></div>
        <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #667eea; animation: typing 1.4s infinite ease-in-out; animation-delay: 0.2s;"></div>
        <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #667eea; animation: typing 1.4s infinite ease-in-out; animation-delay: 0.4s;"></div>
    </div>
    <style>
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    </style>
    """, unsafe_allow_html=True)

def render_results(results: Union[str, pd.DataFrame, Dict], query_id: Optional[str] = None):
    """Render results in a modern, interactive format"""
    if isinstance(results, str):
        st.markdown(results)
    elif isinstance(results, pd.DataFrame):
        st.markdown("### 📊 Results")
        st.dataframe(results, use_container_width=True)
    elif isinstance(results, dict):
        st.markdown("### 📊 Analysis Results")
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                st.markdown(f"#### {key}")
                st.dataframe(value, use_container_width=True)
            else:
                st.write(f"**{key}:** {value}")

def render_source_documents(sources: List[Dict]):
    """Render source documents with enhanced styling"""
    if sources:
        st.markdown("### 📚 Source Documents")
        for i, source in enumerate(sources):
            with st.expander(f"📄 Source {i+1}"):
                st.write(f"**Content:** {source.get('content', 'N/A')}")
                if 'metadata' in source:
                    st.write(f"**Metadata:** {source['metadata']}")

def generate_visualizations(query: str, df: pd.DataFrame, results: Dict) -> List[go.Figure]:
    """Generate visualizations based on query type and results"""
    charts = []
    try:
        # Basic visualization logic
        if 'data' in results and isinstance(results['data'], pd.DataFrame):
            data_df = results['data']
            if len(data_df) > 0 and 'Year' in data_df.columns:
                numeric_cols = data_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    y_col = numeric_cols[0]
                    fig = px.line(data_df, x='Year', y=y_col, title=f'Trend: {y_col}')
                    charts.append(fig)
    except Exception as e:
        st.warning(f"Could not generate visualizations: {str(e)}")
    return charts

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
        st.error("PDF export requires the 'fpdf' library. Please install it with `pip install fpdf`")
        return b""
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return b""

def export_conversation():
    """Export conversation history in multiple formats"""
    st.markdown("### 📤 Export Options")
    
    if 'conversation' in st.session_state and st.session_state.conversation:
        conversation_data = {
            'timestamp': datetime.now().isoformat(),
            'conversations': st.session_state.conversation
        }
        
        # Create export options in columns with adjusted widths
        col1, col2, col3, col4 = st.columns([1, 1, 1.5, 1]) # Adjusted column widths for better spacing, removed summary column
        
        with col1:
            json_str = json.dumps(conversation_data, indent=2, default=str)
            st.download_button(
                label="📄", # Only emoji
                data=json_str,
                file_name="conversation.json",
                mime="application/json",
            )
            st.markdown("<div style='text-align: center; font-size: 0.9em; margin-top: 5px;'>JSON</div>", unsafe_allow_html=True) # Text below, added margin-top
        
        with col2:
            # Flatten conversation data for CSV
            csv_data = []
            for conv in st.session_state.conversation:
                csv_data.append({
                    'timestamp': conv.get('timestamp', ''),
                    'user_query': conv.get('user', ''),
                    'bot_response': conv.get('bot', ''),
                    'confidence': conv.get('confidence', ''),
                    'query_id': conv.get('query_id', '')
                })
            df = pd.DataFrame(csv_data)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊", # Only emoji
                data=csv,
                file_name="conversation.csv",
                mime="text/csv",
            )
            st.markdown("<div style='text-align: center; font-size: 0.9em; margin-top: 5px;'>CSV</div>", unsafe_allow_html=True) # Text below, added margin-top
        
        with col3:
            # Create plain text format
            txt_content = f"Conversation Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            txt_content += "=" * 50 + "\n\n"
            for i, conv in enumerate(st.session_state.conversation, 1):
                txt_content += f"Exchange {i} - {conv.get('timestamp', '')}\n"
                txt_content += f"User: {conv.get('user', '')}\n"
                txt_content += f"Bot: {conv.get('bot', '')}\n"
                txt_content += f"Confidence: {conv.get('confidence', '')}\n"
                txt_content += "-" * 30 + "\n\n"
            
            st.download_button(
                label="📝", # Only emoji
                data=txt_content.encode('utf-8'),
                file_name="conversation.txt",
                mime="text/plain",
            )
            st.markdown("<div style='text-align: center; font-size: 0.9em; margin-top: 5px;'>TXT</div>", unsafe_allow_html=True) # Changed from Transcript to TXT, added margin-top
        
        with col4:
            pdf_bytes = generate_pdf(conversation_data)
            if pdf_bytes:
                st.download_button(
                    label="📄", # Only emoji
                    data=pdf_bytes,
                    file_name="conversation.pdf",
                    mime="application/pdf",
                )
                st.markdown("<div style='text-align: center; font-size: 0.9em; margin-top: 5px;'>PDF</div>", unsafe_allow_html=True) # Text below, added margin-top
        
        # Add a note about the export
        st.caption("💡 Exports include timestamps, queries, responses, and confidence scores where available.")
    else:
        st.warning("No conversation to export")

def collect_feedback(query_id: str, answer: str):
    """Collect user feedback on responses"""
    st.markdown("### 💬 Feedback")
    col1, col2, col3 = st.columns(3)
    
    user_id = st.session_state.get('user_id', 'anonymous') # Get user ID from session state
    timestamp = datetime.now().isoformat()

    feedback_data = {
        "query_id": query_id,
        "answer": answer,
        "user_id": user_id,
        "timestamp": timestamp,
        "feedback_type": None,
        "suggestion": None
    }

    with col1:
        if st.button("👍 Helpful", key=f"helpful_{query_id}"):
            feedback_data["feedback_type"] = "helpful"
            with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback_data) + "\n")
            st.success("Thank you for your feedback!")
    with col2:
        if st.button("👎 Not Helpful", key=f"not_helpful_{query_id}"):
            feedback_data["feedback_type"] = "not_helpful"
            print(f"DEBUG: Attempting to write not helpful feedback for query_id: {query_id}")
            try:
                with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(feedback_data) + "\n")
                print(f"DEBUG: Successfully wrote not helpful feedback for query_id: {query_id}")
                st.error("We're sorry! Please try rephrasing your question.")
            except Exception as file_e:
                print(f"DEBUG: Error writing not helpful feedback to file for query_id {query_id}: {file_e}")
                st.error(f"Error storing feedback: {file_e}")
    with col3:
        if st.button("🔄 Suggest Improvement", key=f"suggest_{query_id}"):
            suggestion = st.text_input("Your suggestion:", key=f"suggestion_{query_id}")
            if suggestion:
                feedback_data["feedback_type"] = "suggestion"
                feedback_data["suggestion"] = suggestion
                print(f"DEBUG: Attempting to write suggestion feedback for query_id: {query_id}")
                try:
                    with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(feedback_data) + "\n")
                    print(f"DEBUG: Successfully wrote suggestion feedback for query_id: {query_id}")
                    st.success("Thank you for your suggestion!")
                except Exception as file_e:
                    print(f"DEBUG: Error writing suggestion feedback to file for query_id {query_id}: {file_e}")
                    st.error(f"Error storing feedback: {file_e}")

def render_analytics_dashboard():
    """Render analytics dashboard"""
    st.markdown("### 📊 Analytics Dashboard")
    if 'analytics_data' in st.session_state:
        analytics = st.session_state.analytics_data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", len(analytics.get('queries', [])))
        with col2:
            avg_time = np.mean(analytics.get('response_times', [0])) if analytics.get('response_times') else 0
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        with col3:
            avg_satisfaction = np.mean(analytics.get('satisfaction_scores', [0])) if analytics.get('satisfaction_scores') else 0
            st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5")
        with col4:
            error_rate = len(analytics.get('error_rates', [])) / max(len(analytics.get('queries', [])), 1) * 100
            st.metric("Error Rate", f"{error_rate:.1f}%")
        # Add a metric for crime types count based on the columns of the main dataframe
        # Assuming df is available in session state or passed as an argument to this function
        # For now, we will add it here. If df is not in session_state, it needs to be passed
        # to this function or retrieved from session_state.
        if 'df' in st.session_state:
            crime_types_count = len([col for col in st.session_state.df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']])
            st.metric("Crime Types", crime_types_count)

def render_filters(df: pd.DataFrame, on_filter_change: Callable) -> None:
    """Render filter controls"""
    st.markdown("### 🔍 Filters")
    years = sorted(df['Year'].unique())
    selected_years = st.multiselect("Years", years, default=[])
    states = sorted(df['STATE/UT'].unique())
    selected_states = st.multiselect("States/UTs", states, default=[])
    # Exclude index column and non-crime columns
    crime_types = [col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0'] and col is not None]
    selected_crimes = st.multiselect("Crime Types", crime_types, default=[])
    if st.button("Generate Prompt"):
        on_filter_change(selected_years, selected_states, selected_crimes)

def generate_automatic_chart(results: Dict, df: pd.DataFrame, analysis_type: str) -> Optional[go.Figure]:
    """Generate automatic visualization based on query results and analysis type"""
    try:
        print(f"DEBUG: generate_automatic_chart called")
        print(f"DEBUG: analysis_type: {analysis_type}")
        print(f"DEBUG: results keys: {results.keys() if results else 'None'}")
        print(f"DEBUG: Full results dictionary: {results}")
        
        # Get crime columns for fallback
        crime_columns = [col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]
        
        # Enhanced data extraction - check multiple possible locations
        crime_totals = None
        state_totals = None
        year_totals = None
        comparison_data = None
        state_ranking = None
        state_ranking_lowest = None
        
        # Extract data from various possible locations based on debug output
        if 'crime_totals' in results:
            crime_totals = results['crime_totals']
        elif 'overall_crime_totals' in results:
            crime_totals = results['overall_crime_totals']
        
        if 'state_totals' in results:
            state_totals = results['state_totals']
        
        if 'year_totals' in results:
            year_totals = results['year_totals']
        
        if 'comparison_data' in results:
            comparison_data = results['comparison_data']
        
        # Check for nested crime data (like in ranking queries)
        for key, value in results.items():
            if isinstance(value, dict):
                if 'state_ranking' in value and value.get('rank_type') == 'highest':
                    if state_ranking is None:
                        state_ranking = {}
                    state_ranking.update(value['state_ranking'])
                if 'state_ranking_lowest' in value and value.get('rank_type') == 'lowest':
                    if state_ranking_lowest is None:
                        state_ranking_lowest = {}
                    state_ranking_lowest.update(value['state_ranking_lowest'])
                elif 'total' in value:
                    # This is the format: {'Rape': {'total': 1014}}
                    if crime_totals is None:
                        crime_totals = {}
                    crime_totals[key] = value['total']
        
        print(f"DEBUG: Extracted data - crime_totals: {type(crime_totals)} -> {crime_totals}")
        print(f"DEBUG: Extracted data - state_totals: {type(state_totals)} -> {state_totals}")
        print(f"DEBUG: Extracted data - year_totals: {type(year_totals)} -> {year_totals}")
        print(f"DEBUG: Extracted data - comparison_data: {type(comparison_data)} -> {comparison_data}")
        print(f"DEBUG: Extracted data - state_ranking: {type(state_ranking)} -> {state_ranking}")
        print(f"DEBUG: Extracted data - state_ranking_lowest: {type(state_ranking_lowest)} -> {state_ranking_lowest}")
        
        # Determine the best chart type based on analysis type and available data
        if analysis_type == 'statistical':
            print("DEBUG: Processing statistical analysis")
            if crime_totals:
                print("DEBUG: Creating crime_totals bar chart")
                if isinstance(crime_totals, dict):
                    crime_totals = pd.Series(crime_totals)
                fig = px.bar(
                    x=crime_totals.index,
                    y=crime_totals.values,
                    title="📊 Distribution of Crime Types",
                    labels={'x': 'Crime Type', 'y': 'Total Cases'},
                    color=crime_totals.values,
                    color_continuous_scale='plasma'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
            elif state_totals:
                print("DEBUG: Creating state_totals bar chart")
                if isinstance(state_totals, dict):
                    state_totals = pd.Series(state_totals)
                top_states = state_totals.nlargest(15)
                fig = px.bar(
                    x=top_states.index,
                    y=top_states.values,
                    title="📊 Top 15 States by Total Crime Cases",
                    labels={'x': 'State', 'y': 'Total Cases'},
                    color=top_states.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
            elif year_totals:
                print("DEBUG: Creating year_totals line chart")
                if isinstance(year_totals, dict):
                    year_totals = pd.Series(year_totals)
                fig = px.line(
                    x=year_totals.index,
                    y=year_totals.values,
                    title="📈 Crime Trends Over Time",
                    markers=True
                )
                fig.update_layout(xaxis_title="Year", yaxis_title="Total Cases", height=500)
                return fig
        
        elif analysis_type == 'ranking':
            print("DEBUG: Processing ranking analysis")
            
            # Initialize variables to hold all highest and lowest data across crimes
            all_highest_states = pd.Series(dtype=float)
            all_lowest_states = pd.Series(dtype=float)

            # Iterate through the results to find all state_ranking and state_ranking_lowest
            for crime_key, crime_data in results.items():
                if isinstance(crime_data, dict):
                    if 'state_ranking' in crime_data and isinstance(crime_data['state_ranking'], dict):
                        current_highest = pd.Series(crime_data['state_ranking'])
                        all_highest_states = pd.concat([all_highest_states, current_highest])
                    if 'state_ranking_lowest' in crime_data and isinstance(crime_data['state_ranking_lowest'], dict):
                        current_lowest = pd.Series(crime_data['state_ranking_lowest']['state_ranking'])
                        all_lowest_states = pd.concat([all_lowest_states, current_lowest])
            
            # Remove duplicates and aggregate if necessary (e.g., if a state appears in multiple crime rankings)
            all_highest_states = all_highest_states.groupby(all_highest_states.index).sum().sort_values(ascending=False)
            all_lowest_states = all_lowest_states.groupby(all_lowest_states.index).sum().sort_values(ascending=True)
            
            # Create combined visualization for highest and lowest
            if not all_highest_states.empty and not all_lowest_states.empty:
                print("DEBUG: Creating combined ranking chart for highest and lowest states")
                # Get top 5 highest and top 5 lowest
                top_5_highest = all_highest_states.nlargest(5)
                top_5_lowest = all_lowest_states.nsmallest(5)
                
                # Create a DataFrame for visualization
                combined_df = pd.DataFrame({
                    'State': top_5_highest.index.tolist() + top_5_lowest.index.tolist(),
                    'Total Cases': top_5_highest.values.tolist() + top_5_lowest.values.tolist(),
                    'Rank Type': ['Highest'] * len(top_5_highest) + ['Lowest'] * len(top_5_lowest)
                })
                
                fig = px.bar(
                    combined_df,
                    x='State',
                    y='Total Cases',
                    color='Rank Type',
                    title="🏆 Top 5 Highest & Bottom 5 Lowest States by Crime Cases",
                    labels={'Total Cases': 'Total Cases'},
                    barmode='group',
                    color_discrete_map={'Highest': '#667eea', 'Lowest': '#fed6e3'}
                )
                fig.update_layout(xaxis_tickangle=-45, height=600)
                return fig
            elif not all_highest_states.empty:
                print("DEBUG: Creating highest states ranking chart")
                fig = px.bar(
                    x=all_highest_states.index,
                    y=all_highest_states.values,
                    title="🏆 Top States by Crime Cases",
                    labels={'x': 'State', 'y': 'Total Cases'},
                    color=all_highest_states.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
            elif not all_lowest_states.empty:
                print("DEBUG: Creating lowest states ranking chart")
                fig = px.bar(
                    x=all_lowest_states.index,
                    y=all_lowest_states.values,
                    title="📉 Bottom States by Crime Cases",
                    labels={'x': 'State', 'y': 'Total Cases'},
                    color=all_lowest_states.values,
                    color_continuous_scale='cividis'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
            elif state_totals:
                print("DEBUG: Creating state_totals ranking bar chart (fallback)")
                if isinstance(state_totals, dict):
                    state_totals = pd.Series(state_totals)
                top_states = state_totals.nlargest(10)
                fig = px.bar(
                    x=top_states.index,
                    y=top_states.values,
                    title="🏆 Top 10 States by Crime Cases",
                    labels={'x': 'State', 'y': 'Total Cases'},
                    color=top_states.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
            elif crime_totals:
                print("DEBUG: Creating crime_totals ranking bar chart (fallback)")
                if isinstance(crime_totals, dict):
                    crime_totals = pd.Series(crime_totals)
                top_crimes = crime_totals.nlargest(10)
                fig = px.bar(
                    x=top_crimes.index,
                    y=top_crimes.values,
                    title="🏆 Top 10 Crime Types by Cases",
                    labels={'x': 'Crime Type', 'y': 'Total Cases'},
                    color=top_crimes.values,
                    color_continuous_scale='plasma'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
        
        elif analysis_type == 'comparison':
            print("DEBUG: Processing comparison analysis")
            if comparison_data:
                print("DEBUG: Creating comparison_data line chart")
                if isinstance(comparison_data, dict):
                    # Convert the nested structure to a proper DataFrame
                    comparison_df = pd.DataFrame(comparison_data).T
                    fig = px.line(
                        comparison_df,
                        title="📈 Crime Trends Comparison",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Year", yaxis_title="Cases", height=500)
                    return fig
            elif state_totals:
                print("DEBUG: Creating state_totals comparison bar chart")
                if isinstance(state_totals, dict):
                    state_totals = pd.Series(state_totals)
                top_states = state_totals.nlargest(8)
                fig = px.bar(
                    x=top_states.index,
                    y=top_states.values,
                    title="📊 State Comparison by Crime Cases",
                    labels={'x': 'State', 'y': 'Total Cases'},
                    color=top_states.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
            elif crime_totals:
                print("DEBUG: Creating crime_totals pie chart")
                if isinstance(crime_totals, dict):
                    crime_totals = pd.Series(crime_totals)
                fig = px.pie(
                    values=crime_totals.values,
                    names=crime_totals.index,
                    title="🥧 Distribution of Crime Types"
                )
                fig.update_layout(height=500)
                return fig
        
        elif analysis_type == 'trend':
            print("DEBUG: Processing trend analysis")
            if year_totals:
                print("DEBUG: Creating year_totals line chart")
                if isinstance(year_totals, dict):
                    year_totals = pd.Series(year_totals)
                fig = px.line(
                    x=year_totals.index,
                    y=year_totals.values,
                    title="📈 Crime Trends Over Time",
                    markers=True
                )
                fig.update_layout(xaxis_title="Year", yaxis_title="Total Cases", height=500)
                return fig
            elif comparison_data:
                print("DEBUG: Creating comparison_data trend line chart")
                if isinstance(comparison_data, dict):
                    comparison_df = pd.DataFrame(comparison_data).T
                    fig = px.line(
                        comparison_df,
                        title="📈 Crime Trends Over Time",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Year", yaxis_title="Cases", height=500)
                    return fig
        
        elif analysis_type == 'breakdown':
            print("DEBUG: Processing breakdown analysis")
            # Prioritize crime_type_breakdown if available (for specific year breakdown by crime type)
            if 'crime_type_breakdown' in results:
                print("DEBUG: Creating crime_type_breakdown bar chart")
                crime_breakdown_data = pd.Series(results['crime_type_breakdown'])
                fig = px.bar(
                    x=crime_breakdown_data.index,
                    y=crime_breakdown_data.values,
                    title=f"📊 Breakdown by Crime Type for {list(params['years'])[0] if params.get('years') else 'Selected Year'}", # More specific title
                    labels={'x': 'Crime Type', 'y': 'Total Cases'},
                    color=crime_breakdown_data.values,
                    color_continuous_scale='plasma',
                    hover_name=crime_breakdown_data.index,
                    hover_data={'y': ':,.0f', 'color': False} # Show precise value on hover
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=500,
                    coloraxis_colorbar=dict(title='Total Cases') # More specific legend title
                )
                return fig
            elif crime_totals:
                print("DEBUG: Creating crime_totals breakdown bar chart")
                if isinstance(crime_totals, dict):
                    crime_totals = pd.Series(crime_totals)
                fig = px.bar(
                    x=crime_totals.index,
                    y=crime_totals.values,
                    title="📊 Breakdown by Crime Type for 2014",
                    labels={'x': 'Crime Type', 'y': 'Total Cases'},
                    color=crime_totals.values,
                    color_continuous_scale='plasma'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
            elif year_totals:
                print("DEBUG: Creating year_totals breakdown bar chart")
                if isinstance(year_totals, dict):
                    year_totals = pd.Series(year_totals)
                fig = px.bar(
                    x=year_totals.index,
                    y=year_totals.values,
                    title="📊 Breakdown by Year",
                    labels={'x': 'Year', 'y': 'Total Cases'},
                    color=year_totals.values,
                    color_continuous_scale='plasma'
                )
                fig.update_layout(height=500)
                return fig
            elif comparison_data:
                print("DEBUG: Creating comparison_data breakdown bar chart")
                if isinstance(comparison_data, dict):
                    comparison_df = pd.DataFrame(comparison_data).T
                    fig = px.bar(
                        comparison_df,
                        title="📊 Detailed Breakdown by Year",
                        barmode='group'
                    )
                    fig.update_layout(xaxis_title="Year", yaxis_title="Cases", height=500)
                    return fig
            elif state_totals:
                print("DEBUG: Creating state_totals breakdown bar chart")
                if isinstance(state_totals, dict):
                    state_totals = pd.Series(state_totals)
                top_states = state_totals.nlargest(15)
                fig = px.bar(
                    x=top_states.index,
                    y=top_states.values,
                    title="📊 State Breakdown by Crime Cases",
                    labels={'x': 'State', 'y': 'Total Cases'},
                    color=top_states.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
        
        elif analysis_type == 'general':
            print("DEBUG: Processing general analysis")
            # For general analysis, try to create the most appropriate chart
            if crime_totals:
                print("DEBUG: Creating general crime_totals bar chart")
                if isinstance(crime_totals, dict):
                    crime_totals = pd.Series(crime_totals)
                fig = px.bar(
                    x=crime_totals.index,
                    y=crime_totals.values,
                    title="📊 Crime Type Distribution",
                    labels={'x': 'Crime Type', 'y': 'Total Cases'},
                    color=crime_totals.values,
                    color_continuous_scale='plasma'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
            elif state_totals:
                print("DEBUG: Creating general state_totals bar chart")
                if isinstance(state_totals, dict):
                    state_totals = pd.Series(state_totals)
                top_states = state_totals.nlargest(12)
                fig = px.bar(
                    x=top_states.index,
                    y=top_states.values,
                    title="📊 Top 12 States by Crime Cases",
                    labels={'x': 'State', 'y': 'Total Cases'},
                    color=top_states.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
            elif year_totals:
                print("DEBUG: Creating general year_totals line chart")
                if isinstance(year_totals, dict):
                    year_totals = pd.Series(year_totals)
                fig = px.line(
                    x=year_totals.index,
                    y=year_totals.values,
                    title="📈 Total Crime Cases Over Time",
                    markers=True
                )
                fig.update_layout(xaxis_title="Year", yaxis_title="Total Cases", height=500)
                return fig
        
        # Final comprehensive fallback - ensure every query gets a visualization
        print("DEBUG: Creating final comprehensive fallback chart")
        if crime_columns:
            crime = crime_columns[0]
            # Create a comprehensive overview chart
            if 'Year' in df.columns and 'STATE/UT' in df.columns:
                # Create a multi-faceted chart
                yearly_data = df.groupby('Year')[crime].sum()
                state_data = df.groupby('STATE/UT')[crime].sum().nlargest(10)
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=(f'{crime} Over Time', f'Top 10 States by {crime}'),
                    specs=[[{"type": "line"}, {"type": "bar"}]]
                )
                
                fig.add_trace(
                    go.Scatter(x=list(yearly_data.index), y=list(yearly_data.values), name="Yearly Trend"),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=list(state_data.index), y=list(state_data.values), name="Top States"),
                    row=1, col=2
                )
                
                fig.update_layout(height=500, title_text=f"📊 {crime} Analysis Overview")
                return fig
            elif 'Year' in df.columns:
                yearly_data = df.groupby('Year')[crime].sum()
                fig = px.line(
                    x=yearly_data.index,
                    y=yearly_data.values,
                    title=f"📈 {crime} Cases Over Time",
                    markers=True
                )
                fig.update_layout(xaxis_title="Year", yaxis_title="Cases", height=500)
                return fig
            elif 'STATE/UT' in df.columns:
                state_data = df.groupby('STATE/UT')[crime].sum().nlargest(10)
                fig = px.bar(
                    x=state_data.index,
                    y=state_data.values,
                    title=f"📊 Top 10 States by {crime} Cases",
                    labels={'x': 'State', 'y': 'Cases'},
                    color=state_data.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                return fig
        
        print("DEBUG: No chart created - no suitable data found")
        return None
        
    except Exception as e:
        print(f"DEBUG: Error generating automatic chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return None 