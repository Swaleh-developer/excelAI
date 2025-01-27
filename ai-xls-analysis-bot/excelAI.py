import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

# Local model configuration
@st.cache_resource
def load_llm():
    """Load local LLM model"""
    try:
        return ChatOllama(model="deepseek-r1:1.5B")  # Use the correct model name
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def analyze_excel(file):
    """Process Excel file and return analysis"""
    df = pd.read_excel(file)
    return df

def generate_local_summary(prompt, llm):
    """Generate summary using local LLM"""
    try:
        # Format the prompt as a list of messages
        messages = [HumanMessage(content=prompt)]
        
        # Generate response using the LLM
        response = llm(messages)
        return response.content.strip()  # Extract the content from the response
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def visualize_data(df, numeric_cols, selected_columns):
    """Generate automated visualizations based on selected columns"""
    st.subheader("Data Visualizations")
    
    if numeric_cols:
        # Combined Bar Chart for selected numeric columns
        st.write("### Combined Bar Chart")
        selected_numeric = st.multiselect(
            "Select numeric columns for the combined bar chart",
            numeric_cols,
            default=numeric_cols[:2]  # Default to first two numeric columns
        )
        
        if selected_numeric:
            # Melt the dataframe for seaborn barplot
            melted_df = df[selected_numeric].melt(var_name="Column", value_name="Value")
            
            # Create the combined bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=melted_df, x="Column", y="Value", ax=ax, errorbar=None)
            ax.set_title("Combined Bar Chart")
            ax.set_xlabel("Columns")
            ax.set_ylabel("Values")
            st.pyplot(fig)
        else:
            st.warning("Select at least one numeric column for the combined bar chart.")
    
    if len(numeric_cols) > 1:
        # Correlation heatmap for selected numeric columns
        st.write("### Correlation Heatmap")
        selected_numeric = st.multiselect(
            "Select numeric columns for correlation heatmap",
            numeric_cols,
            default=numeric_cols[:2]  # Default to first two numeric columns
        )
        if len(selected_numeric) > 1:
            corr = df[selected_numeric].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Select at least two numeric columns for correlation heatmap.")

    # Scatter plot for two numeric columns
    if len(numeric_cols) > 1:
        st.write("### Scatter Plot")
        x_col = st.selectbox("Select X-axis column", numeric_cols)
        y_col = st.selectbox("Select Y-axis column", numeric_cols)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        st.pyplot(fig)

def main():
    st.title("AI Excel Analysis Bot")
    st.write("Upload an Excel file for analysis and insights")

    # Load LLM
    llm = load_llm()
    if not llm:
        st.stop()

    # Upload Excel file
    uploaded_file = st.file_uploader("Choose Excel file", type=["xlsx"])
    
    if uploaded_file:
        df = analyze_excel(uploaded_file)
        
        # Show raw data
        st.subheader("Raw Data Preview")
        st.dataframe(df)

        # Allow user to select columns and rows
        st.subheader("Select Data for Analysis")
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to analyze",
            all_columns,
            default=all_columns  # Default to all columns
        )
        
        # Allow user to filter rows
        st.write("### Filter Rows")
        row_start = st.number_input("Start row", min_value=0, max_value=len(df)-1, value=0)
        row_end = st.number_input("End row", min_value=row_start+1, max_value=len(df), value=len(df))
        
        # Filter data based on user selection
        filtered_df = df[selected_columns].iloc[row_start:row_end]
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()

        # Show filtered data
        st.subheader("Filtered Data Preview")
        st.dataframe(filtered_df)

        # Basic file info
        st.subheader("Basic File Information")
        st.write(f"Shape: {filtered_df.shape} (Rows x Columns)")
        st.write(f"Columns: {', '.join(filtered_df.columns)}")
        st.write(f"Numeric columns: {', '.join(numeric_cols)}")
        
        # Generate LLM summary
        st.subheader("AI Analysis")
        prompt = f"""
        Analyze this Excel file metadata and provide business insights:
        - Shape: {filtered_df.shape}
        - Columns: {filtered_df.columns.tolist()}
        - Numeric columns: {numeric_cols}
        - Descriptive statistics: {filtered_df.describe().to_dict()}
        - Missing values: {filtered_df.isnull().sum().to_dict()}
        
        Provide concise bullet points highlighting key findings and recommendations.
        """
        
        with st.spinner("Generating insights with local LLM..."):
            analysis = generate_local_summary(prompt, llm)
        st.write(analysis)
        
        # Automated visualizations
        visualize_data(filtered_df, numeric_cols, selected_columns)
        
        # User query input
        st.subheader("Query the Data")
        user_query = st.text_input("Enter your query about the data:")
        
        if user_query:
            query_prompt = f"""
            The user has provided the following query about the data:
            {user_query}
            
            Here is the data summary:
            - Shape: {filtered_df.shape}
            - Columns: {filtered_df.columns.tolist()}
            - Numeric columns: {numeric_cols}
            - Descriptive statistics: {filtered_df.describe().to_dict()}
            - Missing values: {filtered_df.isnull().sum().to_dict()}
            
            Please provide a detailed response to the user's query.
            """
            
            with st.spinner("Processing your query..."):
                query_response = generate_local_summary(query_prompt, llm)
            st.write(query_response)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### System Information")
        st.sidebar.markdown("**Author:** Swaleh Cosmas")
        st.sidebar.markdown("**Copyright © 2025**")
        st.sidebar.markdown("All rights reserved.")

if __name__ == "__main__":
    main()
