import os
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
import json
import yaml
from datetime import datetime
import re
import asyncio
from typing import List, Dict, Any

# === CONFIGURATION ===
MODEL_NAME = "llama3-70b-8192"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOOLS = ["qgis", "gdal", "grass", "geopandas", "osmnx", "rasterio", "whitebox"]
EMBEDDING_BASE_PATH = "embeddings"
TOP_K = 3

# === PREMIUM UI CONFIGURATION ===
st.set_page_config(
    page_title="Geo-LLM Pro | Advanced GIS Reasoning",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === ENHANCED CUSTOM CSS ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #667eea;
        --primary-light: #764ba2;
        --secondary-color: #4facfe;
        --accent-color: #00f2fe;
        --success-color: #00d4aa;
        --warning-color: #ff8a80;
        --bg-main: #f8fafc;
        --bg-card: #ffffff;
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --text-primary: #2d3748;
        --text-secondary: #718096;
        --text-light: #a0aec0;
        --border-color: #e2e8f0;
        --border-light: #f7fafc;
        --shadow-soft: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-medium: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-large: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp {
        background: var(--bg-main);
    }
    
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    /* Header Section */
    .hero-section {
        background: var(--bg-gradient);
        color: white;
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: var(--shadow-large);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Card Components */
    .card {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--border-color);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-light);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        margin-left: 0.5rem;
    }
    
    .card-icon {
        font-size: 1.5rem;
        color: var(--primary-color);
    }
    
    /* Input Section */
    .input-section {
        background: var(--bg-card);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: var(--shadow-soft);
        margin-bottom: 3rem;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Reasoning Steps */
    .reasoning-container {
        background: var(--bg-card);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: var(--shadow-soft);
        margin-bottom: 3rem;
    }
    
    .step-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .step-card {
        background: var(--bg-main);
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .step-card.active {
        border-color: var(--primary-color);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        transform: scale(1.02);
    }
    
    .step-card.completed {
        border-color: var(--success-color);
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.05) 0%, rgba(0, 242, 254, 0.05) 100%);
    }
    
    .step-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .step-number {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--primary-color);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 1rem;
        transition: all 0.3s ease;
    }
    
    .step-number.active {
        background: var(--primary-light);
        animation: pulse 2s infinite;
    }
    
    .step-number.completed {
        background: var(--success-color);
    }
    
    .step-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .step-description {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    .step-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-pending {
        background: rgba(160, 174, 192, 0.1);
        color: var(--text-light);
    }
    
    .status-processing {
        background: rgba(255, 138, 128, 0.1);
        color: var(--warning-color);
        animation: statusPulse 1.5s infinite;
    }
    
    .status-completed {
        background: rgba(0, 212, 170, 0.1);
        color: var(--success-color);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes statusPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Results Section */
    .results-container {
        background: var(--bg-card);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: var(--shadow-soft);
        margin-bottom: 3rem;
    }
    
    .live-output {
        background: #1a202c;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        color: #e2e8f0;
        font-family: 'Monaco', 'Menlo', monospace !important;
        font-size: 0.9rem;
        line-height: 1.6;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #2d3748;
    }
    
    /* Metrics Section */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .metric-card {
        background: var(--bg-gradient);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: var(--shadow-soft);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-medium);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: var(--bg-gradient) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-soft) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-medium) !important;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-soft);
    }
    
    /* Documentation Section */
    .doc-section {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-soft);
    }
    
    .doc-chunk {
        background: var(--bg-main);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
        font-family: 'Monaco', 'Menlo', monospace !important;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    
    /* Footer */
    .footer {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--border-color);
    }
    
    .footer-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .footer-subtitle {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
        }
        
        .card, .input-section, .reasoning-container, .results-container {
            padding: 1.5rem;
        }
        
        .step-grid {
            grid-template-columns: 1fr;
        }
        
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# === ENHANCED SIDEBAR CONFIGURATION ===
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("## ⚙️ Configuration")
    
    # Model selection
    selected_model = st.selectbox(
        "🧠 Reasoning Model",
        [MODEL_NAME, "llama3-8b-8192", "mixtral-8x7b-32768"],
        help="Choose the LLM model for reasoning"
    )
    
    # Reasoning depth
    reasoning_depth = st.slider(
        "🔍 Reasoning Depth",
        min_value=1,
        max_value=5,
        value=3,
        help="Higher values provide more detailed step-by-step reasoning"
    )
    
    # Tool selection mode
    st.markdown("### 🛠️ Tool Selection")
    tool_selection_mode = st.radio(
        "Selection Mode",
        ["Auto-select by AI", "Manual selection"],
        help="Let AI choose tools automatically or select manually"
    )
    
    if tool_selection_mode == "Manual selection":
        selected_tools = st.multiselect(
            "Select GIS Tools",
            TOOLS,
            default=TOOLS,
            help="Choose which tool documentation to include"
        )
    else:
        selected_tools = TOOLS
        st.info("🤖 AI will automatically select the most relevant tools based on your query")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.slider("Max Tokens", 1024, 4096, 2048, 256)
        top_k_docs = st.slider("Top K Documents", 1, 5, TOP_K)
        show_live_steps = st.checkbox("Show Live Step Updates", True)
    
    st.markdown('</div>', unsafe_allow_html=True)

REASONING_STEPS = [
       {
           "id": "analyze_query",
           "title": "🎯 Analyzing Query",
           "description": "Breaking down the geospatial requirements and objectives"
       },
       {
           "id": "select_tools",
           "title": "🛠️ Selecting Tools",
           "description": "Choosing optimal GIS tools for the analysis"
       },
       {
           "id": "identify_data",
           "title": "📊 Identifying Data Sources",
           "description": "Determining required input data and formats"
       },
       {
           "id": "design_workflow",
           "title": "⚙️ Designing Workflow",
           "description": "Creating step-by-step processing pipeline"
       },
       {
           "id": "generate_code",
           "title": "💻 Generating Implementation",
           "description": "Producing executable workflow and code"
       },
       {
           "id": "validate_approach",
           "title": "✅ Validating Approach",
           "description": "Reviewing methodology and suggesting improvements"
       }
]

def display_reasoning_steps(current_step_id=None):
    """Display dynamic reasoning steps with enhanced UI"""
    st.markdown('''
    <div class="reasoning-container">
        <h2 class="section-title">🧠 Reasoning Process</h2>
        <div class="step-grid">
    ''', unsafe_allow_html=True)
    
    step_ids = [s['id'] for s in REASONING_STEPS]
    
    for i, step in enumerate(REASONING_STEPS):
        step_num = i + 1
        
        # Determine status
        if current_step_id is None:
            status = "pending"
            status_text = "Pending"
        elif current_step_id == "completed":
            status = "completed"
            status_text = "Completed"
        elif step['id'] == current_step_id:
            status = "active"
            status_text = "Processing"
        else:
            try:
                current_step_index = step_ids.index(current_step_id)
                if i < current_step_index:
                    status = "completed"
                    status_text = "Completed"
                else:
                    status = "pending"
                    status_text = "Pending"
            except ValueError:
                status = "pending"
                status_text = "Pending"
        
        st.markdown(f'''
        <div class="step-card {status}">
            <div class="step-header">
                <div class="step-number {status}">{step_num}</div>
                <div class="step-title">{step['title']}</div>
            </div>
            <div class="step-description">{step['description']}</div>
            <div class="step-status status-{status.replace('active', 'processing')}">{status_text}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)

# === MAIN INTERFACE ===

# Initialize reasoning step state
if 'current_step' not in st.session_state:
    st.session_state.current_step = "analyze_query"

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Hero Section
st.markdown('''
<div class="hero-section">
    <h1 class="hero-title">🛰️ Flood Risk Assessment Tool</h1>
    <p class="hero-subtitle">Comprehensive flood risk analysis for Indian cities using geospatial data and advanced modeling</p>
</div>
''', unsafe_allow_html=True)

# Input Section
st.markdown('''
<div class="input-section">
    <h2 class="section-title">💭 What geospatial challenge can I help you solve?</h2>
</div>
''', unsafe_allow_html=True)

# Example queries
example_queries = [
    "Generate a flood risk map for Pune using DEM and land use data",
    "Analyze urban heat island effect using satellite imagery and weather data",
    "Create a vegetation health index using NDVI and precipitation data",
    "Perform watershed analysis for drainage network extraction",
    "Calculate carbon sequestration potential using forest cover and biomass data"
]

selected_example = st.selectbox(
    "🎯 Try an example query:",
    [""] + example_queries,
    index=0,
    help="Select an example or write your own query below"
)

query = st.text_area(
    "📝 Your Query",
    value=selected_example,
    height=120,
    placeholder="Describe your geospatial analysis task in detail...",
    help="The more specific you are, the better the reasoning and workflow will be"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_button = st.button("🚀 Start Advanced Reasoning", use_container_width=True)

# === STEP DEFINITIONS ===
REASONING_STEPS = [
    {
        "id": "analyze_query",
        "title": "🎯 Analyzing Query",
        "description": "Breaking down the geospatial requirements and objectives"
    },
    {
        "id": "select_tools",
        "title": "🛠️ Selecting Tools",
        "description": "Choosing optimal GIS tools for the analysis"
    },
    {
        "id": "identify_data",
        "title": "📊 Identifying Data Sources",
        "description": "Determining required input data and formats"
    },
    {
        "id": "design_workflow",
        "title": "⚙️ Designing Workflow",
        "description": "Creating step-by-step processing pipeline"
    },
    {
        "id": "generate_code",
        "title": "💻 Generating Implementation",
        "description": "Producing executable workflow and code"
    },
    {
        "id": "validate_approach",
        "title": "✅ Validating Approach",
        "description": "Reviewing methodology and suggesting improvements"
    }
]

# === ENHANCED FUNCTIONS ===


def auto_select_tools(query: str) -> List[str]:
    """Use AI to automatically select the most relevant tools"""
    GROQ_API_KEY = "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"
    client = Groq(api_key=GROQ_API_KEY)
    
    tool_descriptions = {
        "qgis": "Desktop GIS software for visualization, analysis, and data management",
        "gdal": "Geospatial data abstraction library for raster and vector data",
        "grass": "Geographic analysis system for raster and vector processing",
        "geopandas": "Python library for geospatial data manipulation and analysis",
        "osmnx": "Python library for street networks and urban analytics",
        "rasterio": "Python library for reading and writing geospatial raster data",
        "whitebox": "Advanced geospatial analysis library for terrain analysis"
    }
    
    prompt = f"""
    Given this geospatial analysis query: "{query}"
    
    Available tools:
    {json.dumps(tool_descriptions, indent=2)}
    
    Select the 3-5 most relevant tools for this task. Return only a JSON array of tool names.
    Consider:
    - What type of data will be processed (raster, vector, network)
    - What analysis operations are needed
    - What visualization or output is required
    
    Example response: ["gdal", "geopandas", "rasterio"]
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        json_match = re.search(r'\[.*\]', content)
        if json_match:
            selected = json.loads(json_match.group())
            return [tool for tool in selected if tool in TOOLS]
        else:
            return ["gdal", "geopandas", "rasterio"]
            
    except Exception as e:
        st.warning(f"Auto-selection failed: {e}. Using default tools.")
        return ["gdal", "geopandas", "rasterio"]

@st.cache_resource
def get_enhanced_context(user_query, selected_tools, k=3):
    """Enhanced context retrieval with better ranking"""
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    context_chunks = []
    tool_contexts = {}
    
    for tool in selected_tools:
        try:
            vs_path = os.path.join(EMBEDDING_BASE_PATH, f"{tool}_faiss")
            db = FAISS.load_local(vs_path, embedding_model, allow_dangerous_deserialization=True)
            docs = db.similarity_search(user_query, k=k)
            context_chunks.extend(docs)
            tool_contexts[tool] = docs
        except Exception as e:
            st.warning(f"❌ Failed to load vector store for {tool}: {e}")
    
    combined_context = "\n\n".join(doc.page_content for doc in context_chunks)
    return combined_context, tool_contexts

def enhanced_query_llama_stream(query, context, selected_tools, reasoning_depth=3, temperature=0.3, max_tokens=2048):
    """Enhanced LLM call with structured reasoning"""
    GROQ_API_KEY = "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"
    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = f"""You are an expert GIS analyst providing step-by-step reasoning for geospatial analysis.

Your response must follow this EXACT structure:

## Step 1: Query Analysis
I need to analyze the query: "{query}"

The main objectives are:
- [List 2-3 key objectives]
- [What type of analysis is needed]
- [Expected outputs/deliverables]

## Step 2: Tool Selection
For this analysis, I need these tools:
- [Tool 1]: [Why this tool is needed]
- [Tool 2]: [Why this tool is needed]
- [Tool 3]: [Why this tool is needed]

Selected tools: {selected_tools}

## Step 3: Data Requirements
I need the following data:
- [Data type 1]: [Source and format]
- [Data type 2]: [Source and format]
- [Data type 3]: [Source and format]

## Step 4: Workflow Design
The processing workflow will be:
1. [Step 1 description]
2. [Step 2 description]
3. [Step 3 description]
[Continue with more steps]

## Step 5: Implementation
Here's the detailed workflow:

yaml
workflow:
  name: "GIS Analysis Workflow"
  description: "Step-by-step geospatial analysis"
  
  steps:
    - id: step_1
      name: "Data Preparation"
      tool: "gdal"
      command: "gdal_translate -of GTiff input.tif output.tif"
      description: "Convert and prepare input data"
      
    - id: step_2
      name: "Analysis"
      tool: "geopandas"
      command: "gdf.overlay(gdf1, gdf2, how='intersection')"
      description: "Perform spatial analysis"
      
    # Add more steps as needed


## Step 6: Validation & Quality Check
To ensure quality results:
- [Validation method 1]
- [Validation method 2]
- [Quality check procedures]

Remember to be specific about actual commands, file formats, and processing steps.
"""

    full_prompt = f"""
Context from GIS documentation:
{context}

User Query: {query}
Selected Tools: {selected_tools}

Provide detailed step-by-step reasoning for this geospatial analysis task.
"""

    try:
        stream = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Error in reasoning: {str(e)}"

# === MAIN EXECUTION LOGIC ===
if run_button and query.strip():
    start_time = datetime.now()
    progress_container = st.empty()
    results_container = st.empty()

    # Step 1: Analyze Query
    with progress_container.container():
        display_reasoning_steps("analyze_query")
    time.sleep(1)

    # Step 2: Select Tools
    with progress_container.container():
        display_reasoning_steps("select_tools")
    if tool_selection_mode == "Auto-select by AI":
        with st.spinner("🤖 AI is selecting optimal tools..."):
            selected_tools = auto_select_tools(query)
            st.success(f"✅ Auto-selected tools: {', '.join(selected_tools)}")
    time.sleep(1)

    # Step 3: Identify Data
    with progress_container.container():
        display_reasoning_steps("identify_data")
    with st.spinner("📚 Retrieving documentation..."):
        context, tool_contexts = get_enhanced_context(query, selected_tools, top_k_docs)
    time.sleep(1)

    # Step 4: Design Workflow
    with progress_container.container():
        display_reasoning_steps("design_workflow")
    time.sleep(1)

    # Step 5: Generate Code
    with progress_container.container():
        display_reasoning_steps("generate_code")
    st.markdown('''
    <div class="results-container">
        <h2 class="section-title">🧠 Live Reasoning Output</h2>
        <div class="live-output">
    ''', unsafe_allow_html=True)
    stream_placeholder = st.empty()
    full_output = ""
    for chunk in enhanced_query_llama_stream(
        query, context, selected_tools, reasoning_depth, temperature, max_tokens
    ):
        full_output += chunk
        stream_placeholder.markdown(full_output)
        time.sleep(0.02)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Step 6: Validate
    with progress_container.container():
        display_reasoning_steps("validate_approach")
    time.sleep(1)

    # Final step - mark all as completed
    with progress_container.container():
        display_reasoning_steps("completed")

    # Display final results
    st.markdown('''
    <div class="card">
        <div class="card-header">
            <span class="card-icon">📊</span>
            <h3 class="card-title">Analysis Complete</h3>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Documentation chunks (show actual retrieved docs)
    st.markdown('''
    <div class="doc-section">
        <h2 class="section-title">📚 Retrieved Documentation Chunks</h2>
    ''', unsafe_allow_html=True)
    for tool, docs in tool_contexts.items():
        with st.expander(f"📖 {tool.upper()} Documentation"):
            for idx, doc in enumerate(docs, 1):
                st.markdown(f"**Chunk {idx}:**")
                st.code(doc.page_content, language="markdown")
    st.markdown('</div>', unsafe_allow_html=True)

    # Display metrics
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📈 Analysis Summary</h3>', unsafe_allow_html=True)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    st.markdown(f'''
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">⏱ Processing Time</div>
            <div class="metric-value">{processing_time:.1f}s</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">🛠 Tools Used</div>
            <div class="metric-value">{len(selected_tools)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">📊 Analysis Depth</div>
            <div class="metric-value">{reasoning_depth}</div>
        </div>
        
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Download buttons for workflow and reasoning ---
    # Extract workflow from full_output (look for yaml code block or workflow section)
    import re
    workflow_text = None
    workflow_match = re.search(r'```yaml\n(.*?)\n```', full_output, re.DOTALL)
    if workflow_match:
        workflow_text = workflow_match.group(1).strip()
    else:
        # Fallback: look for 'yaml\nflood_risk_workflow:' and extract until next double newline or end
        fallback_match = re.search(r'yaml\s*(flood_risk_workflow:.*?)(?:\n\n|\Z)', full_output, re.DOTALL)
        if fallback_match:
            workflow_text = fallback_match.group(1).strip()
    if workflow_text:
        st.download_button(
            label="⬇️ Download Generated Workflow (YAML)",
            data=workflow_text,
            file_name="flood_risk_workflow.yaml",
            mime="text/yaml"
        )
    st.download_button(
        label="⬇️ Download Reasoning Steps & Workflow (TXT)",
        data=full_output,
        file_name="reasoning_steps.txt",
        mime="text/plain"
    )

    # Tool documentation
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📚 Tool Documentation</h3>', unsafe_allow_html=True)

# Footer
st.markdown('''
<div class="footer">
    <div class="footer-title">🌊 Flood Risk Assessment Tool</div>
    <p>Advanced geospatial analysis for flood risk management and disaster preparedness</p>
</div>
''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
