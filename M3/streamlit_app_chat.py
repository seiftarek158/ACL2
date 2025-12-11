"""
British Airways Graph-RAG System - Streamlit Chat UI

A chat-style interface demonstrating the Graph-RAG system with:
- Claude-like chat interface
- Chat history
- Settings in sidebar
- Cypher query transparency
"""

import streamlit as st
import os
import json
import warnings
from typing import Dict, Any

# Fix protobuf compatibility issue
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the LLM layer components
from llm_layer import (
    AirlineInsightsAssistant,
    GeminiProvider,
    HuggingFaceProvider,
    RetrievalResult,
    LLMResponse
)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="BA Flight Insights Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS Styling
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
        padding: 1rem 0;
        border-bottom: 2px solid #3B82F6;
        margin-bottom: 1rem;
    }

    .chat-message-user {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    .chat-message-assistant {
        background-color: #E8F4F8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
    }

    .cypher-query {
        background-color: #1E1E1E;
        color: #D4D4D4;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-family: 'Courier New', monospace;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State Initialization
# ============================================================================

if 'assistant' not in st.session_state:
    st.session_state.assistant = None
    st.session_state.initialized = False
    st.session_state.chat_history = []
    st.session_state.available_providers = {}
    st.session_state.selected_model = "Gemini 2.5 Flash"
    st.session_state.retrieval_method = "Both (Hybrid)"

# ============================================================================
# Helper Functions
# ============================================================================

def initialize_assistant() -> AirlineInsightsAssistant:
    """Initialize the assistant with default Gemini provider."""
    try:
        with st.spinner("Initializing Graph-RAG system..."):
            provider = GeminiProvider(model="gemini-2.5-flash")
            assistant = AirlineInsightsAssistant(providers=[provider])

            # Cache all available providers
            st.session_state.available_providers = {
                "Gemini 2.5 Flash": GeminiProvider(model="gemini-2.5-flash"),
                "HuggingFace Gemma 2B": HuggingFaceProvider(model="gemma"),
                "HuggingFace Qwen 7B": HuggingFaceProvider(model="qwen")
            }

            return assistant
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None


def get_provider(model_name: str):
    """Get or initialize a provider for the given model name."""
    if model_name not in st.session_state.available_providers:
        try:
            if model_name == "Gemini 2.5 Flash":
                st.session_state.available_providers[model_name] = GeminiProvider(model="gemini-2.5-flash")
            elif model_name == "HuggingFace Gemma 2B":
                st.session_state.available_providers[model_name] = HuggingFaceProvider(model="gemma")
            elif model_name == "HuggingFace Qwen 7B":
                st.session_state.available_providers[model_name] = HuggingFaceProvider(model="qwen")
        except Exception as e:
            st.error(f"Error initializing {model_name}: {str(e)}")
            return None

    return st.session_state.available_providers.get(model_name)


def filter_retrieval_results(retrieval_result: RetrievalResult, method: str) -> RetrievalResult:
    """Filter retrieval results based on selected method."""
    filtered_result = RetrievalResult(
        query_intent=retrieval_result.query_intent,
        entities=retrieval_result.entities
    )

    if method == "Baseline Only (Cypher Queries)":
        filtered_result.baseline_results = retrieval_result.baseline_results
        filtered_result.embedding_results = []
    elif method == "Embeddings Only (Semantic Search)":
        filtered_result.baseline_results = []
        filtered_result.embedding_results = retrieval_result.embedding_results
    else:  # Both
        filtered_result.baseline_results = retrieval_result.baseline_results
        filtered_result.embedding_results = retrieval_result.embedding_results

    return filtered_result


def extract_cypher_queries(filtered_result: RetrievalResult, assistant) -> list:
    """Extract executed Cypher queries if available."""
    cypher_queries = []

    if filtered_result.baseline_results and filtered_result.query_intent:
        try:
            query_library = assistant.retriever.query_library
            query, params = query_library.get_query(
                filtered_result.query_intent,
                filtered_result.entities or {}
            )
            # Format query with parameters
            formatted_query = f"// Intent: {filtered_result.query_intent}\n{query}\n// Parameters: {params}"
            cypher_queries.append(formatted_query)
        except:
            pass

    return cypher_queries


def display_chat_message(msg: Dict[str, Any]):
    """Display a single chat message."""
    if msg['role'] == 'user':
        st.markdown(f"""
        <div class="chat-message-user">
            <strong>👤 You:</strong><br/>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message-assistant">
            <strong>🤖 Assistant ({msg['metadata']['model']}):</strong><br/><br/>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)

        # Expandable metadata section
        with st.expander("📊 View Details (KG Context, Cypher Queries, Metadata)"):
            metadata = msg['metadata']

            # Metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Retrieval Method", metadata.get('retrieval_method', 'N/A'))
            with col2:
                st.metric("Response Time", f"{metadata.get('response_time', 0):.2f}s")
            with col3:
                st.metric("Tokens Used", metadata.get('token_count', 0))

            # Cypher Queries
            if metadata.get('cypher_queries'):
                st.markdown("### 🔍 Cypher Queries Executed")
                for i, query in enumerate(metadata['cypher_queries'], 1):
                    st.code(query, language="cypher")

            # KG Context
            kg_results = metadata.get('kg_results')
            if kg_results:
                st.markdown("### 📊 Knowledge Graph Context")

                # Baseline results
                if kg_results.baseline_results:
                    st.markdown(f"**Structured Results:** {len(kg_results.baseline_results)} records")
                    with st.expander(f"View {len(kg_results.baseline_results)} structured records"):
                        for i, result in enumerate(kg_results.baseline_results[:10], 1):
                            st.json(result)

                # Embedding results
                if kg_results.embedding_results:
                    st.markdown(f"**Semantic Results:** {len(kg_results.embedding_results)} similar journeys")
                    with st.expander(f"View {len(kg_results.embedding_results)} semantic results"):
                        for i, result in enumerate(kg_results.embedding_results[:5], 1):
                            st.markdown(f"**Journey {i}:**")
                            st.markdown(result.get("content", "")[:300] + "...")
                            if result.get("metadata"):
                                st.json(result["metadata"])


# ============================================================================
# Main Application
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">✈️ British Airways Flight Insights Assistant</div>',
                unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Settings")

        # Initialize button
        if not st.session_state.initialized:
            if st.button("🚀 Initialize System", type="primary", use_container_width=True):
                st.session_state.assistant = initialize_assistant()
                st.session_state.initialized = st.session_state.assistant is not None
                if st.session_state.initialized:
                    st.success("✅ System initialized!")
                    st.rerun()

        # Status
        if st.session_state.initialized:
            st.success("✅ Ready")
        else:
            st.warning("⚠️ Not Initialized")

        st.divider()

        # Model and Retrieval settings (only when initialized)
        if st.session_state.initialized:
            st.markdown("### 🤖 Model")
            st.session_state.selected_model = st.selectbox(
                "Select Model",
                options=[
                    "Gemini 2.5 Flash",
                    "HuggingFace Gemma 2B",
                    "HuggingFace Qwen 7B"
                ],
                key="sidebar_model",
                label_visibility="collapsed"
            )

            st.markdown("### 🔍 Retrieval")
            st.session_state.retrieval_method = st.selectbox(
                "Select Method",
                options=[
                    "Both (Hybrid)",
                    "Baseline Only (Cypher Queries)",
                    "Embeddings Only (Semantic Search)"
                ],
                key="sidebar_retrieval",
                label_visibility="collapsed"
            )

            st.divider()

            # Clear chat
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        st.divider()

        # Info
        st.markdown("### 📚 About")
        st.markdown("""
        Powered by:
        - Knowledge Graph (Neo4j)
        - Semantic Embeddings
        - LLM Reasoning
        """)

    # Main chat area
    if not st.session_state.initialized:
        st.info("👈 Initialize the system to start chatting")

        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - Which routes have the worst delays?
        - What's the food satisfaction for Millennials?
        - Show me flights with customer retention risk
        - Which fleet types have the best performance?
        """)
        return

    # Display chat history
    for msg in st.session_state.chat_history:
        display_chat_message(msg)

    # Chat input (always at bottom)
    user_question = st.chat_input(
        "Ask about flights, delays, satisfaction, routes...",
        key="chat_input"
    )

    # Process question
    if user_question:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })

        try:
            # Retrieve from KG
            with st.spinner("Retrieving from Knowledge Graph..."):
                retrieval_result = st.session_state.assistant.retriever.retrieve(
                    question=user_question,
                    intent=None,
                    entities={}
                )

                filtered_result = filter_retrieval_results(
                    retrieval_result,
                    st.session_state.retrieval_method
                )

            # Generate response
            with st.spinner(f"Generating with {st.session_state.selected_model}..."):
                context = st.session_state.assistant.prompt_builder.build_context(filtered_result)
                context_section, prompt_section = st.session_state.assistant.prompt_builder.build_prompt(
                    user_question, context
                )

                provider = get_provider(st.session_state.selected_model)
                if not provider:
                    st.error("Failed to initialize model")
                    return

                llm_response = provider.generate(prompt_section, context_section)

            # Extract Cypher queries
            cypher_queries = extract_cypher_queries(filtered_result, st.session_state.assistant)

            # Add assistant message
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': llm_response.response_text,
                'metadata': {
                    'model': st.session_state.selected_model,
                    'retrieval_method': st.session_state.retrieval_method,
                    'response_time': llm_response.response_time,
                    'token_count': llm_response.token_count,
                    'cypher_queries': cypher_queries,
                    'kg_results': filtered_result
                }
            })

            # Rerun to show new messages
            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
