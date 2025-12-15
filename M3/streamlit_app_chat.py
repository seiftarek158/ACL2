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
import time
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
        background-color: #2D3748;
        color: #E2E8F0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    .chat-message-assistant {
        background-color: #1A365D;
        color: #E2E8F0;
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
    st.session_state.embedding_model = "sentence-transformers/all-mpnet-base-v2"
    st.session_state.needs_reinit = False

# ============================================================================
# Helper Functions
# ============================================================================

def initialize_assistant(embedding_model: str = None) -> AirlineInsightsAssistant:
    """Initialize the assistant with specified embedding model."""
    try:
        # Use session state embedding model if not specified
        if embedding_model is None:
            embedding_model = st.session_state.embedding_model

        model_display = embedding_model.split('/')[-1]
        with st.spinner(f"Initializing Graph-RAG system with {model_display}..."):
            provider = GeminiProvider(model="gemini-2.5-flash")
            assistant = AirlineInsightsAssistant(
                providers=[provider],
                embedding_model=embedding_model
            )

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


def map_retrieval_mode(ui_mode: str) -> str:
    """Map UI dropdown values to internal retrieval mode strings."""
    mapping = {
        "Both (Hybrid)": "hybrid",
        "Baseline Only (Cypher Queries)": "baseline_only",
        "Embeddings Only (Semantic Search)": "embeddings_only"
    }
    return mapping.get(ui_mode, "hybrid")


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
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Retrieval Method", metadata.get('retrieval_method', 'N/A'))
            with col2:
                # Hide embedding model when using "Baseline Only (Cypher Queries)"
                retrieval_method = metadata.get('retrieval_method', 'N/A')
                if retrieval_method == "Baseline Only (Cypher Queries)":
                    embedding_display = "N/A"
                else:
                    embedding_display = metadata.get('embedding_model', 'N/A')
                    if embedding_display != 'N/A':
                        embedding_display = embedding_display.split('/')[-1][:15]
                st.metric("Embedding Model", embedding_display)
            with col3:
                st.metric("Response Time", f"{metadata.get('response_time', 0):.2f}s")
            with col4:
                st.metric("Tokens Used", metadata.get('token_count', 0))

            # Cypher Queries
            if metadata.get('cypher_queries'):
                st.markdown("### 🔍 Cypher Queries Executed")
                for i, query in enumerate(metadata['cypher_queries'], 1):
                    st.code(query, language="cypher")

            # Thinking Steps
            thinking_steps = metadata.get('thinking_steps')
            if thinking_steps:
                st.markdown("### 🧠 Thinking Process")
                thinking_text = "\n".join(thinking_steps)
                st.markdown(f"""
                <div style="color: #6B7280; font-style: italic; opacity: 0.8; font-family: monospace; white-space: pre-wrap; background-color: #F9FAFB; padding: 10px; border-radius: 5px;">
                {thinking_text}
                </div>
                """, unsafe_allow_html=True)

            # KG Context
            kg_results = metadata.get('kg_results')
            if kg_results:
                st.markdown("### 📊 Knowledge Graph Context")

                # Baseline results - use tabs instead of nested expanders
                if kg_results.baseline_results:
                    st.markdown(f"**Structured Results:** {len(kg_results.baseline_results)} records")
                    for i, result in enumerate(kg_results.baseline_results[:10], 1):
                        st.json(result)
                    if len(kg_results.baseline_results) > 10:
                        st.caption(f"... and {len(kg_results.baseline_results) - 10} more records")

                # Embedding results
                if kg_results.embedding_results:
                    st.markdown(f"**Semantic Results:** {len(kg_results.embedding_results)} similar journeys")
                    for i, result in enumerate(kg_results.embedding_results[:5], 1):
                        st.markdown(f"**Journey {i}:**")
                        st.markdown(result.get("content", "")[:300] + "...")
                        if result.get("metadata"):
                            st.json(result["metadata"])


def display_thinking_steps(thinking_steps: list):
    """Display thinking steps with animated styling."""
    # Colors for shifting effect
    colors = ["#6B7280", "#4B5563", "#374151", "#1F2937"]
    
    placeholder = st.empty()
    current_text = ""
    
    for i, step in enumerate(thinking_steps):
        current_text += step + "\n"
        color = colors[i % len(colors)]
        
        placeholder.markdown(f"""
        <div style="color: {color}; font-style: italic; opacity: 0.7; font-family: monospace; white-space: pre-wrap;">
        {current_text}
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(0.3)  # Delay for animation effect
    
    # Final display with consistent styling
    placeholder.markdown(f"""
    <div style="color: #6B7280; font-style: italic; opacity: 0.6; font-family: monospace; white-space: pre-wrap; border-left: 2px solid #3B82F6; padding-left: 10px;">
    {current_text}
    </div>
    """, unsafe_allow_html=True)


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

        # Embedding model selection (before initialization)
        st.markdown("### 🧠 Embedding Model")
        new_embedding_model = st.selectbox(
            "Select Embedding Model",
            options=[
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2"
            ],
            index=0 if st.session_state.embedding_model == "sentence-transformers/all-mpnet-base-v2" else 1,
            help="all-mpnet-base-v2: More accurate but slower\nall-MiniLM-L6-v2: Faster and lighter",
            key="sidebar_embedding",
            label_visibility="collapsed"
        )

        # Check if embedding model changed
        if new_embedding_model != st.session_state.embedding_model:
            st.session_state.embedding_model = new_embedding_model
            if st.session_state.initialized:
                st.session_state.needs_reinit = True
                st.warning("⚠️ Embedding model changed. Please reinitialize the system.")

        st.divider()

        # Initialize/Reinitialize button
        if not st.session_state.initialized or st.session_state.needs_reinit:
            button_text = "🔄 Reinitialize System" if st.session_state.needs_reinit else "🚀 Initialize System"
            if st.button(button_text, type="primary", use_container_width=True):
                st.session_state.assistant = initialize_assistant(st.session_state.embedding_model)
                st.session_state.initialized = st.session_state.assistant is not None
                st.session_state.needs_reinit = False
                if st.session_state.initialized:
                    st.success("✅ System initialized!")
                    st.rerun()

        # Status
        if st.session_state.initialized and not st.session_state.needs_reinit:
            st.success("✅ Ready")
            model_display = st.session_state.embedding_model.split('/')[-1]
            st.caption(f"Using: {model_display}")
        elif st.session_state.needs_reinit:
            st.warning("⚠️ Needs Reinitialization")
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
            # Map UI mode to internal mode
            internal_mode = map_retrieval_mode(st.session_state.retrieval_method)
            
            # Generate response using the LLM layer with specified retrieval mode
            with st.spinner(f"Processing with {st.session_state.selected_model} ({st.session_state.retrieval_method})..."):
                provider = get_provider(st.session_state.selected_model)
                if not provider:
                    st.error("Failed to initialize model")
                    return
                
                # Temporarily set the assistant's providers to only the selected one
                original_providers = st.session_state.assistant.providers
                st.session_state.assistant.providers = [provider]
                
                try:
                    # Use the assistant's answer_question method with retrieval mode
                    llm_response = st.session_state.assistant.answer_question(
                        user_question, 
                        provider_index=0,
                        retrieval_mode=internal_mode
                    )
                finally:
                    # Restore original providers
                    st.session_state.assistant.providers = original_providers
                
                # For metadata, we need to reconstruct what was retrieved
                # Get the retrieval result by calling the internal logic
                intent, entities, query_results, _ = None, {}, [], []
                if internal_mode in ["hybrid", "baseline_only"]:
                    intent, entities, query_results, _ = st.session_state.assistant.classifier.classify_and_execute(user_question)
                
                retrieval_result = RetrievalResult(
                    baseline_results=query_results if internal_mode in ["hybrid", "baseline_only"] else [],
                    query_intent=intent or "general",
                    entities=entities
                )
                
                if internal_mode in ["hybrid", "embeddings_only"]:
                    if st.session_state.assistant.retriever.retriever:
                        try:
                            docs = st.session_state.assistant.retriever.retriever.invoke(user_question)
                            retrieval_result.embedding_results = [
                                {"content": doc.page_content, "metadata": doc.metadata}
                                for doc in docs[:5]
                            ]
                        except:
                            pass

            # Extract Cypher queries
            cypher_queries = extract_cypher_queries(retrieval_result, st.session_state.assistant)

            # Display thinking steps
            thinking_steps = llm_response.metadata.get('thinking_steps', [])
            if thinking_steps:
                display_thinking_steps(thinking_steps)

            # Add assistant message
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': llm_response.response_text,
                'metadata': {
                    'model': st.session_state.selected_model,
                    'retrieval_method': st.session_state.retrieval_method,
                    'embedding_model': st.session_state.embedding_model,
                    'response_time': llm_response.response_time,
                    'token_count': llm_response.token_count,
                    'cypher_queries': cypher_queries,
                    'kg_results': retrieval_result,
                    'intent': intent,
                    'entities': entities,
                    'thinking_steps': thinking_steps
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
