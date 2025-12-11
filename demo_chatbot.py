#!/usr/bin/env python
"""
Interactive demo of the British Airways Flight Insights Assistant.

This script demonstrates:
1. Model selection (Gemini, HuggingFace, Ollama)
2. Intent selection from available query templates
3. Entity parameter input
4. Multi-turn conversation
5. Multi-model comparison

Usage:
    python M3/demo_chatbot.py
"""

import os
import sys

# Add M3 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'M3'))

from llm_layer import run_chatbot

if __name__ == "__main__":
    print("\n")
    try:
        run_chatbot()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
