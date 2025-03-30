import streamlit as st
from modules.agno_agent.option_agent import generate_options
from modules.agno_agent.enhance_agent import enhance_prompt

# Initial options with emojis
INITIAL_OPTIONS = {
    "👤 Human": "Human",
    "🐕 Animal": "Animal",
    "📦 Object/Item": "Object/Item",
    "🌅 Scenery/Environment": "Scenery/Environment"
}

def initialize_session_state():
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'initial'
    if 'available_options' not in st.session_state:
        st.session_state.available_options = INITIAL_OPTIONS.copy()
    if 'selected_keywords' not in st.session_state:
        st.session_state.selected_keywords = []
    if 'generated_options' not in st.session_state:
        st.session_state.generated_options = []
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = {}
    if 'enhanced_prompt' not in st.session_state:
        st.session_state.enhanced_prompt = ""
    if 'enhancement_explanation' not in st.session_state:
        st.session_state.enhancement_explanation = ""
    if 'roll_count' not in st.session_state:
        st.session_state.roll_count = 0

def enhance_current_prompt():
    """Generate an enhanced prompt from selected keywords using RAG and LLM"""
    if st.session_state.selected_keywords:
        enhanced, debug_info = enhance_prompt(st.session_state.selected_keywords)
        st.session_state.enhanced_prompt = enhanced
        st.session_state.enhancement_explanation = debug_info.get("explanation", "")
        st.session_state.debug_info.update({"enhancement": debug_info})
        st.rerun()

def reroll_options():
    if st.session_state.selected_keywords:
        new_options, debug_info = generate_options(st.session_state.selected_keywords)
        if new_options:
            st.session_state.generated_options = {
                opt: opt.split(' ', 1)[1] for opt in new_options
            }
            st.session_state.debug_info = debug_info
            # Increment roll count to make keys unique
            st.session_state.roll_count += 1
            st.rerun()

def select_option(option: str, display_name: str):
    if st.session_state.current_step == 'initial':
        # Clear initial options after first selection
        st.session_state.available_options = {}
    
    # Add the selected option to keywords
    st.session_state.selected_keywords.append(option)
    
    # Generate new options based on selected keywords
    new_options, debug_info = generate_options(st.session_state.selected_keywords)
    if new_options:
        st.session_state.generated_options = {
            opt: opt.split(' ', 1)[1] for opt in new_options
        }
        st.session_state.debug_info = debug_info
    
    st.session_state.current_step = 'refining'
    # Reset roll count for new selection
    st.session_state.roll_count = 0
    st.rerun()

def run():
    st.title("Refiner")
    initialize_session_state()
    
    # Show selection path and enhance button
    if st.session_state.selected_keywords:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write("Your selection path:")
            path = " → ".join(st.session_state.selected_keywords)
            st.info(path)
            st.write("Selected options:")
            st.success(", ".join(st.session_state.selected_keywords))
        with col2:
            if st.button("✨ Enhance", key="enhance_prompt"):
                enhance_current_prompt()
        
        # Show enhanced prompt and explanation if available
        if st.session_state.enhanced_prompt:
            st.text_area("Enhanced Prompt", value=st.session_state.enhanced_prompt, height=100, key="enhanced_prompt_area")
            if st.session_state.enhancement_explanation:
                with st.expander("💡 Enhancement Explanation"):
                    st.write(st.session_state.enhancement_explanation)
    
    # Show available options
    if st.session_state.current_step == 'initial':
        st.subheader("Choose a category:")
        cols = st.columns(2)
        for idx, (display_name, option) in enumerate(INITIAL_OPTIONS.items()):
            with cols[idx % 2]:
                if st.button(display_name, key=f"initial_{option}"):
                    select_option(option, display_name)
    
    # Show generated options
    if st.session_state.current_step == 'refining' and st.session_state.generated_options:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("Choose to refine further:")
        with col2:
            if st.button("🎲 Reroll", key=f"reroll_{st.session_state.roll_count}"):
                reroll_options()
                
        cols = st.columns(2)
        for idx, (display_name, option) in enumerate(st.session_state.generated_options.items()):
            with cols[idx % 2]:
                # Add roll count to make keys unique
                if st.button(display_name, key=f"gen_{option}_{st.session_state.roll_count}"):
                    select_option(option, display_name)
    
    # Add reset button and debug expander
    if st.session_state.selected_keywords:
        st.divider()
        col1, col2 = st.columns([4, 1])
        with col1:
            with st.expander("🔍 Debug Information"):
                st.json(st.session_state.debug_info)
        with col2:
            if st.button("🔄 Start Over"):
                st.session_state.clear()
                st.rerun()
