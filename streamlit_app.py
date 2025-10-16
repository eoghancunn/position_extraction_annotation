import streamlit as st
import pandas as pd
import json
from pathlib import Path
import re
from rapidfuzz import fuzz, process
from datetime import datetime
import base64

STOPWORDS = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                    'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
                    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'clear', 'non'])

# Set page config
st.set_page_config(page_title="Debate Position Viewer", layout="centered", initial_sidebar_state="collapsed")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("reconstructed_positions.csv")
    return df

# Load report JSON
@st.cache_data
def load_report(report_name):
    if pd.isna(report_name) or report_name == "":
        return None
    
    report_path = Path(f"reports/{report_name}.json")
    if report_path.exists():
        with open(report_path, 'r') as f:
            return json.load(f)['summary']
    return None

# Load feedback data
def load_feedback():
    feedback_file = Path("feedback_data.json")
    if feedback_file.exists():
        with open(feedback_file, 'r') as f:
            return json.load(f)
    return {}

# Save feedback data
def save_feedback(feedback_data):
    feedback_file = Path("feedback_data.json")
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=2)

# Record feedback for a specific summary
def record_feedback(row_index, summary_type, feedback_value, comment=None):
    feedback_data = load_feedback()
    
    # Create unique key for this row
    key = f"row_{row_index}"
    
    # Initialize row if it doesn't exist
    if key not in feedback_data:
        feedback_data[key] = {}
    
    # Store feedback
    feedback_data[key][summary_type] = feedback_value == "up"
    
    # Store comment if provided
    if comment is not None:
        feedback_data[key][f"{summary_type}_comment"] = comment
    
    save_feedback(feedback_data)

# Get feedback for a specific summary
def get_feedback(row_index, summary_type):
    feedback_data = load_feedback()
    key = f"row_{row_index}"
    return feedback_data.get(key, {}).get(summary_type, None)

# Get comment for a specific summary
def get_comment(row_index, summary_type):
    feedback_data = load_feedback()
    key = f"row_{row_index}"
    return feedback_data.get(key, {}).get(f"{summary_type}_comment", "")

def highlight_speaker_name(text, speaker_name, threshold=80):
    """
    Highlight occurrences of the speaker's name in the text using fuzzy matching.
    
    Args:
        text: The text to search in
        speaker_name: The speaker's full name
        threshold: Fuzzy matching threshold (0-100)
    
    Returns:
        Text with highlighted speaker names
    """
    if not text or not speaker_name:
        return text
    
    # Split speaker name into parts
    name_parts = speaker_name.split()
    
    # Find all words in the text
    words = re.findall(r'\b\w+\b', text)
    positions = []
    
    # For each word in text, check if it fuzzy matches any part of the speaker's name
    for match in re.finditer(r'\b\w+\b', text):
        word = match.group()
        # Skip stopwords
        if word.lower() not in STOPWORDS:
            # Check against full name and each name part
            names_to_check = [speaker_name] + name_parts
            
            for name in names_to_check:
                score = fuzz.ratio(word.lower(), name.lower())
                if score >= threshold:
                    positions.append((match.start(), match.end()))
                    break
    
    # Build highlighted text
    if not positions:
        return text
    
    # Sort positions by start index
    positions = sorted(set(positions))
    
    # Build result with highlights
    result = []
    last_end = 0
    
    for start, end in positions:
        # Add text before match
        result.append(text[last_end:start])
        # Add highlighted match
        result.append(f'<mark style="background-color: #FFD700; padding: 2px 4px; border-radius: 3px;">{text[start:end]}</mark>')
        last_end = end
    
    # Add remaining text
    result.append(text[last_end:])
    
    return ''.join(result)

def main():
    # st.title("Speaker Position Extraction")
    
    # Add custom CSS for centered layout width and column sizing
    st.markdown("""
        <style>
        /* Set the max width for centered layout */
        .block-container {
            max-width: 1400px !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        /* Target only the main container's direct columns */
        .main > div > div > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
            flex: 2 !important;
        }
        .main > div > div > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
            flex: 1 1 200px !important;
            min-width: 200px !important;
            max-width: 400px !important;
        }
        
        /* Make subheaders smaller */
        .main h3 {
            font-size: 1.3rem !important;
        }

        .main h2 {
            font-size: 1.4rem !important;
        }

        .main h1 {
            font-size: 1.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load the data
    df = load_data()
    
    # Filter out rows with no speaker data and reset index
    df_filtered = df[df['speaker'].notna()].reset_index(drop=True)
    
    # Create display names with intervention numbers for duplicates
    speaker_counts = {}
    display_names = []
    for speaker in df_filtered['speaker']:
        if speaker not in speaker_counts:
            speaker_counts[speaker] = 0
        speaker_counts[speaker] += 1
        # Add intervention number if speaker appears multiple times in total
        total_count = (df_filtered['speaker'] == speaker).sum()
        if total_count > 1:
            display_names.append(f"{speaker} (Intervention {speaker_counts[speaker]})")
        else:
            display_names.append(speaker)
    
    # Sidebar
    st.sidebar.header("Annotation Tool")
    
    # Codebook
    st.sidebar.subheader("Codebook")
    
    # Load and provide download for codebook
    try:
        with open("codebook.pdf", "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        
        st.sidebar.download_button(
            label="Download Codebook (PDF)",
            data=pdf_bytes,
            file_name="codebook.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except FileNotFoundError:
        st.sidebar.warning("Codebook not found")
    
    st.sidebar.markdown("---")
    
    # Progress tracking
    st.sidebar.subheader("Progress")
    feedback_data = load_feedback()
    total_entries = len(df_filtered)
    annotated_entries = len(feedback_data)
    st.sidebar.metric("Entries Annotated", f"{annotated_entries} / {total_entries}")
    if total_entries > 0:
        progress = annotated_entries / total_entries
        st.sidebar.progress(progress)
    
    st.sidebar.markdown("---")
    
    # Upload previous session
    st.sidebar.subheader("Save & Load")
    
    # Use session state to track upload status
    if 'upload_processed' not in st.session_state:
        st.session_state.upload_processed = False
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload previous annotations",
        type=['json'],
        help="Continue from where you left off",
        key='annotation_uploader'
    )
    
    if uploaded_file is not None and not st.session_state.upload_processed:
        try:
            uploaded_data = json.load(uploaded_file)
            save_feedback(uploaded_data)
            st.session_state.upload_processed = True
            st.sidebar.success("Annotations loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    elif uploaded_file is None:
        st.session_state.upload_processed = False
    
    # Download current annotations
    current_feedback = load_feedback()
    feedback_json = json.dumps(current_feedback, indent=2)
    
    st.sidebar.download_button(
        label="Download Annotations",
        data=feedback_json,
        file_name=f"feedback_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        help="Save your work and submit this file",
        use_container_width=True
    )
    
    if annotated_entries > 0:
        st.sidebar.caption("Remember to download before closing your browser!")
    
    st.sidebar.markdown("---")
    
    # Fuzzy matching threshold control
    st.sidebar.subheader("Highlighting Settings")
    fuzzy_threshold = st.sidebar.slider(
        "Name matching sensitivity",
        min_value=60,
        max_value=100,
        value=80,
        step=5,
        help="Lower values will match more variations of the name but may have false positives"
    )
    
    # Speaker selection
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    # Initialize navigation button states
    if 'nav_prev_clicked' not in st.session_state:
        st.session_state.nav_prev_clicked = False
    if 'nav_next_clicked' not in st.session_state:
        st.session_state.nav_next_clicked = False
    
    # Handle navigation button clicks FIRST, before any rendering
    if st.session_state.nav_prev_clicked:
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1
        st.session_state.nav_prev_clicked = False
    
    if st.session_state.nav_next_clicked:
        if st.session_state.current_index < len(df_filtered) - 1:
            st.session_state.current_index += 1
        st.session_state.nav_next_clicked = False
    
    # No dropdown - navigation is via Previous/Next buttons only
    
    # Get current speaker's data using the row index
    speaker_data = df_filtered.iloc[st.session_state.current_index]
    actual_speaker_name = speaker_data['speaker']
    
    # Reminder to save progress
    if annotated_entries > 0 and annotated_entries % 10 == 0:
        st.info("💡 Reminder: Don't forget to download your annotations to save your progress!", icon="💾")
    
    # Main content area - two columns
    left_col, right_col = st.columns([1, 1])
    
    with right_col:

        st.markdown("---")

        st.markdown(f"""
            <div style="height: 30px; display: flex; align-items: center; margin-bottom: 10px;">
                <h1 style="margin: 0;">{actual_speaker_name}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons that set flags for next rerun
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        with nav_col1:
            if st.button("◀ Previous", use_container_width=True, key="nav_prev_btn"):
                st.session_state.nav_prev_clicked = True
                st.rerun()
        with nav_col2:
            if st.button("Next ▶", use_container_width=True, key="nav_next_btn"):
                st.session_state.nav_next_clicked = True
                st.rerun()
        with nav_col3:
            st.markdown(f"**{st.session_state.current_index + 1} / {len(df_filtered)}**")
        
        st.markdown("---")
        
        # Scrollable container for speaker info
        with st.container():
            # Issue Summary
            col_header, col_thumb_up, col_thumb_down = st.columns([6, 1, 1])
            with col_header:
                st.subheader("Extracted Issue")
            current_feedback = get_feedback(st.session_state.current_index, "issue")
            with col_thumb_up:
                if st.button("👍", key="issue_up", use_container_width=True, 
                           type="primary" if current_feedback is True else "secondary"):
                    record_feedback(st.session_state.current_index, "issue", "up")
                    st.rerun()
            with col_thumb_down:
                if st.button("👎", key="issue_down", use_container_width=True,
                           type="primary" if current_feedback is False else "secondary"):
                    record_feedback(st.session_state.current_index, "issue", "down")
                    st.rerun()
            
            if pd.notna(speaker_data['issueSum']) and speaker_data['issueSum'] != "":
                st.write(speaker_data['issueSum'])
            else:
                st.info("No issue extracted")
            
            # Comment box for issue
            issue_comment = st.text_area(
                "Comments/Notes:",
                value=get_comment(st.session_state.current_index, "issue"),
                key="issue_comment",
                height=80,
                placeholder="Add any notes or comments about this extracted issue..."
            )
            if issue_comment != get_comment(st.session_state.current_index, "issue"):
                record_feedback(st.session_state.current_index, "issue", 
                              "up" if current_feedback is True else "down" if current_feedback is False else "up",
                              comment=issue_comment)
            st.markdown("---")
            # Position Summary
            col_header, col_thumb_up, col_thumb_down = st.columns([6, 1, 1])
            with col_header:
                st.subheader("Extracted Position")
            current_feedback = get_feedback(st.session_state.current_index, "position")
            with col_thumb_up:
                if st.button("👍", key="position_up", use_container_width=True,
                           type="primary" if current_feedback is True else "secondary"):
                    record_feedback(st.session_state.current_index, "position", "up")
                    st.rerun()
            with col_thumb_down:
                if st.button("👎", key="position_down", use_container_width=True,
                           type="primary" if current_feedback is False else "secondary"):
                    record_feedback(st.session_state.current_index, "position", "down")
                    st.rerun()
            if pd.notna(speaker_data['positionSum']) and speaker_data['positionSum'] != "":
                st.write(speaker_data['positionSum'])
            else:
                st.info("No position extracted")
            
            # Comment box for position
            position_comment = st.text_area(
                "Comments/Notes:",
                value=get_comment(st.session_state.current_index, "position"),
                key="position_comment",
                height=80,
                placeholder="Add any notes or comments about this extracted position..."
            )
            if position_comment != get_comment(st.session_state.current_index, "position"):
                record_feedback(st.session_state.current_index, "position", 
                              "up" if current_feedback is True else "down" if current_feedback is False else "up",
                              comment=position_comment)
            
            st.markdown("---")
            # Argument Summary
            col_header, col_thumb_up, col_thumb_down = st.columns([6, 1, 1])
            with col_header:
                st.subheader("Extracted Argument")
            current_feedback = get_feedback(st.session_state.current_index, "argument")
            with col_thumb_up:
                if st.button("👍", key="argument_up", use_container_width=True,
                           type="primary" if current_feedback is True else "secondary"):
                    record_feedback(st.session_state.current_index, "argument", "up")
                    st.rerun()
            with col_thumb_down:
                if st.button("👎", key="argument_down", use_container_width=True,
                           type="primary" if current_feedback is False else "secondary"):
                    record_feedback(st.session_state.current_index, "argument", "down")
                    st.rerun()
            
            if pd.notna(speaker_data['argSum']) and speaker_data['argSum'] != "":
                st.write(speaker_data['argSum'])
            else:
                st.info("No argument extracted")
            
            # Comment box for argument
            argument_comment = st.text_area(
                "Comments/Notes:",
                value=get_comment(st.session_state.current_index, "argument"),
                key="argument_comment",
                height=80,
                placeholder="Add any notes or comments about this extracted argument..."
            )
            if argument_comment != get_comment(st.session_state.current_index, "argument"):
                record_feedback(st.session_state.current_index, "argument", 
                              "up" if current_feedback is True else "down" if current_feedback is False else "up",
                              comment=argument_comment)
            
            st.markdown("---")
            # Proposal Summary
            col_header, col_thumb_up, col_thumb_down = st.columns([6, 1, 1])
            with col_header:
                st.subheader("Extracted Proposal")
            current_feedback = get_feedback(st.session_state.current_index, "proposal")
            with col_thumb_up:
                if st.button("👍", key="proposal_up", use_container_width=True,
                           type="primary" if current_feedback is True else "secondary"):
                    record_feedback(st.session_state.current_index, "proposal", "up")
                    st.rerun()
            with col_thumb_down:
                if st.button("👎", key="proposal_down", use_container_width=True,
                           type="primary" if current_feedback is False else "secondary"):
                    record_feedback(st.session_state.current_index, "proposal", "down")
                    st.rerun()
            
            if pd.notna(speaker_data['propSum']) and speaker_data['propSum'] != "":
                st.write(speaker_data['propSum'])
            else:
                st.info("No proposal extracted")
            
            # Comment box for proposal
            proposal_comment = st.text_area(
                "Comments/Notes:",
                value=get_comment(st.session_state.current_index, "proposal"),
                key="proposal_comment",
                height=80,
                placeholder="Add any notes or comments about this extracted proposal..."
            )
            if proposal_comment != get_comment(st.session_state.current_index, "proposal"):
                record_feedback(st.session_state.current_index, "proposal", 
                              "up" if current_feedback is True else "down" if current_feedback is False else "up",
                              comment=proposal_comment)
    
    with left_col:
        # st.header("📄 Report Details")
        report_name = speaker_data['report']
        # st.markdown('<div style="height: 180px;"></div>', unsafe_allow_html=True)
        
        # Scrollable container for report
        with st.container(height=850):
            if pd.notna(report_name) and report_name != "":
                # Load and display report
                report_data = load_report(report_name)
                
                if report_data:
                    # Highlight speaker name in the report
                    highlighted_report = highlight_speaker_name(
                        report_data, 
                        actual_speaker_name,
                        threshold=fuzzy_threshold
                    )
                    # Display report content with highlighted names
                    st.markdown(highlighted_report, unsafe_allow_html=True)
                else:
                    st.warning("Report file not found or could not be loaded")
            else:
                st.info("No report associated with this speaker")
    

if __name__ == "__main__":
    main()
