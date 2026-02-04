
import streamlit as st
import pandas as pd
import io
from sentiment_analyzer import SentimentAnalyzer
from data_processor import DataProcessor
from visualization import SentimentVisualizer


@st.cache_resource
def get_analyzer():
    return SentimentAnalyzer()

@st.cache_resource
def get_processor():
    return DataProcessor()

@st.cache_resource
def get_visualizer():
    return SentimentVisualizer()

def main():
    st.set_page_config(
        page_title="E-consultation Sentiment Analysis",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 E-consultation Sentiment Analysis Tool")
    st.markdown("Analyze sentiment of comments received through E-consultation modules")

    analyzer = get_analyzer()
    processor = get_processor()
    visualizer = get_visualizer()

    # Sidebar configuration
    st.sidebar.header("Configuration")
    analysis_method = st.sidebar.selectbox("Sentiment Analysis Method", ["TextBlob", "VADER"])
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.02 if analysis_method=="VADER" else 0.1, 0.01
    )

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Input Comments", "📈 Analysis Results", "📊 Export Data"])

    # ---------- Tab 1: Input ----------
    with tab1:
        st.header("Input E-consultation Comments")
        input_method = st.radio("Choose input method:", ["Upload CSV/Excel File", "Paste Comments", "Manual Entry"])

        comments_data = None

        # --- Upload File ---
        if input_method == "Upload CSV/Excel File":
            uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
            if uploaded_file:
                try:
                    comments_data = processor.process_uploaded_file(uploaded_file)
                    st.success(f"Loaded {len(comments_data)} comments")

                    comment_column = st.selectbox("Select comment column:", comments_data.columns.tolist())
                    st.subheader("Data Preview")
                    st.dataframe(comments_data.head(10))

                    if st.button("Analyze Comments"):
                        with st.spinner("Analyzing..."):
                            results = analyzer.analyze_batch(
                                comments_data[comment_column].tolist(),
                                method=analysis_method.lower(),
                                confidence_threshold=confidence_threshold
                            )
                            st.session_state['analysis_results'] = results
                            st.session_state['original_data'] = comments_data
                            st.session_state['comment_column'] = comment_column
                            st.success("Analysis completed! Check the 'Analysis Results' tab.")

                except Exception as e:
                    st.error(f"Error: {e}")

        # --- Paste Comments ---
        elif input_method == "Paste Comments":
            comments_text = st.text_area("Paste comments (one per line):", height=200)
            if comments_text.strip() and st.button("Analyze Pasted Comments"):
                comments_list = [c.strip() for c in comments_text.split('\n') if c.strip()]
                if comments_list:
                    with st.spinner("Analyzing..."):
                        results = analyzer.analyze_batch(
                            comments_list,
                            method=analysis_method.lower(),
                            confidence_threshold=confidence_threshold
                        )
                        st.session_state['analysis_results'] = results
                        st.session_state['original_data'] = pd.DataFrame({'comment': comments_list})
                        st.session_state['comment_column'] = 'comment'
                        st.success(f"Analyzed {len(comments_list)} comments!")

        # --- Manual Entry ---
        else:
            if 'manual_comments' not in st.session_state:
                st.session_state['manual_comments'] = []

            new_comment = st.text_input("Enter a comment:")
            if st.button("Add Comment") and new_comment.strip():
                st.session_state['manual_comments'].append(new_comment.strip())
                st.success("Comment added!")
                st.rerun()

            if st.session_state['manual_comments']:
                st.subheader(f"Current Comments ({len(st.session_state['manual_comments'])})")
                for i, comment in enumerate(st.session_state['manual_comments']):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"{i+1}. {comment[:100]}{'...' if len(comment) > 100 else ''}")
                    with col2:
                        if st.button("Remove", key=f"remove_{i}"):
                            st.session_state['manual_comments'].pop(i)
                            st.rerun()

                if st.button("Analyze All Comments"):
                    with st.spinner("Analyzing..."):
                        results = analyzer.analyze_batch(
                            st.session_state['manual_comments'],
                            method=analysis_method.lower(),
                            confidence_threshold=confidence_threshold
                        )
                        st.session_state['analysis_results'] = results
                        st.session_state['original_data'] = pd.DataFrame({'comment': st.session_state['manual_comments']})
                        st.session_state['comment_column'] = 'comment'
                        st.success("Analysis completed!")

    # ---------- Tab 2: Results ----------
    with tab2:
        st.header("Sentiment Analysis Results")
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']

            col1, col2, col3, col4 = st.columns(4)
            positive_count = sum(r['sentiment']=='positive' for r in results)
            neutral_count = sum(r['sentiment']=='neutral' for r in results)
            negative_count = sum(r['sentiment']=='negative' for r in results)
            total = len(results)

            col1.metric("Total Comments", total)
            col2.metric("Positive", positive_count, f"{positive_count/total*100:.1f}%")
            col3.metric("Neutral", neutral_count, f"{neutral_count/total*100:.1f}%")
            col4.metric("Negative", negative_count, f"{negative_count/total*100:.1f}%")

            st.subheader("Sentiment Distribution")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(visualizer.create_pie_chart(results), use_container_width=True)
            with col2:
                st.plotly_chart(visualizer.create_bar_chart(results), use_container_width=True)

            st.subheader("Confidence Score Distribution")
            st.plotly_chart(visualizer.create_confidence_histogram(results), use_container_width=True)

            st.subheader("Detailed Analysis")
            results_df = pd.DataFrame([{
                'Comment': r['comment'][:100] + '...' if len(r['comment'])>100 else r['comment'],
                'Sentiment': r['sentiment'].title(),
                'Confidence': f"{r['confidence']:.3f}",
                'Positive Score': f"{r['scores']['positive']:.3f}",
                'Neutral Score': f"{r['scores']['neutral']:.3f}",
                'Negative Score': f"{r['scores']['negative']:.3f}"
            } for r in results])

            sentiment_filter = st.multiselect("Filter by sentiment:", ['Positive','Neutral','Negative'], default=['Positive','Neutral','Negative'])
            filtered_df = results_df[results_df['Sentiment'].isin(sentiment_filter)]
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.info("No analysis results available. Please analyze comments first.")

    # ---------- Tab 3: Export ----------
    with tab3:
        st.header("Export Analysis Results")
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            export_format = st.selectbox("Choose export format:", ["CSV","Excel","JSON"])
            export_data = processor.prepare_export_data(results, st.session_state['original_data'], st.session_state['comment_column'])

            st.subheader("Export Preview")
            st.dataframe(export_data.head(10))

            if export_format=="CSV":
                csv_buffer = io.StringIO()
                export_data.to_csv(csv_buffer,index=False)
                st.download_button("Download CSV", data=csv_buffer.getvalue(), file_name="sentiment_analysis_results.csv", mime="text/csv")
            elif export_format=="Excel":
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    export_data.to_excel(writer, sheet_name='Sentiment Analysis', index=False)
                    summary_data = processor.create_summary_sheet(results)
                    summary_data.to_excel(writer, sheet_name='Summary', index=False)
                st.download_button("Download Excel", data=excel_buffer.getvalue(), file_name="sentiment_analysis_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.download_button("Download JSON", data=export_data.to_json(orient='records', indent=2), file_name="sentiment_analysis_results.json", mime="application/json")

            st.subheader("Analysis Summary Report")
            summary_stats = processor.generate_summary_stats(results)
            for k,v in summary_stats.items():
                st.write(f"**{k}:** {v}")
        else:
            st.info("No analysis results available for export.")

if __name__=="__main__":
    main()
