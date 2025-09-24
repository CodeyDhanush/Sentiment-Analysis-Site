import pandas as pd
import io
import streamlit as st
from datetime import datetime
import json

class DataProcessor:
    """
    Data processing utilities for handling input and output data
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls']
    
    def process_uploaded_file(self, uploaded_file):
        """
        Process uploaded CSV or Excel file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame: Processed data
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings for CSV
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)  # Reset file pointer
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='cp1252')
            
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Basic data validation
            if df.empty:
                raise ValueError("The uploaded file is empty")
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Convert all columns to string for text analysis
            for col in df.columns:
                df[col] = df[col].astype(str)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    
    def validate_comments(self, comments):
        """
        Validate and clean comment data
        
        Args:
            comments (list): List of comments to validate
            
        Returns:
            list: Cleaned and validated comments
        """
        validated_comments = []
        
        for i, comment in enumerate(comments):
            # Convert to string if not already
            comment_str = str(comment) if comment is not None else ""
            
            # Skip empty or very short comments
            if len(comment_str.strip()) < 2:
                continue
            
            # Skip common non-content entries
            skip_patterns = ['nan', 'null', 'none', 'n/a', '']
            if comment_str.lower().strip() in skip_patterns:
                continue
            
            validated_comments.append(comment_str.strip())
        
        return validated_comments
    
    def prepare_export_data(self, analysis_results, original_data=None, comment_column=None):
        """
        Prepare data for export with analysis results
        
        Args:
            analysis_results (list): Sentiment analysis results
            original_data (DataFrame): Original input data
            comment_column (str): Name of the comment column
            
        Returns:
            pandas.DataFrame: Export-ready dataframe
        """
        export_data = []
        
        for i, result in enumerate(analysis_results):
            row = {
                'Comment_ID': i + 1,
                'Comment': result['comment'],
                'Sentiment': result['sentiment'].title(),
                'Confidence_Score': round(result['confidence'], 4),
                'Positive_Score': round(result['scores']['positive'], 4),
                'Neutral_Score': round(result['scores']['neutral'], 4),
                'Negative_Score': round(result['scores']['negative'], 4),
                'Analysis_Method': result['method'].upper(),
                'Analysis_Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add raw scores if available
            if 'raw_polarity' in result:
                row['Raw_Polarity'] = round(result['raw_polarity'], 4)
                row['Subjectivity'] = round(result['subjectivity'], 4)
            elif 'raw_compound' in result:
                row['Raw_Compound'] = round(result['raw_compound'], 4)
            
            # Add error information if present
            if 'error' in result:
                row['Analysis_Error'] = result['error']
            
            # Merge with original data if available
            if original_data is not None and not original_data.empty and i < len(original_data):
                for col in original_data.columns:
                    if col != comment_column:  # Avoid duplicate comment column
                        row[f'Original_{col}'] = original_data.iloc[i][col]
            
            export_data.append(row)
        
        return pd.DataFrame(export_data)
    
    def create_summary_sheet(self, analysis_results):
        """
        Create summary statistics for export
        
        Args:
            analysis_results (list): Sentiment analysis results
            
        Returns:
            pandas.DataFrame: Summary statistics
        """
        if not analysis_results:
            return pd.DataFrame()
        
        # Calculate statistics
        total_comments = len(analysis_results)
        sentiments = [r['sentiment'] for r in analysis_results]
        confidences = [r['confidence'] for r in analysis_results]
        
        positive_count = sentiments.count('positive')
        neutral_count = sentiments.count('neutral')
        negative_count = sentiments.count('negative')
        
        summary_data = [
            {'Metric': 'Total Comments Analyzed', 'Value': total_comments},
            {'Metric': 'Positive Comments', 'Value': positive_count},
            {'Metric': 'Neutral Comments', 'Value': neutral_count},
            {'Metric': 'Negative Comments', 'Value': negative_count},
            {'Metric': 'Positive Percentage', 'Value': f"{(positive_count/total_comments)*100:.2f}%"},
            {'Metric': 'Neutral Percentage', 'Value': f"{(neutral_count/total_comments)*100:.2f}%"},
            {'Metric': 'Negative Percentage', 'Value': f"{(negative_count/total_comments)*100:.2f}%"},
            {'Metric': 'Average Confidence Score', 'Value': f"{sum(confidences)/len(confidences):.4f}"},
            {'Metric': 'Highest Confidence Score', 'Value': f"{max(confidences):.4f}"},
            {'Metric': 'Lowest Confidence Score', 'Value': f"{min(confidences):.4f}"},
            {'Metric': 'Analysis Method', 'Value': analysis_results[0]['method'].upper()},
            {'Metric': 'Analysis Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ]
        
        return pd.DataFrame(summary_data)
    
    def generate_summary_stats(self, analysis_results):
        """
        Generate human-readable summary statistics
        
        Args:
            analysis_results (list): Sentiment analysis results
            
        Returns:
            dict: Summary statistics
        """
        if not analysis_results:
            return {}
        
        total = len(analysis_results)
        sentiments = [r['sentiment'] for r in analysis_results]
        confidences = [r['confidence'] for r in analysis_results]
        
        positive_count = sentiments.count('positive')
        neutral_count = sentiments.count('neutral')
        negative_count = sentiments.count('negative')
        
        # Determine overall sentiment trend
        if positive_count > neutral_count and positive_count > negative_count:
            overall_trend = "Predominantly Positive"
        elif negative_count > neutral_count and negative_count > positive_count:
            overall_trend = "Predominantly Negative"
        elif neutral_count > positive_count and neutral_count > negative_count:
            overall_trend = "Mostly Neutral"
        else:
            overall_trend = "Mixed Sentiment"
        
        # Confidence assessment
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence > 0.7:
            confidence_level = "High"
        elif avg_confidence > 0.4:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"
        
        return {
            "Total Comments": f"{total:,}",
            "Overall Sentiment Trend": overall_trend,
            "Positive Comments": f"{positive_count:,} ({positive_count/total*100:.1f}%)",
            "Neutral Comments": f"{neutral_count:,} ({neutral_count/total*100:.1f}%)",
            "Negative Comments": f"{negative_count:,} ({negative_count/total*100:.1f}%)",
            "Average Confidence": f"{avg_confidence:.3f} ({confidence_level})",
            "Analysis Method": analysis_results[0]['method'].upper(),
            "Generated On": datetime.now().strftime('%B %d, %Y at %I:%M %p')
        }
    
    def export_to_json(self, analysis_results, include_metadata=True):
        """
        Export analysis results to JSON format
        
        Args:
            analysis_results (list): Sentiment analysis results
            include_metadata (bool): Whether to include metadata
            
        Returns:
            str: JSON string
        """
        export_obj = {
            'results': analysis_results
        }
        
        if include_metadata:
            export_obj['metadata'] = {
                'total_comments': len(analysis_results),
                'analysis_method': analysis_results[0]['method'] if analysis_results else 'unknown',
                'export_timestamp': datetime.now().isoformat(),
                'summary': self.generate_summary_stats(analysis_results)
            }
        
        return json.dumps(export_obj, indent=2, ensure_ascii=False)
