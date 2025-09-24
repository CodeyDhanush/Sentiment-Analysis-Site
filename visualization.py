import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from collections import Counter

class SentimentVisualizer:
    """
    Visualization utilities for sentiment analysis results
    """
    
    def __init__(self):
        # Color scheme for sentiments
        self.colors = {
            'positive': '#2E8B57',  # Sea Green
            'neutral': '#4682B4',   # Steel Blue
            'negative': '#DC143C'   # Crimson
        }
        
        self.color_sequence = [self.colors['positive'], self.colors['neutral'], self.colors['negative']]
    
    def create_pie_chart(self, analysis_results):
        """
        Create pie chart showing sentiment distribution
        
        Args:
            analysis_results (list): Sentiment analysis results
            
        Returns:
            plotly.graph_objects.Figure: Pie chart figure
        """
        if not analysis_results:
            return go.Figure()
        
        # Count sentiments
        sentiments = [result['sentiment'] for result in analysis_results]
        sentiment_counts = Counter(sentiments)
        
        # Prepare data
        labels = [sentiment.title() for sentiment in sentiment_counts.keys()]
        values = list(sentiment_counts.values())
        colors = [self.colors.get(sentiment, '#808080') for sentiment in sentiment_counts.keys()]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo='label+percent+value',
            textfont=dict(size=12),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': 'Sentiment Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def create_bar_chart(self, analysis_results):
        """
        Create bar chart showing sentiment counts
        
        Args:
            analysis_results (list): Sentiment analysis results
            
        Returns:
            plotly.graph_objects.Figure: Bar chart figure
        """
        if not analysis_results:
            return go.Figure()
        
        # Count sentiments
        sentiments = [result['sentiment'] for result in analysis_results]
        sentiment_counts = Counter(sentiments)
        
        # Ensure all sentiment types are represented
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment not in sentiment_counts:
                sentiment_counts[sentiment] = 0
        
        # Sort by sentiment type
        ordered_sentiments = ['positive', 'neutral', 'negative']
        labels = [sentiment.title() for sentiment in ordered_sentiments]
        values = [sentiment_counts[sentiment] for sentiment in ordered_sentiments]
        colors = [self.colors[sentiment] for sentiment in ordered_sentiments]
        
        # Calculate percentages
        total = sum(values)
        percentages = [f"{(value/total)*100:.1f}%" if total > 0 else "0%" for value in values]
        
        # Create bar chart
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{val}<br>({pct})" for val, pct in zip(values, percentages)],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': 'Sentiment Count Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Sentiment',
            yaxis_title='Number of Comments',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            showlegend=False
        )
        
        return fig
    
    def create_confidence_histogram(self, analysis_results):
        """
        Create histogram showing confidence score distribution
        
        Args:
            analysis_results (list): Sentiment analysis results
            
        Returns:
            plotly.graph_objects.Figure: Histogram figure
        """
        if not analysis_results:
            return go.Figure()
        
        # Extract confidence scores
        confidences = [result['confidence'] for result in analysis_results]
        
        # Create histogram
        fig = go.Figure(data=[go.Histogram(
            x=confidences,
            nbinsx=20,
            marker_color='steelblue',
            opacity=0.7,
            hovertemplate='Confidence Range: %{x}<br>Count: %{y}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': 'Confidence Score Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Confidence Score',
            yaxis_title='Number of Comments',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            bargap=0.1
        )
        
        # Add statistical information
        mean_confidence = np.mean(confidences)
        fig.add_vline(
            x=mean_confidence,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_confidence:.3f}"
        )
        
        return fig
    
    def create_sentiment_by_confidence(self, analysis_results):
        """
        Create scatter plot showing sentiment vs confidence
        
        Args:
            analysis_results (list): Sentiment analysis results
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot figure
        """
        if not analysis_results:
            return go.Figure()
        
        # Prepare data
        df_data = []
        for i, result in enumerate(analysis_results):
            df_data.append({
                'index': i + 1,
                'sentiment': result['sentiment'].title(),
                'confidence': result['confidence'],
                'comment_preview': result['comment'][:100] + '...' if len(result['comment']) > 100 else result['comment']
            })
        
        df = pd.DataFrame(df_data)
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='index',
            y='confidence',
            color='sentiment',
            color_discrete_map={
                'Positive': self.colors['positive'],
                'Neutral': self.colors['neutral'],
                'Negative': self.colors['negative']
            },
            hover_data=['comment_preview'],
            title='Sentiment Confidence by Comment Index'
        )
        
        fig.update_layout(
            xaxis_title='Comment Index',
            yaxis_title='Confidence Score',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def create_score_comparison(self, analysis_results, max_comments=50):
        """
        Create stacked bar chart comparing positive, neutral, and negative scores
        
        Args:
            analysis_results (list): Sentiment analysis results
            max_comments (int): Maximum number of comments to display
            
        Returns:
            plotly.graph_objects.Figure: Stacked bar chart figure
        """
        if not analysis_results:
            return go.Figure()
        
        # Limit results for readability
        results_subset = analysis_results[:max_comments]
        
        # Extract scores
        indices = list(range(1, len(results_subset) + 1))
        positive_scores = [result['scores']['positive'] for result in results_subset]
        neutral_scores = [result['scores']['neutral'] for result in results_subset]
        negative_scores = [result['scores']['negative'] for result in results_subset]
        
        # Create stacked bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Positive',
            x=indices,
            y=positive_scores,
            marker_color=self.colors['positive'],
            hovertemplate='Comment %{x}<br>Positive: %{y:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Neutral',
            x=indices,
            y=neutral_scores,
            marker_color=self.colors['neutral'],
            hovertemplate='Comment %{x}<br>Neutral: %{y:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Negative',
            x=indices,
            y=negative_scores,
            marker_color=self.colors['negative'],
            hovertemplate='Comment %{x}<br>Negative: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f'Sentiment Score Breakdown (First {len(results_subset)} Comments)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Comment Index',
            yaxis_title='Score',
            barmode='stack',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
