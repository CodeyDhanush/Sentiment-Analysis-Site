# E-consultation Sentiment Analysis Tool

## Overview

This is a Streamlit-based web application designed to analyze sentiment in comments received through E-consultation modules. The tool provides comprehensive sentiment analysis capabilities using multiple algorithms (TextBlob and VADER) and offers interactive visualizations to help understand public opinion and feedback patterns from consultation processes.

The application processes uploaded data files (CSV/Excel), performs sentiment analysis on text content, and presents results through various charts and statistics. It's specifically designed for government agencies, organizations, or researchers conducting public consultations who need to quickly understand the sentiment trends in large volumes of feedback.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Web-based interface providing real-time interaction
- **Component-based Design**: Modular structure with separate classes for analysis, processing, and visualization
- **Caching Strategy**: Uses Streamlit's `@st.cache_resource` decorator for singleton components to improve performance
- **Responsive Layout**: Wide layout configuration optimized for data visualization

### Data Processing Pipeline
- **Multi-format Support**: Handles CSV, XLSX, and XLS file uploads
- **Encoding Resilience**: Implements fallback encoding strategies (UTF-8 → Latin-1 → CP1252) for CSV files
- **Data Validation**: Includes empty file detection and row cleaning functionality
- **Text Preprocessing**: Standardized text cleaning including HTML tag removal and whitespace normalization

### Sentiment Analysis Engine
- **Dual Algorithm Support**: 
  - TextBlob: Provides polarity and subjectivity scores
  - VADER: Optimized for social media text and informal language
- **Confidence Thresholding**: Configurable threshold system for sentiment classification
- **Preprocessing Pipeline**: Text normalization and cleaning before analysis
- **Error Handling**: Graceful degradation when sentiment libraries are unavailable

### Visualization System
- **Plotly Integration**: Interactive charts using Plotly for enhanced user experience
- **Multiple Chart Types**: Pie charts, bar charts, and distribution plots
- **Consistent Color Scheme**: Standardized colors for positive (Sea Green), neutral (Steel Blue), and negative (Crimson) sentiments
- **Dynamic Rendering**: Real-time chart updates based on analysis results

### Configuration Management
- **Sidebar Controls**: User-configurable analysis parameters
- **Method Selection**: Runtime switching between sentiment analysis algorithms
- **Threshold Adjustment**: Sliding scale for confidence level customization

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis for file processing
- **NumPy**: Numerical computing support for data operations

### Sentiment Analysis
- **TextBlob**: Natural language processing library for sentiment analysis
- **VADER Sentiment**: Social media optimized sentiment analysis tool

### Visualization
- **Plotly**: Interactive plotting library for charts and graphs
- **Plotly Express**: High-level plotting interface for quick visualizations

### File Processing
- **Built-in IO modules**: For handling file uploads and data streaming
- **Excel support**: Pandas integration for XLSX/XLS file processing

### Data Storage
- **In-memory Processing**: No persistent database required
- **Session State**: Streamlit's built-in session management for temporary data storage
- **File Upload Handling**: Temporary file processing through Streamlit's upload mechanism