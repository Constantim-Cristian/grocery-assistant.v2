# AI-Powered Grocery Price Comparison System

An intelligent web scraping and data analysis platform that leverages advanced AI/ML techniques to compare grocery prices across major Romanian retailers. The system combines robust data acquisition, fine-tuned language models, and intelligent search capabilities to provide accurate price-per-unit comparisons.

## Overview

This system scrapes product data from six major Romanian grocery retailers (Freshful, Profi, Penny, Auchan, Carrefour, and Kaufland) via Wolt's API, then applies sophisticated AI models to extract precise quantities and units from product titles, enabling true price-per-unit comparisons across different stores and package sizes.

## Key Features

### Advanced AI Integration
- **Fine-tuned LoRA Models**: Custom-trained lightweight adapters for precise quantity/unit extraction from complex product titles  
- **AI-Powered Category Mapping**: Machine learning-driven system that creates unified product categories across all retailers  
- **Intelligent Search System**: Hybrid TF-IDF and AI-powered semantic search with smart fallback mechanisms  
- **Optimized AI Usage**: Reuses processed data and saves new results to minimize computational overhead  

### Robust Data Acquisition
- **Resilient API Scraping**: Sophisticated retry mechanisms with exponential backoff for handling rate limits and network issues  
- **Advanced Pattern Recognition**: Complex regex patterns for extracting quantities like `6x0.33L`, `100g/bucată`, `vrac/kg`  
- **Unit Standardization**: Intelligent conversion system normalizing diverse Romanian measurement units  
- **Dynamic Pagination Handling**: Comprehensive coverage of all product listings across multiple store categories  

### Smart Data Processing
- **Real-time Quality Assessment**: Flags potentially incorrect extractions using price-per-unit thresholds  
- **Duplicate Detection**: Image-based deduplication ensuring data accuracy  
- **Historical Data Management**: Efficient daily archiving with compression for trend analysis  
- **Error Recovery**: Infinite retry system ensuring complete data collection  

### Interactive Dashboard
- **Stateful UI Management**: Persistent user selections and shopping cart across sessions  
- **Dynamic Filtering**: Real-time filter options based on available data  
- **Cost Comparison**: Instant total price calculations across all retailers  
- **Visual Product Gallery**: High-quality image integration with detailed product information  

## Technical Architecture

### AI/ML Pipeline
Raw Product Titles → Preprocessing → Fine-tuned LoRA Model → Quantity/Unit Extraction
                                                         ↓
Category Classification ← AI Category Mapping ← Standardized Data
                                                         ↓
Search Index Creation ← TF-IDF + Semantic Search ← Processed Products

### Data Flow
API Endpoints → Scraper → AI Processing → Database → Dashboard
     ↓              ↓           ↓            ↓          ↓
Rate Limiting   Retry Logic   LoRA Models   Storage   User Interface

## Core Components

### 1. Data Acquisition Engine
- Multi-store API Integration: Unified scraping across 6 major retailers  
- Intelligent Error Handling: Exponential backoff and infinite retry mechanisms  
- Dynamic Category Discovery: Automatic detection of store category structures  
- Pagination Management: Complete product catalog coverage  

### 2. AI Processing Pipeline
- LoRA Fine-tuning: Lightweight model adaptation for Romanian product titles  
- Quantity Extraction: Advanced pattern matching with AI fallback  
- Category Unification: ML-driven mapping between store-specific categories  
- Quality Validation: AI-assisted detection of extraction errors  

### 3. Search & Discovery System
- Hybrid Search: TF-IDF primary search with AI semantic fallback  
- Performance Optimization: Cached results and incremental processing  
- Multi-language Support: Romanian product title understanding  
- Fuzzy Matching: Handles variations in product naming  

### 4. Interactive Frontend
- React-based Dashboard: Responsive and intuitive user interface  
- Real-time Updates: Dynamic price comparisons and cart calculations  
- Advanced Filtering: Multi-dimensional product discovery  
- Visual Analytics: Price trend visualization and comparison charts  
