## Core Functionality & Technical Highlights

### Data Acquisition Module

The data acquisition module is engineered for **resilient and precise data extraction** from dynamic API endpoints within the Wolt platform. It targets six prominent Romanian grocery retailers: Freshful, Profi, Penny, Auchan, Carrefour, and Kaufland.

* **Robust API Interaction:** Employs sophisticated **retry mechanisms with exponential backoff** to effectively manage API rate limits and network transients. This ensures data integrity and reliability during collection.
* **Advanced Data Extraction & Normalization:** Leverages **powerful regular expressions** for complex pattern matching and extraction of critical product attributes (e.g., quantities like "6x0.33L", "100g") from varied and often inconsistent product titles. Includes dedicated functions for data cleaning and standardization.
* **Unit Standardization & Price-per-Unit Calculation:** Implements custom logic to standardize diverse measurement units (e.g., converting 'g' to 'kg', 'ml' to 'l'). This accurately calculates the crucial "price per unit of measure," enabling genuine value comparison across different product sizes and packaging.
* **Efficient Data Archiving:** Collected data is intelligently **compressed for daily archiving**, ensuring efficient storage and providing a historical dataset for potential future analysis without consuming excessive disk space.
* **Dynamic Category Mapping:** Features a structured approach for **rule-based product categorization**, demonstrating a systematic method for semantic data organization.

### Interactive Dashboard Module

The interactive dashboard module provides a **highly responsive and intuitive user interface** for exploring and optimizing grocery purchases based on the locally collected data.

* **Stateful UI Management:** Utilizes advanced techniques for **persistent management of user selections, filters, and shopping cart contents** across application reruns, ensuring a seamless and intuitive user experience.
* **Dynamic Filtering & Search:** Implements **dynamic filtering capabilities** where options for units and categories are generated based on the currently available data, enhancing usability and relevance. Features a robust search bar for immediate product discovery.
* **Unique Item Tracking:** Employs a system for **assigning unique identifiers** to each product. This enables precise tracking and manipulation of items within the dynamic shopping cart, preventing ambiguity and ensuring accurate quantity adjustments and price calculations.
* **Real-time Cost Comparison:** Provides **instantaneous updates to shopping cart totals** across all included stores, allowing users to visually compare the overall cost of their selected items and identify the most economical purchasing option.
* **Scalable Data Handling:** Integrates powerful data manipulation libraries for **efficient in-memory data processing, filtering, and aggregation**, crucial for responsive performance even with growing datasets.
* **Polished Presentation:** Incorporates techniques for **embedding graphical elements directly** into the user interface, ensuring a crisp, high-quality visual presentation.

## Potential for Advanced AI/ML Integration

This project is architected as a **strong foundation for future enhancements** involving Artificial Intelligence and Machine Learning, showcasing foresight in system design:

* **AI-Driven Category Classification:** Explore **Machine Learning models** for more sophisticated and automated product categorization, moving beyond rule-based methods to enhance search and filter accuracy.
* **Enhanced Natural Language Processing (NLP) for Product Data:** Implement **NLP techniques** to improve the understanding and normalization of complex product descriptions, including handling variations, synonyms, and implicit information.
* **Generative AI for Personalized Recommendations:** Develop **Large Language Model (LLM) integrations** to create a more natural language-based shopping assistant, offering personalized recommendations or fulfilling complex queries (e.g., "Find ingredients for a pasta dish").
* **MLOps Best Practices:** Demonstrate the application of **MLOps principles** for managing the lifecycle of AI models within a self-hosted environment, ensuring reproducibility, monitoring, and continuous improvement of the data pipeline and predictive capabilities.

## Demo

https://appuctstoreappmainpy-qxaaofjlmggjkcejf3xyxx.streamlit.app/
