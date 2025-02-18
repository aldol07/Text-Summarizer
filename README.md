# Text Summarizer Project

## Overview

This project implements a text summarization application using natural language processing techniques. It provides a web interface (built with Flask) for users to input text and generate concise summaries. The project explores different summarization methods and techniques.

## Features

*   **Extractive Summarization:** Extracts the most important sentences from the input text to create a summary.
*   **Abstractive Summarization:** (Potentially - *If implemented*) Generates new sentences that capture the main idea of the input text.
*   **Web Interface:** A user-friendly web application built with Flask for easy interaction.
*   **Configurable Parameters:**  Allows users to adjust summarization parameters (e.g., summary length) via the `params.yaml` file.
*   **Modular Structure:** Well-organized codebase with separate modules for data processing, model training, and application deployment.

## Installation

1.  **Clone the repository:**

    ```
    git clone https://github.com/aldol07/Text-Summarizer.git
    cd Text-Summarizer
    ```

2.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate   # On Linux/macOS
    venv\Scripts\activate.bat  # On Windows
    ```

3.  **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

## Usage

1.  **Configure parameters:**

    *   Modify the `params.yaml` file to adjust summarization settings (e.g., model paths, summary length).  Refer to comments within the file for details.

2.  **Run the application:**

    ```
    python app.py
    ```

3.  **Access the web interface:**





