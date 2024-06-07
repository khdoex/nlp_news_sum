# Summarization and BERTScore Evaluation Script

This project provides a script to generate text summaries using different models and evaluate the generated summaries using BERTScore. The script allows you to specify various parameters for summarization and calculates BERTScore metrics, saving the results in a CSV file.

## Features

- Generate summaries using different models and parameters
- Calculate BERTScore for evaluating generated summaries
- Save generated summaries and BERTScore metrics in a CSV file

## Requirements


- Install the necessary packages:
  ```bash
  pip install pandas torch transformers tqdm nltk bert_score
    ```

## Usage
- Save the Script: Save the script as summarization_and_bert_score.py.

- Prepare Your Data: Ensure your data file (e.g., your_data.csv) contains articles and reference summaries.

- Run the Script: Use the command line to run the script with your desired parameters. For example:

    ```bash
    python summarization_and_bert_score.py --model "facebook/bart-large-cnn" --data_path "your_data.csv" --output_path "output_summaries" --num_beams 8 --length_penalty 1.5 --max_length 120 --min_length 40

    ```

## Command-Line Arguments
- model: Model name for summarization (default: facebook/bart-large-cnn)
- data_path: Path to the data file (default: chunk_2.csv)
- output_path: Output path prefix for the results (default: generated_summaries)
- num_beams: Number of beams for beam search (default: 4)
- length_penalty: Length penalty for beam search (default: 2.0)
- max_length: Maximum length of the summary (default: 150)
- min_length: Minimum length of the summary (default: 30)
- batch_size: Batch size for summary generation (default: 8)
- limit: Limit the number of samples (default: 7000)


## Script Overview
- Load Data: Loads the input data file containing articles.
- Initialize Model and Tokenizer: Loads the specified summarization model and tokenizer.
- Tokenize Articles: Preprocesses and tokenizes the articles for the model.
- Generate Summaries: Generates summaries for the articles using the model.
- Calculate BERTScore: Calculates BERTScore for the generated summaries against the reference summaries.
- Save Results: Saves the generated summaries along with BERTScore metrics to a CSV file.