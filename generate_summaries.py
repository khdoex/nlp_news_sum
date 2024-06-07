import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
from bert_score import score as bert_score
import nltk
from concurrent.futures import ThreadPoolExecutor

# Ensure that the necessary NLTK data files are downloaded
nltk.download('punkt')

# Preprocess text for BERTScore
def preprocess_text(text):
    return ' '.join(text.split()).lower()

def load_data(data_path, limit=7000):
    data = pd.read_csv(data_path)
    data = data.head(limit)  # Limit to the first `limit` samples
    return data['article'].tolist(), data

def initialize_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move the model to the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return tokenizer, model, device

def tokenize_articles(articles, tokenizer, max_length=1024):
    inputs = {'input_ids': [], 'attention_mask': []}
    for article in tqdm(articles, desc="Tokenizing articles"):
        tokenized = tokenizer(article, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
        inputs['input_ids'].append(tokenized['input_ids'])
        inputs['attention_mask'].append(tokenized['attention_mask'])
    
    # Stack the tokenized inputs
    inputs['input_ids'] = torch.cat(inputs['input_ids'], dim=0)
    inputs['attention_mask'] = torch.cat(inputs['attention_mask'], dim=0)
    return inputs

def generate_summaries(inputs, model, tokenizer, device, num_beams=4, length_penalty=2.0, max_length=150, min_length=30, batch_size=8):
    summaries = []
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for i in tqdm(range(0, len(inputs['input_ids']), batch_size), desc="Generating summaries"):
            input_ids_batch = inputs['input_ids'][i:i+batch_size].to(device)
            attention_mask_batch = inputs['attention_mask'][i:i+batch_size].to(device)
            summary_ids = model.generate(
                input_ids_batch,
                attention_mask=attention_mask_batch,
                num_beams=num_beams,
                length_penalty=length_penalty,
                max_length=max_length,
                min_length=min_length,
                early_stopping=True
            )
            decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            summaries.extend(decoded_summaries)
    return summaries

def calculate_bert_score(data, device):
    data['summary'] = data['summary'].fillna('')
    data['generated_summary'] = data['generated_summary'].fillna('')
    
    data['preprocessed_summary'] = data['summary'].apply(preprocess_text)
    data['preprocessed_generated_summary'] = data['generated_summary'].apply(preprocess_text)

    P, R, F1 = bert_score(data['preprocessed_generated_summary'].tolist(), data['preprocessed_summary'].tolist(), lang='en', verbose=True, device=device)
    data['bert_precision'] = P.tolist()
    data['bert_recall'] = R.tolist()
    data['bert_f1'] = F1.tolist()

    return data

def save_results(data, params, output_path):
    output_file = f'{output_path}_beams_{params["num_beams"]}_lengthpenalty_{params["length_penalty"]}_maxlength_{params["max_length"]}_minlength_{params["min_length"]}.csv'
    data.to_csv(output_file, index=False)
    print(f"Results with summaries and BERTScore metrics saved to '{output_file}'")

def main():
    parser = argparse.ArgumentParser(description="Generate summaries using various models and parameters and calculate BERTScores.")
    parser.add_argument('--model', type=str, default="facebook/bart-large-cnn", help="Model name (default: facebook/bart-large-cnn)")
    parser.add_argument('--data_path', type=str, default='chunk_2.csv', help="Path to the data file (default: 'chunk_2.csv')")
    parser.add_argument('--output_path', type=str, default='generated_summaries', help="Output path prefix (default: 'generated_summaries')")
    parser.add_argument('--num_beams', type=int, default=4, help="Number of beams for beam search (default: 4)")
    parser.add_argument('--length_penalty', type=float, default=2.0, help="Length penalty for beam search (default: 2.0)")
    parser.add_argument('--max_length', type=int, default=150, help="Maximum length of the summary (default: 150)")
    parser.add_argument('--min_length', type=int, default=30, help="Minimum length of the summary (default: 30)")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for summary generation (default: 8)")
    parser.add_argument('--limit', type=int, default=7000, help="Limit the number of samples (default: 7000)")

    args = parser.parse_args()

    # Load data
    articles, data = load_data(args.data_path, args.limit)
    
    # Initialize model and tokenizer
    tokenizer, model, device = initialize_model_and_tokenizer(args.model)
    
    # Tokenize articles
    inputs = tokenize_articles(articles, tokenizer)

    # Generate summaries
    params = {
        'num_beams': args.num_beams,
        'length_penalty': args.length_penalty,
        'max_length': args.max_length,
        'min_length': args.min_length,
        'batch_size': args.batch_size
    }
    summaries = generate_summaries(inputs, model, tokenizer, device, **params)
    data['generated_summary'] = summaries

    # Calculate BERTScore
    data = calculate_bert_score(data, device)
    
    # Save results
    save_results(data, params, args.output_path)

if __name__ == "__main__":
    main()
