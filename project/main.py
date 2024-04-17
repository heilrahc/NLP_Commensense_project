import numpy as np
import pandas as pd
import os
import random

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags

# Assuming these imports are for evaluation metrics, replace with actual libraries as needed
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from evaluate import load

from bert_score import score as bert_score
from bleurt import score as bleurt_score

from QRelScore.evalpackage.qrelscore import QRelScore
from QRelScore.evalpackage.clmscore import CLMScore
from QRelScore.evalpackage.mlmscore import MLMScore

def load_data(dataset_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(dataset_path)
    
    # Return the DataFrame
    return data

def evaluate_question(generated_question, reference_question, context=None, answer=None):
    """
    Compute evaluation metrics for the generated question against the reference question.
    Including BLEU, ROUGE, BERTScore, and BLEURT. RQUGE is computed if reference_question is None.
    """
    # Initialize the scores
    scores = {}
    
    rouge = Rouge()
    rqugescore = load("alirezamsh/rquge")
    bleurt_scorer = bleurt_score.BleurtScorer()
    QRelScorer = QRelScore()
    CLMScorer = CLMScore()
    MLMScorer = MLMScore()
    
    
    print("generated_question length: ", len(generated_question))
    print("answer length: ", len(answer))
    #Truncate the context and generated question to fit within the model's max sequence length
    max_length = 512  # Maximum sequence length
    context = ' '.join(context.split()[:max_length//2]) if len(context) > max_length else context
    generated_question = ' '.join(generated_question.split()[:max_length//2]) if len(generated_question) > max_length else generated_question
    answer = ' '.join(answer.split()[:max_length//2]) if len(answer) > max_length else answer

    # Compute evaluation metrics only if there is a reference question
    if reference_question:
        # Compute BLEU score
        scores['BLEU'] = sentence_bleu([reference_question.split()], generated_question.split())
        print("BLEU: ", scores['BLEU'])
        
        # Compute ROUGE score
        rouge_score = rouge.get_scores(generated_question, reference_question)[0]
        scores['ROUGE'] = rouge_score['rouge-1']['f']  # F1-score of ROUGE-1
        print("ROUGE: ", scores['ROUGE'])
        
        # Compute BERTScore
        _, _, bert_score_f = bert_score([generated_question], [reference_question], lang="en")
        scores['BERTScore'] = bert_score_f.mean().item()  # F1-score of BERTScore
        print("BERTScore: ", scores['BERTScore'])
        
        # Compute BLEURT score
        print("start BLEURT")
        scores['BLEURT'] = bleurt_scorer.score(references=[reference_question], candidates=[generated_question])[0]
        print("BLEURT: ", scores['BLEURT'])
        
        
    
    ### Reference Free ###
    scores['RQUGE'] = rqugescore.compute(generated_questions=[generated_question], contexts=[context], answers=[answer])['mean_score'] #range from 1 to 5
    print("RQUGE: ", scores['RQUGE'])
    scores['QREL'] = QRelScorer.compute_score_flatten([context], [generated_question])
    print("QREL: ", scores['QREL'])
    scores['CLM'] = CLMScorer.compute_score_flatten([context], [generated_question])
    print("CLM: ", scores['CLM'])
    scores['MLM'] = MLMScorer.compute_score_flatten([context], [generated_question])
    print("MLM: ", scores['MLM'])
    


    return scores

def extract_answer_span(questions_series, portion=0.5):
    all_answer_spans = ""
    
    questions = sent_tokenize(questions_series)
        
    # Calculate the number of questions to sample
    sample_size = max(1, int(len(questions) * portion))
        
    # Randomly select a portion of the questions
    sampled_questions = random.sample(questions, sample_size)

    for question in sampled_questions:
        tokens = word_tokenize(question)
        pos_tags = pos_tag(tokens)

        pattern = 'NP: {<DT>?<JJ>*<NN.*>+}'
        cp = nltk.RegexpParser(pattern)
        cs = cp.parse(pos_tags)

        iob_tagged = tree2conlltags(cs)
        chunks = [word for word, pos, chunk in iob_tagged if chunk.startswith('B-') or chunk.startswith('I-')]

        answer_span = ' '.join(chunks) if chunks else "The response would involve discussing aspects related to the question's focus."
        all_answer_spans += answer_span + " "

    return all_answer_spans.strip()

def main(dataset_path):
    print("start main")
    # Ensure the necessary NLTK resources are available
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    # Load the dataset
    print("loading the data")
    data = load_data(dataset_path)
    
    # Evaluate each row
    evaluation_results = []
    for index, row in data.iterrows():
        print("ROW #: ", index)
        context = f"{row['summarized_chat_history']}{row['Q']} {row['Summarized_A']}".strip() if pd.notnull(row['summarized_chat_history']) else f"{row['Q']} {row['Summarized_A']}".strip()
        print("Sequence length of context:", len(context))
        answer = extract_answer_span(row['QL'])
        reference_question = row['follow_up'] if 'follow_up' in row and pd.notnull(row['follow_up']) else None

        generated_questions = [
            ('qa_follow_up', row['qa_follow_up']),
            ('qa_ql_follow_up', row['qa_ql_follow_up']),
            ('qa_ch_follow_up', row['qa_ch_follow_up']),
            ('qa_ch_ql_follow_up', row['qa_ch_ql_follow_up'])
        ]

        for gen_q_label, gen_q in generated_questions:
            if pd.notnull(gen_q):
                scores = evaluate_question(gen_q, reference_question, context, answer)
                evaluation_results.append((index, gen_q_label, scores))

    # Convert to DataFrame for better visualization
    evaluation_df = pd.DataFrame(evaluation_results, columns=['Row', 'Generated Question Label', 'Scores'])
    csv_file_path = os.path.join(os.path.dirname(dataset_path), 'evaluation_results_more_metrics_trimmed.csv')
    evaluation_df.to_csv(csv_file_path, index=False)
    print(f"Saved evaluation results to {csv_file_path}")

if __name__ == "__main__":
    dataset_path = '/Users/charliehe/Desktop/nlp commense/project/data/interview_full_data_v2.csv'
    main(dataset_path)