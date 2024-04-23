from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch
import pandas as pd
from PIL import Image
import io
from rouge import Rouge
import openai


def evaluate(output_path, llava_bench_path, our_model_path, llava_identifier):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # LLaVA model checkpoint
    model_id = llava_identifier

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

    # Provide the local path to the directory containing the model checkpoints
    our_model_checkpoint_path = our_model_path

    our_processor = AutoProcessor.from_pretrained(our_model_checkpoint_path)
    our_model = LlavaForConditionalGeneration.from_pretrained(our_model_checkpoint_path, quantization_config=quantization_config, device_map="auto")

    # Load LLaVA bench dataset into a dataframe
    df = pd.read_parquet(llava_bench_path)

    # Generate answers from both models
    questions = []
    captions = []
    llava_answers = []
    our_answers = []
    ground_truths = []

    for index, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        caption = row['caption']
        image = Image.open(io.BytesIO(row['image']['bytes']))
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        
        # Generation from baseline LLaVA model
        llava_inputs = processor([prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
        llava_output = model.generate(**llava_inputs, max_new_tokens=250)
        llava_generated_text = processor.batch_decode(llava_output, skip_special_tokens=True)
        llava_answers.append(llava_generated_text[0].split("ASSISTANT:")[-1].strip())
        
        # Generation from our LVQA model
        our_inputs = our_processor([prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
        our_output = our_model.generate(**our_inputs, max_new_tokens=250)
        our_generated_text = our_processor.batch_decode(our_output, skip_special_tokens=True)
        our_answers.append(our_generated_text[0].split("ASSISTANT:")[-1].strip())
        
        questions.append(question)
        ground_truths.append(answer)
        captions.append(caption)
        
    df = pd.DataFrame({'Question' : questions, 'LLaVA_answer': llava_answers, 'LVQA_answer': our_answers, 'Groundtruth_answer' : ground_truths, 'Caption' : captions})
    df.to_csv(output_path, index=False)

    # Collect scores

    # Initialize lists for the scores
    llava_scores = []
    lvqa_scores = []

    for index, row in df.iterrows():
        question = row['Question']
        description = row['Groundtruth_answer']
        assistant_1_answer = row['LLaVA_answer']
        assistant_2_answer = row['LVQA_answer']
        
        prompt = f'''
        We would like to request your feedback on the performance of two AI assistants in response to user questions. 
        The user asks the question on observing an image. For your reference, the visual content in the image is represented as a description with a few sentences 
        describing the image. \nPlease rate the helpfulness, relevance, accuracy, coherence, level of details of their responses. Each assistant receives an 
        overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing 
        only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, 
        please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses 
        were presented does not affect your judgment. Use the image description wisely to evaluate the answers.
        
        

        USER QUESTION : {question}
        IMAGE DESCRIPTION : {description}
        ASSISTANT 1 ANSWER : {assistant_1_answer}
        ASSISTANT 2 ANSWER : {assistant_2_answer}
        '''

        response = openai.ChatCompletion.create( model="gpt-4", messages=[{"role": "user", "content": prompt}])

        answer = response.choices[0].message.content

        scores = answer.split('\n')[0].split()
        
        llava_scores.append(int(scores[0]))
        lvqa_scores.append(int(scores[1]))

    # Calculate the average of each list
    average_llava = sum(llava_scores) / len(llava_scores)
    average_lvqa = sum(lvqa_scores) / len(lvqa_scores)

    print("Average of llava scores:", average_llava)
    print("Average of lvqa scores:", average_lvqa)

    # Calculate BLEU scores of assistant responses with GPT-4 generated answer as reference

    llava_bleu_scores = []
    lvqa_bleu_scores = []

    for index, row in df.iterrows():
        
        reference_text = row['Groundtruth_answer']
        llava_answer = row['LLaVA_answer']
        lvqa_answer = row['LVQA_answer']

        # Tokenize the texts
        reference_tokens = reference_text.split()
        llava_answer_tokens = llava_answer.split()
        lvqa_answer_tokens = lvqa_answer.split()

        # Calculate BLEU score
        llava_bleu_score = sentence_bleu([reference_tokens], llava_answer_tokens)
        lvqa_bleu_score = sentence_bleu([reference_tokens], lvqa_answer_tokens)
        
        # Append the scores to a list
        llava_bleu_scores.append(llava_bleu_score)
        lvqa_bleu_scores.append(lvqa_bleu_score)
        
    # Calculate the average of each list
    average_llava = sum(llava_bleu_scores) / len(llava_bleu_scores)
    average_lvqa = sum(lvqa_bleu_scores) / len(lvqa_bleu_scores)

    print("Average LLaVA BLEU score:", average_llava)
    print("Average LVQA BLEU score:", average_lvqa)

    # Calculate ROUGE scores of assistant responses with GPT-4 generated answer as reference

    # Initialize Rouge
    rouge = Rouge()

    llava_rouge_1_scores = []
    lvqa_rouge_1_scores = []
    llava_rouge_2_scores = []
    lvqa_rouge_2_scores = []
    llava_rouge_l_scores = []
    lvqa_rouge_l_scores = []

    for index, row in df.iterrows():
        
        reference_answer = row['Groundtruth_answer']
        llava_answer = row['LLaVA_answer']
        lvqa_answer = row['LVQA_answer']

        # Calculate ROUGE scores
        llava_scores = rouge.get_scores(llava_answer, reference_answer)
        lvqa_scores = rouge.get_scores(lvqa_answer, reference_answer)
        
        # Extract ROUGE-1, ROUGE-2, and ROUGE-L scores
        llava_rouge_1_score = llava_scores[0]['rouge-1']['f']
        llava_rouge_2_score = llava_scores[0]['rouge-2']['f']
        llava_rouge_l_score = llava_scores[0]['rouge-l']['f']
        
        lvqa_rouge_1_score = lvqa_scores[0]['rouge-1']['f']
        lvqa_rouge_2_score = lvqa_scores[0]['rouge-2']['f']
        lvqa_rouge_l_score = lvqa_scores[0]['rouge-l']['f']
        
        # Append the scores to a list
        llava_rouge_1_scores.append(llava_rouge_1_score)
        lvqa_rouge_1_scores.append(lvqa_rouge_1_score)
        llava_rouge_2_scores.append(llava_rouge_2_score)
        lvqa_rouge_2_scores.append(lvqa_rouge_2_score)
        llava_rouge_l_scores.append(llava_rouge_l_score)
        lvqa_rouge_l_scores.append(lvqa_rouge_l_score)
        
    # Calculate the average of each list
    average_llava_rouge_1_score = sum(llava_rouge_1_scores) / len(llava_rouge_1_scores)
    average_lvqa_rouge_1_score = sum(lvqa_rouge_1_scores) / len(lvqa_rouge_1_scores)
    average_llava_rouge_2_score = sum(llava_rouge_2_scores) / len(llava_rouge_2_scores)
    average_lvqa_rouge_2_score = sum(lvqa_rouge_2_scores) / len(lvqa_rouge_2_scores)
    average_llava_rouge_l_score = sum(llava_rouge_l_scores) / len(llava_rouge_l_scores)
    average_lvqa_rouge_l_score = sum(lvqa_rouge_l_scores) / len(lvqa_rouge_l_scores)

    print("Average LLaVA ROUGE 1 score:", average_llava_rouge_1_score)
    print("Average LVQA ROUGE 1 score:", average_lvqa_rouge_1_score)
    print("Average LLaVA ROUGE 2 score:", average_llava_rouge_2_score)
    print("Average LVQA ROUGE 2 score:", average_lvqa_rouge_2_score)
    print("Average LLaVA ROUGE L score:", average_llava_rouge_l_score)
    print("Average LVQA ROUGE L score:", average_lvqa_rouge_l_score)


if __name__ == "__main__":
    checkpoint_dir = "./checkpoints"
    output_path = "/kaggle/working/answers_df.csv"
    llava_bench_path = "/kaggle/working/answers_df.csv"
    our_model_path = "/kaggle/input/lvqa/checkpoints"
    llava_identifier = "llava-hf/llava-1.5-7b-hf"

    evaluate(output_path, llava_bench_path, our_model_path, llava_identifier)