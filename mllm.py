import requests
import json
import os
from transformers import AutoTokenizer
import transformers
import torch
import re
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
# from openai import AzureOpenAI
import openai
#openai 0.27.0
def extract_output(text):
    # Find the output in the text
    output_pattern = r'### Output:(.*?)(?=###|$)'
    output_match = re.search(output_pattern, text, re.DOTALL)
    return output_match.group(1).strip() if output_match else None

def GPT4(prompt,version,key):
    openai.api_type = "azure"
    openai.api_base = "https://test-gpt-4-turbo-australia-east.openai.azure.com/"
    openai.api_version = "2023-07-01-preview"
    openai.api_key = "b1485beab36d4796841878836f6b3575"
    with open('template/template.txt', 'r') as f:
        template=f.readlines()
    if version=='multi-attribute':
        with open('template/human_multi_attribute_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    elif version=='complex-object':
        with open('template/complex_multi_object_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    
    textprompt= f"{' '.join(template)} \n {' '.join(incontext_examples)} \n {user_textprompt}"

    response = openai.ChatCompletion.create(
                    model='gpt-4-0314',
                    engine="gpt-4",
                    messages=[
                        {
                        "role": "user",
                        "content": textprompt
                        }
                    ],
                    temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                )

    # print(response.model_dump_json(indent=2))
    text=response.choices[0].message.content
    print(text)
    # Extract the split ratio and regional prompt

    return get_params_dict(text), textprompt

    # Define a function to query the OpenAI API and evaluate the answer
    def get_yes_no_answer(question):
        while True:
            try:
                
                break
            except openai.error.RateLimitError:
                pass
            except Exception as e:
                print(e)
            time.sleep(NUM_SECONDS_TO_SLEEP)

        answer = response['choices'][0]['message']['content']
        yes_no_regex = re.compile(r"^(yes|no)$", re.IGNORECASE)





    # gets the API Key from environment variable AZURE_OPENAI_API_KEY
    client = AzureOpenAI(
        base_url="https://test-gpt-4-turbo-australia-east.openai.azure.com/",
        api_key = key,
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        api_version="2023-07-01-preview",
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        # azure_endpoint="https://gcrendpoint.azurewebsites.net",
        # azure_deployment="gpt-4/chat/completions"
    )

    with open('template/template.txt', 'r') as f:
        template=f.readlines()
    if version=='multi-attribute':
        with open('template/human_multi_attribute_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    elif version=='complex-object':
        with open('template/complex_multi_object_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    
    textprompt= f"{' '.join(template)} \n {' '.join(incontext_examples)} \n {user_textprompt}"
    completion = client.chat.completions.create(
        model="gpt-4-0314",  # e.g. gpt-35-instant
        max_tokens=300,
        messages=[
            {
            "role": "user",
            "content": textprompt
            }
        ],
    )
    print(completion.model_dump_json(indent=2))
    text=completion.choices[0].message.content
    print(text)
    # Extract the split ratio and regional prompt

    return get_params_dict(text)



def GPT4_old(prompt,version,key):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = key
    with open('template/template.txt', 'r') as f:
        template=f.readlines()
    if version=='multi-attribute':
        with open('template/human_multi_attribute_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    elif version=='complex-object':
        with open('template/complex_multi_object_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    
    textprompt= f"{' '.join(template)} \n {' '.join(incontext_examples)} \n {user_textprompt}"
    
    payload = json.dumps({
    "model": "gpt-4-1106-preview", # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details. 
    "messages": [
        {
            "role": "user",
            "content": textprompt
        }
    ]
    })
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }
    print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    obj=response.json()
    text=obj['choices'][0]['message']['content']
    print(text)
    # Extract the split ratio and regional prompt

    return get_params_dict(text)

def local_llm(prompt,version,model_path=None):
    if model_path==None:
        model_id = "Llama-2-13b-chat-hf" 
    else:
        model_id=model_path
    print('Using model:',model_id)
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
    with open('template/template.txt', 'r') as f:
        template=f.readlines()
    if version=='multi-attribute':
        with open('template/human_multi_attribute_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    elif version=='complex-object':
        with open('template/complex_multi_object_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    textprompt= f"{' '.join(template)} \n {' '.join(incontext_examples)} \n {user_textprompt}"
    model_input = tokenizer(textprompt, return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad():
        print('waiting for LLM response')
        res = model.generate(**model_input, max_new_tokens=1024)[0]
        output=tokenizer.decode(res, skip_special_tokens=True)
        output = output.replace(textprompt,'')
    return get_params_dict(output)

def get_params_dict(output_text):
    split_ratio_marker = "Split ratio: "
    regional_prompt_marker = "Regional Prompt: "
    output_text=extract_output(output_text)
    print(output_text)
    # Find the start and end indices for the split ratio and regional prompt
    split_ratio_start = output_text.find(split_ratio_marker) + len(split_ratio_marker)
    split_ratio_end = output_text.find("\n", split_ratio_start)
    regional_prompt_start = output_text.find(regional_prompt_marker) + len(regional_prompt_marker)
    regional_prompt_end = len(output_text)  # Assuming Regional Prompt is at the end

    # Extract the split ratio and regional prompt from the text
    split_ratio = output_text[split_ratio_start:split_ratio_end].strip()
    regional_prompt = output_text[regional_prompt_start:regional_prompt_end].strip()
    #Delete the possible "(" and ")" in the split ratio
    split_ratio=split_ratio.replace('(','').replace(')','')
    # Create the dictionary with the extracted information
    image_region_dict = {
        'split ratio': split_ratio,
        'Regional Prompt': regional_prompt
    }
    print(image_region_dict)
    return image_region_dict