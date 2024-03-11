from openai import AzureOpenAI
import base64
import requests
from PIL import Image
import io
import json
import tqdm
 
   
# Function to encode the image
def encode_image(image_path):
    # Open an image file
    with Image.open(image_path) as img:
        # get the image size
        width, height = img.size
        print('original image size: ', img.size)
        if width > 2048 or height > 2048:
            aspect_ratio = width / height
            if width > height:  
                new_width = 2048  
                new_height = round(new_width / aspect_ratio)  
            else:  
                new_height = 2048  
                new_width = round(new_height * aspect_ratio)  
            # Resize the image
            img = img.resize((new_width, new_height))
            print('new image size: ', img.size)
           
        buffer = io.BytesIO()  
        img.save(buffer, format="JPEG")  
 
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
 
json_path="/pyy/openseg_blob/yuyang/datasets/RPG/shape10k/meta.json" 
with open(json_path, 'r') as f:
    meta = json.load(f)
error_cnt=0
for n,item in tqdm.tqdm(enumerate(meta)):
    image_path = item['path'].replace("shape","shape10k")
    meta[n]
    shape=item['shape']
    fill=item['fill']
    color=item['color']
    outline_color=item['outline_color']

 
    # Getting the base64 string
    base64_image = encode_image(image_path)
    
    # gets the API Key from environment variable AZURE_OPENAI_API_KEY
    client = AzureOpenAI(
        api_key = "oaip_FVObatuUKrwbktgmyhlCFLQxiAreiNaC",
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        api_version="2023-05-15",
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint="https://gcrendpoint.azurewebsites.net",
    )
    instruction="You are an experienced stable diffusion prompt engineer. Please give the caption about the image given above that can be used by a SDXL model as the text prompt to generate exactly the same image. The caption should be a complete sentence and no more than 70 words. The caption should be relevant to the reference words. Here are some examples:\n "
    example="Example one: reference:<shape: oval>, <color: silver>, <fill: solid>. caption: The image displays a single, two-dimensional oval shape with a solid fill. The oval appears centrally placed against a pure black background that extends to the edges of the frame. The overall impression of the image is one of minimalism and clarity, with the focus entirely on the solitary silver oval against the deep black backdrop.\n Example two: reference:<shape: parallelogram>, <color: black>, <fill: empty>, <outline_color: linen>. caption: The image displays a graphic representation of a two-dimensional geometric shape, specifically a parallelogram which is outlined with a linen color. The outline of the shape is distinct, providing a sharp contrast that clearly defines the edges of the parallelogram.The interior of the parallelogram is black, different from the outline color. Overall, the image conveys a sense of simplicity and precision.\n "
    if fill:
        fill="solid"
        reference=f"Now the reference words are:<shape: {shape}>, <color: {color}>, <fill: {fill}>. Please give the caption about the image:"
    else:
        fill="empty"
        reference=f"Now the reference words are:<shape: {shape}>, <color: {color}>, <fill: {fill}>, <outline_color: {outline_color}>. Please give the caption about the image:"
    text = instruction+example+reference
    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo-v",  # e.g. gpt-35-instant
            max_tokens=300,
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    # "text": "Write a caption for this image."
                    # "text": "Write a sentence to describe this image. Focus on the scene, objects and movements. No more than 40 words."
                    "text": text
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                    }
                ]
                }
            ],
        )
    
        # print(completion.model_dump_json(indent=2))
        print(completion.choices[0].message.content)
        text_prompt=completion.choices[0].message.content
    except Exception as e:
        print("ERROR!")
        error_cnt+=1
        text_prompt=""
    meta[n]['text_prompt']=text_prompt
    if n>10:
        break
save_path=json_path.replace('meta.json','meta_caption.json')
with open(save_path, 'w') as f:
    json.dump(meta, f, indent=4)
print(f"error count: {error_cnt}")
