from openai import AzureOpenAI
import base64
import requests
from PIL import Image
import io
 
   
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
 
# Path to your image
image_path = "/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/outputs/txt2img-images/2024-02-22/00000-4007541002.png"
 
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
 
ref = "girl playing trumpet"
text = "Write a sentence to describe this image. Focus on the scene, objects and movements. No more than 40 words. These are some reference words; if you find the reference words relevant to the image content, you can take them into consideration, otherwise, ignore the reference words. Reference words: {ref}".format(ref=ref)
print(text)
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
print(completion.model_dump_json(indent=2))
print(completion.choices[0].message.content)

