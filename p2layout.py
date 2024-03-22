import openai
import time
from openai import AzureOpenAI
import json
import os
from tqdm import tqdm

PROMPT_1 = """You are an excellent image layout designer who designs images based on the user's intention for the background image, the images for each layer, and the placement of each layer on the background image. I will give you the user's intention and your task is to output the description of the background image, the description for each layer of images and the arrangement of the placement of each layer of images on the background image. The base image is 1456 wide and 1457 high. Therefore, left and right should not exceed 1456, top and bottom should not exceed 1457. The layers are given from low to high, so the bounding boxes of higher layers shouldn't be too big."""


client = AzureOpenAI(
    api_key="oaip_lDubqnzOrsOLoSJAEWLWTprxKsZhfezx",  
    api_version="2023-03-15-preview",
    azure_endpoint="https://gcrgpt4aoai9c.azurewebsites.net"
)
# oaip_lDubqnzOrsOLoSJAEWLWTprxKsZhfezx
#oaip_epiXDtSUSdYCAmXQuzGixfnaqCaFroSw

shot_1_input = """Create an advertisement for an Autumn Super Sale event offering up to 50% off. The ad should have an eye-catching design with autumn leaves, and maybe include a shopper to suggest the idea of buying and carrying bags. Include placeholder text for additional details."""
shot_1_output = """[{"background": "The layer is completely light yellow color."}, {"layer": 1, "category": "element", "caption": "The layer features a stylized illustration of a person walking, dressed in autumnal attire. Visible is an orange plaid coat over a pink garment, with a shoulder bag in dark green, carrying multiple shopping bags in the opposite hand. Below, beige boots with red laces are worn over white socks. The entire figure is set against a plain background, emphasizing the fashionable fall outfit and shopping theme.", "top_left": [855, 413], "bottom_right": [1455, 1393]}, {"layer": 2, "category": "element", "caption": "Three colorful shopping bags with visible handles and simplistic design details on their surface.", "top_left": [130, 1045], "bottom_right": [527, 1401]}, {"layer": 3, "category": "text", "caption": "Text \"Autumn\n\" in <color-1>. ", "top_left": [149, 210], "bottom_right": [736, 380]}, {"layer": 4, "category": "text", "caption": "Text \"super sale\n\" in <color-1>. ", "top_left": [148, 436], "bottom_right": [738, 549]}, {"layer": 5, "category": "text", "caption": "Text \"up to 50% off\n\" in <color-1>, <font-24>. ", "top_left": [150, 636], "bottom_right": [654, 687]}, {"layer": 6, "category": "text", "caption": "Text \"Lorem ipsum dolor sit amet, consectetur\nadipiscing elit. Ut diam tellus,\nelementum ac facilisis et\n\" in <color-1>, <font-24>. ", "top_left": [146, 764], "bottom_right": [771, 885]}]"""
shot_2_input = """Create an appealing graphic to promote a Happy Spring Sale, offering discounts of up to 50% off. The design incorporates floral and nature-themed elements to evoke the freshness of spring."""
shot_2_output = """[{"background": "The layer is completely light pink color."}, {"layer": 1, "category": "element", "caption": "The layer features botanical illustrations with a pale rose and green eucalyptus-like leaves against a light-colored background.", "top_left": [5, 1104], "bottom_right": [244, 1456]}, {"layer": 2, "category": "element", "caption": "A person dressed in a green jacket, white shirt, and striped pants with red shoes, depicted in a watercolor or pastel style illustration.", "top_left": [1102, 370], "bottom_right": [1416, 1456]}, {"layer": 3, "category": "element", "caption": "The layer features a person with their hands covering where the face would be. They have curly hair adorned with flowers, including a red rose and yellow blossoms. The person is dressed in a white shirt with button details on the sleeves, all set against a neutral off-white background.", "top_left": [174, 300], "bottom_right": [1031, 1157]}, {"layer": 4, "category": "element", "caption": "This layer features a series of large, orange, floral-like motifs with elegant white outlines, arranged in a flowing, organic pattern against a soft peach background. The design gives the impression of a continuous line drawing of abstract foliage, contributing a decorative, botanical aesthetic to the overall image.", "top_left": [1050, 1], "bottom_right": [1455, 348]}, {"layer": 5, "category": "element", "caption": "The layer displays a part of an abstract floral design with long, slender petals, outlined in a warm-toned hue against a soft beige background.", "top_left": [0, 1], "bottom_right": [335, 295]}, {"layer": 6, "category": "text", "caption": "Text \"Happy spring sale\n\" in <color-15>, <font-133>. ", "top_left": [225, 963], "bottom_right": [986, 1077]}, {"layer": 7, "category": "text", "caption": "Text \"Up to\n50% off\n\" in <color-15>, <font-0>. ", "top_left": [137, 473], "bottom_right": [273, 612]}]"""
shot_3_input = """Create a social media graphic to promote National Hobby Month with a focus on photography as a hobby. Include the quote \"Taking pictures is savoring life intensely\" and the social media handle @reallygreatsite."""
shot_3_output = """[{"background": "The layer is completely light yellow color."}, {"layer": 1, "category": "element", "caption": "The layer features a stylized illustration of a person wearing a hat, holding a camera with a large lens, indicative of someone with an interest in photography. The colors are muted with a limited palette, focusing on shades of beige, brown, and orange for a harmonious and simple visual appeal.", "top_left": [877, 630], "bottom_right": [1455, 1456]}, {"layer": 2, "category": "element", "caption": "A stylized picture of green mountains with snow-capped peaks, framed with a white border, giving the appearance of a Polaroid or instant photo.", "top_left": [65, 1096], "bottom_right": [287, 1313]}, {"layer": 3, "category": "element", "caption": "The layer consists of three overlapping photographic prints with illustrations. The top print showcases a mountainous landscape under a sunny sky, the middle print features lush green mountains with white peaks, and the bottom print presents a traditional building, possibly a pagoda, set against a soft red and green background. Each print has a white border, giving the impression of instant photographs or snapshots, which contributes to the hobby photography theme of the main image.", "top_left": [70, 75], "bottom_right": [264, 294]}, {"layer": 4, "category": "text", "caption": "Text \"Taking pictures is\nsavoring life\nintesely\n\" in <color-12>, <font-304>. ", "top_left": [287, 660], "bottom_right": [776, 895]}, {"layer": 5, "category": "text", "caption": "Text \"national hobby\nmonth\n\" in <color-3>, <font-45>. ", "top_left": [371, 198], "bottom_right": [1084, 399]}, {"layer": 6, "category": "text", "caption": "Text \"@reallygreatsite\n\" in <color-0>, <font-304>. ", "top_left": [350, 980], "bottom_right": [713, 1033]}]"""
shot_4_input = """Design an image to announce and remind students that every Wednesday is \"Wednesday Reading\", a day when all students are required to bring reading books. Include the website address @reallygreatsite for more information or queries."""
shot_4_output = """[{"background": "The layer is completely dark green color."}, {"layer": 1, "category": "element", "caption": "Two cartoon-style children sitting and lying down reading books, with the boy sitting cross-legged to the left and the girl lying on her stomach to the right. Both are wearing purple and white school uniforms and have happy expressions on their faces.", "top_left": [379, 239], "bottom_right": [1076, 805]}, {"layer": 2, "category": "element", "caption": "An open book with pages visibly spread out, depicted in white contour lines against a solid background color.", "top_left": [0, 256], "bottom_right": [300, 661]}, {"layer": 3, "category": "element", "caption": "The layer displays a stack of colorful books with details on the spines that suggest titles or decorations, indicating that the books are a significant visual aspect of the design meant to be in the foreground.", "top_left": [1137, 298], "bottom_right": [1455, 590]}, {"layer": 4, "category": "text", "caption": "Text \"@reallygreatsite\n\" in <color-2>. ", "top_left": [606, 1275], "bottom_right": [849, 1306]}, {"layer": 5, "category": "text", "caption": "Text \"Wednesday reading\n\" in <color-2>, <font-137>. ", "top_left": [196, 892], "bottom_right": [1258, 964]}, {"layer": 6, "category": "text", "caption": "Text \"Every Wednesday, all students are required to bring reading books\n\" in <color-2>, <font-59>. ", "top_left": [153, 1053], "bottom_right": [1303, 1191]}, {"layer": 7, "category": "text", "caption": "Text \"Announcement\n\" in <color-2>. ", "top_left": [631, 155], "bottom_right": [822, 179]}]"""
example = [(shot_1_input, shot_1_output), (shot_2_input, shot_2_output), (shot_3_input, shot_3_output), (shot_4_input, shot_4_output)]

def GPT_response(example, content, max_tokens=1000):
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": PROMPT_1
                }, {
                    "role": "user",
                    "content": f"{example[0][0]}"
                }, {
                    "role": "assistant",
                    "content": f"{example[0][1]}"
                }, 
                {
                    "role": "user",
                    "content": f"{example[1][0]}"
                }, 
                {
                    "role": "assistant",
                    "content": f"{example[1][1]}"
                }, 
                {
                    "role": "user",
                    "content": f"{example[2][0]}"
                }, {
                    "role": "assistant",
                    "content": f"{example[2][1]}"
                }, 
                {
                    "role": "user",
                    "content": f"{example[3][0]}"
                }, {
                    "role": "assistant",
                    "content": f"{example[3][1]}"
                },
                {
                    "role": "user",
                    "content": f"{content}"
                }
                ],
                temperature=0.2,
                # max_tokens=max_tokens
            )
            break
        except Exception as e:
            print(e)
            time.sleep(5)
    return response.choices[0].message.content


def intension2(intension):
    
    result = GPT_response(example, intension).replace("\"Text \"", "\"Text \\\"").replace("\" in", "\\\" in").replace("\n", "\\n")
    try: 
        print(json.loads(result))
        # for n, layer in enumerate(json.loads(result)):
        #     if n>1 and layer["bottom_right"][0]-layer["top_left"][0]>1450 and layer["bottom_right"][1]-layer["top_left"][1]>1450:
        #         raise Exception("Up Layer too big")
        return(json.loads(result))
    except:
        # return intension2(intension)
        return []
     
if __name__ == "__main__":
    intension2("""Create an invitation for a Valentine's Day party to be held on February 14, 2022, at 123 Anywhere St., Any City from 7 PM until the end of the event.""")


# [
#     {
#         'layer': 0, 'category': 'element', 'caption': 'The layer features a large, stylized heart shape in a deep red color, with a subtle gradient effect to give it depth and dimension. The heart is set against a soft pink background, creating a romantic and inviting atmosphere.', 'top_left': [0, 0], 'bottom_right': [1456, 1457]
#     },
#     {
#         'layer': 1, 'category': 'element', 'caption': 'A pair of champagne glasses with bubbles and a heart-shaped splash, symbolizing celebration and romance.', 'top_left': [1050, 300], 'bottom_right': [1455, 600]
#     },
#     {
#         'layer': 2, 'category': 'text', 'caption': 'Text "You\'re invited!\n" in <color-1>, <font-137>. ', 'top_left': [100, 100], 'bottom_right': [600, 200]
#     },
#     {
#         'layer': 3, 'category': 'text', 'caption': 'Text "Valentine\'s Day Party\n" in <color-1>, <font-137>. ', 'top_left': [100, 250], 'bottom_right': [800, 350]
#     },
#     {
#         'layer': 4, 'category': 'text', 'caption': 'Text "February 14, 2022\n" in <color-1>, <font-59>. ', 'top_left': [100, 400], 'bottom_right': [600, 500]
#     },
#     {
#         'layer': 5, 'category': 'text', 'caption': 'Text "7 PM until the end of the event\n" in <color-1>, <font-59>. ', 'top_left': [100, 550], 'bottom_right': [800, 650]
#     },
#     {
#         'layer': 6, 'category': 'text', 'caption': 'Text "123 Anywhere St., Any City\n" in <color-1>, <font-59>. ', 'top_left': [100, 700], 'bottom_right': [800, 800]
#     }
# ]