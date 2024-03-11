import numpy as np
from PIL import Image,ImageFont,ImageDraw,ImageColor
import math
import json
import os
import tqdm
MAIN_COLOR=["red","blue","green","yellow","purple","orange","pink","brown","black","white","gray"]
def color_modulate(color):
    color_type=color
    if color in MAIN_COLOR:
        sigma=10
    else:
        sigma=2
    color=ImageColor.getrgb(color)
    if color_type=="black" or color_type=="white":
        return color
    modulate=np.random.normal(0,sigma,3).astype(np.int)
    color=np.array(color)+modulate
    color=tuple(np.clip(color,0,255))
    return color

def draw_shape(shape,color,fill,size=1024,save_path=None,outline_color=None):
    color=color_modulate(color)
    if not fill and outline_color!=None:
        outline_color=color_modulate(outline_color)
    else:
        outline_color=None
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    width=np.random.randint(6,30)
    if shape=="square":        
        draw.rectangle([0,0,size,size],fill=color,outline=outline_color,width=width)
    elif shape=="rectangle":
        ratio=np.random.uniform(0.2,1)
        h=size
        w=int(size*ratio)
        left=(size-w)//2
        right=left+w
        draw.rectangle([left,0,right,h],fill=color,outline=outline_color,width=width)
    elif shape=="circle":
        draw.ellipse([0,0,size,size],fill=color,outline=outline_color,width=width)
    elif shape=="oval":
        ratio=np.random.uniform(0.2,1)
        h=size
        w=int(size*ratio)
        left=(size-w)//2
        right=left+w
        draw.ellipse([left,0,right,h],fill=color,outline=outline_color,width=width)
    elif shape=="semi-circle":
        draw.pieslice([0,0,size,size],0,180,fill=color,outline=outline_color,width=width)
    elif shape=="quarter-circle":
        draw.pieslice([0,0,size*2,size*2],180,270,fill=color,outline=outline_color,width=width)
    elif shape=="triangle":
        x1,y1=size//2,np.random.randint(0,size*2//3)
        x2,y2=0,size
        x3,y3=size,size
        draw.polygon([x1,y1,x2,y2,x3,y3],fill=color,outline=outline_color,width=width)
    elif shape=="diamond":
        ratio=np.random.uniform(0.2,1)
        h=size
        w=int(size*ratio)
        x1,y1=size//2,0
        x2,y2=(size-w)//2,size//2
        x3,y3=size//2,size
        x4,y4=(size+w)//2,size//2
        draw.polygon([x1,y1,x2,y2,x3,y3,x4,y4],fill=color,outline=outline_color,width=width)
    elif shape=="parallelogram":
        ratio=np.random.uniform(0.2,1)
        w=np.random.randint(size//2,size)
        h=int(size*ratio)
        w=int(size*ratio)
        x1,y1=0,(size+h)//2
        x2,y2=w,(size+h)//2
        x3,y3=size,(size-h)//2
        x4,y4=size-w,(size-h)//2
        draw.polygon([x1,y1,x2,y2,x3,y3,x4,y4],fill=color,outline=outline_color,width=width)
    elif shape=="trapezoid":
        ratio=np.random.uniform(0.2,1)
        w=np.random.randint(size//2,size)
        h=int(size*ratio)
        x1,y1=0,(size+h)//2
        x2,y2=size,(size+h)//2
        x3,y3=(size+w)//2,(size-h)//2
        x4,y4=(size-w)//2,(size-h)//2
        draw.polygon([x1,y1,x2,y2,x3,y3,x4,y4],fill=color,outline=outline_color,width=width)
    elif shape=="rounded rectangle":
        ratio=np.random.uniform(0.2,1)
        h=size
        w=int(size*ratio)
        left=(size-w)//2
        right=left+w
        draw.rounded_rectangle([left,0,right,h],fill=color,outline=outline_color,width=width,radius=np.random.randint(5,w//2))
    elif shape=="pentagon":
        ratio=np.random.uniform(0.8,1.2)
        if ratio>1:
            w=size
            h=int(size/ratio)
        else:
            h=size
            w=int(size*ratio)
        side = 5
        xy = [ 
            ((math.cos(th) + 1) * w//2, 
            (math.sin(th) + 1) * h//2 ) 
            for th in [i * (2 * math.pi) / side +math.pi/2 for i in range(side)] 
        ]   
        draw.polygon(xy, fill=color,outline=outline_color,width=width)
    elif shape=="hexagon":
        ratio=np.random.uniform(0.8,1.2)
        if ratio>1:
            w=size
            h=int(size/ratio)
        else:
            h=size
            w=int(size*ratio)
        side = 6
        start=np.random.randint(0,2)
        xy = [ 
            ((math.cos(th) + 1) * w//2, 
            (math.sin(th) + 1) * h//2 ) 
            for th in [i * (2 * math.pi) / side +start*math.pi/2 for i in range(side)] 
        ]   
        draw.polygon(xy, fill=color,outline=outline_color,width=width)
    else:
        print(shape)
        raise NotImplementedError
    if save_path!=None:
        img.save(save_path)

def normalize(p):
    p=np.array(p).astype(np.float)
    p=p/np.sum(p)
    return p
    
if __name__=="__main__":
    save_dir="/pyy/openseg_blob/yuyang/datasets/RPG/shape1k"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset_size=1000
    json_path=os.path.join(save_dir,"meta.json")
    SHAPES=["square","rectangle","circle","triangle","oval","pentagon","hexagon","diamond","parallelogram","trapezoid","rounded rectangle","semi-circle","quarter-circle"]
    shapes_p=[15,30,30,2,6,1,1,2,3,3,20,5,4]
    shapes_p=normalize(shapes_p)
    COLORS=["black","white","red","blue","green","yellow","purple","orange","pink","brown","gray","darkkhaki", "mediumblue", "wheat", "deeppink", "orangered", "lime", "olivedrab", "goldenrod", "violet", "lightcoral", "indianred", "navy", "gainsboro", "mistyrose", "skyblue", "springgreen", "slategray", "lightgoldenrodyellow", "lightyellow", "palegreen", "maroon", "dimgray", "hotpink", "sandybrown", "indigo", "papayawhip", "peru", "ivory", "peachpuff", "lemonchiffon", "lightsalmon", "gold", "lightblue", "khaki", "mediumaquamarine", "mediumslateblue", "tan", "moccasin", "cyan", "fuchsia", "darkgoldenrod", "paleturquoise", "yellowgreen", "linen", "mediumturquoise", "darkgreen", "darkorange", "palegoldenrod", "turquoise", "firebrick", "darkmagenta", "plum", "palevioletred", "seagreen", "mediumpurple", "saddlebrown", "lavender", "magenta", "teal", "deepskyblue", "darkturquoise", "tomato", "mediumseagreen", "snow", "darksalmon", "sienna", "slateblue", "lightgray", "honeydew", "darkviolet", "midnightblue", "oldlace", "forestgreen", "seashell", "darkseagreen", "mediumspringgreen", "ghostwhite", "lightgreen", "darkslategray", "olive", "dodgerblue", "lightskyblue", "mintcream", "lightseagreen", "navajowhite", "lightslategray", "lightpink", "salmon", "floralwhite", "steelblue", "darkslateblue", "greenyellow", "lavenderblush", "rosybrown", "darkcyan", "limegreen", "royalblue", "powderblue", "aqua", "darkblue", "silver", "mediumorchid", "mediumvioletred", "lightsteelblue", "lawngreen", "thistle", "orchid", "lightcyan", "coral", "darkorchid", "whitesmoke", "darkred", "darkolivegreen"]


    colors_p=[15 for i in range(len(MAIN_COLOR))]+[1 for i in range(len(COLORS)-len(MAIN_COLOR))]
    colors_p=normalize(colors_p)
    fill_p=0.7
    outline_p=colors_p
    in_p=[15 for i in range(len(MAIN_COLOR))]+[1 for i in range(len(COLORS)-len(MAIN_COLOR))]
    in_p[0]=1000
    in_p[1]=100
    in_p=normalize(in_p)
    meta=[]
    shapes=np.random.choice(SHAPES,dataset_size,p=shapes_p)
    colors=np.random.choice(COLORS,dataset_size,p=colors_p)
    fills=np.random.choice([True,False],dataset_size,p=[fill_p,1-fill_p])
    outline_colors=np.random.choice(COLORS,dataset_size,p=outline_p)
    for i in tqdm.tqdm(range(dataset_size)):
        item={}
        shape=shapes[i]
        color=colors[i]
        fill=fills[i]
        outline_color=outline_colors[i]
        if not fill:
            color=np.random.choice(COLORS,1,p=in_p)[0]

        
        save_path=os.path.join(save_dir,str(i)+".png")
        draw_shape(shape,color,fill,save_path=save_path,outline_color=outline_color)
        if color=="black" and fill:
            shape="square"
        item['shape']=shape
        item['color']=color
        item['fill']=bool(fill)
        item['outline_color']=outline_color
        item['path']=save_path
        meta.append(item)
    with open(json_path,"w") as f:
        json.dump(meta,f,ensure_ascii=False,indent=4)
    

    