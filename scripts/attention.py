import math
from pprint import pprint
import ldm.modules.attention as atm
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, Resize  # Mask.
from xformers.ops import memory_efficient_attention
TOKENSCON = 77
TOKENS = 75

def db(self,text):
    if self.debug:
        print(text)

def main_forward(module,x,context,mask,divide,isvanilla = False,userpp = False,tokens=[],width = 64,height = 64,step = 0, isxl = False, negpip = None, inhr = None):
    USE_XFORMERS=False
    # Forward.

    if negpip:
        conds, contokens = negpip
        context = torch.cat((context,conds),1)

    h = module.heads
    if isvanilla: # SBM Ddim / plms have the context split ahead along with x.
        pass
    else: # SBM I think divide may be redundant.
        h = h // divide
    q = module.to_q(x)

    context = atm.default(context, x)
    k = module.to_k(context)
    v = module.to_v(context)

    
    if USE_XFORMERS:
        q = atm.rearrange(q, 'b n (h d)-> b n h d', h=h)
        k = atm.rearrange(k, 'b n (h d)-> b n h d', h=h)
        v = atm.rearrange(v, 'b n (h d)-> b n h d', h=h)
        out=memory_efficient_attention(q, k, v, mask, scale=module.scale)
        out = atm.rearrange(out, 'b n h d -> b n (h d)')
    else:
        q, k, v = map(lambda t: atm.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = atm.einsum('b i d, b j d -> b i j', q, k) * module.scale

        if negpip:
            conds, contokens = negpip
            if contokens:
                for contoken in contokens:
                    start = (v.shape[1]//77 - len(contokens)) * 77
                    v[:,start+1:start+contoken,:] = -v[:,start+1:start+contoken,:] 

        if atm.exists(mask):
            # mask = atm.rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = atm.repeat(mask, 'b j k-> (b h) j k', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        ## for prompt mode make basemask from attention maps

        global pmaskshw,pmasks

        if inhr and not hiresfinished: hiresscaler(height,width,attn)

        if userpp and step > 0:
            for b in range(attn.shape[0] // 8):
                if pmaskshw == []:
                    pmaskshw = [(height,width)]
                elif (height,width) not in pmaskshw:
                    pmaskshw.append((height,width))

                for t in tokens:
                    power = 4 if isxl else 1.2
                    add = attn[8*b:8*(b+1),:,t[0]:t[0]+len(t)]**power
                    add = torch.sum(add,dim = 2)
                    t = f"{t}-{b}"         
                    if t not in pmasks:
                        pmasks[t] = add
                    else:
                        if pmasks[t].shape[1] != add.shape[1]:
                            add = add.view(8,height,width)
                            add = F.resize(add,pmaskshw[0])
                            add = add.reshape_as(pmasks[t])

                        pmasks[t] = pmasks[t] + add

        out = atm.einsum('b i j, b j d -> b i d', attn, v)
        out = atm.rearrange(out, '(b h) n d -> b n (h d)', h=h)
    try:
        out = module.to_out(out)
    except:
        length=len(module.to_out)
        for i in range(length):
            out = module.to_out[i](out)

    return out

def main_debug_forward(module,x,context,mask,divide,isvanilla = False,userpp = False,tokens=[],width = 64,height = 64,step = 0, isxl = False, negpip = None, inhr = None):
    
    # Forward.

    if negpip:
        conds, contokens = negpip
        context = torch.cat((context,conds),1)

    h = module.heads
    if isvanilla: # SBM Ddim / plms have the context split ahead along with x.
        pass
    else: # SBM I think divide may be redundant.
        h = h // divide
    q = module.to_q(x)

    context = atm.default(context, x)
    k = module.to_k(context)
    v = module.to_v(context)

    q, k, v = map(lambda t: atm.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = atm.einsum('b i d, b j d -> b i j', q, k) * module.scale

    if negpip:
        conds, contokens = negpip
        if contokens:
            for contoken in contokens:
                start = (v.shape[1]//77 - len(contokens)) * 77
                v[:,start+1:start+contoken,:] = -v[:,start+1:start+contoken,:] 

    if atm.exists(mask):
        # mask = atm.rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = atm.repeat(mask, 'b j k -> (b h) j k', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim=-1)

    ## for prompt mode make basemask from attention maps

    global pmaskshw,pmasks

    if inhr and not hiresfinished: hiresscaler(height,width,attn)

    if userpp and step > 0:
        for b in range(attn.shape[0] // 8):
            if pmaskshw == []:
                pmaskshw = [(height,width)]
            elif (height,width) not in pmaskshw:
                pmaskshw.append((height,width))

            for t in tokens:
                power = 4 if isxl else 1.2
                add = attn[8*b:8*(b+1),:,t[0]:t[0]+len(t)]**power
                add = torch.sum(add,dim = 2)
                t = f"{t}-{b}"         
                if t not in pmasks:
                    pmasks[t] = add
                else:
                    if pmasks[t].shape[1] != add.shape[1]:
                        add = add.view(8,height,width)
                        add = F.resize(add,pmaskshw[0])
                        add = add.reshape_as(pmasks[t])

                    pmasks[t] = pmasks[t] + add

    out = atm.einsum('b i j, b j d -> b i d', attn, v)
    out = atm.rearrange(out, '(b h) n d -> b n (h d)', h=h)
    try:
        out = module.to_out(out)
    except:
        length=len(module.to_out)
        for i in range(length):
            out = module.to_out[i](out)

    return out

def hook_forwards(self, root_module: torch.nn.Module, remove=False):
    self.allow_selfattn_hook=True
    self.hooked = True if not remove else False
    self.restrict_selfattn_threshold=10.0*10.0/64/64
    self.masks={}
    self.selfattn_cnt=0
    for name, module in root_module.named_modules():
        if "attn2" in name:
            temp=0
        if "attn2" in name and (module.__class__.__name__ == "CrossAttention" or module.__class__.__name__ == "Attention"):
            module.forward = hook_forward(self, module)
            if remove:
                del module.forward
        # if self.allow_selfattn_hook and "attn1" in name and module.__class__.__name__ == "CrossAttention":
        #     module.original_forward = module.forward
        #     module.forward = hook_self_attn_forward(self, module)
        #     self.selfattn_cnt+=1
        #     if remove:
        #         del module.forward
    print(f"selfattn_cnt:{self.selfattn_cnt}")


################################################################################
##### Attention mode 

def hook_forward(self, module):
    def forward(x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        if self.debug:
            print("input : ", x.size())
            print("tokens : ", context.size())
            print("module : ", getattr(module, self.layer_name,None))
        if "conds" in self.log:
            if self.log["conds"] != context.size():
                self.log["conds2"] = context.size()
        else:
            self.log["conds"] = context.size()

        if self.xsize == 0: self.xsize = x.shape[1]
        if "input" in getattr(module, self.layer_name,""):
            if x.shape[1] > self.xsize:
                self.in_hr = True

        height = self.hr_h if self.in_hr and self.hr else self.h 
        width = self.hr_w if self.in_hr and self.hr else self.w

        xs = x.size()[1]
        scale = round(math.sqrt(height * width / xs))

        dsh = round(height / scale)
        dsw = round(width / scale)
        ha, wa = xs % dsh, xs % dsw
        if ha == 0:
            dsw = int(xs / dsh)
        elif wa == 0:
            dsh = int(xs / dsw)

        contexts = context.clone()
        

        # SBM Matrix mode.
        def matsepcalc(x,contexts,mask,pn,divide):
            db(self,f"in MatSepCalc")
            h_states = []
            xs = x.size()[1]
            (dsh,dsw) = split_dims(xs, height, width, self)
            
            if "Horizontal" in self.mode: # Map columns / rows first to outer / inner.
                dsout = dsw
                dsin = dsh
            elif "Vertical" in self.mode:
                dsout = dsh
                dsin = dsw
            # if pn:
            #     tll = self.pt
            # else:

            #     tll = self.nt
                # tll[0] = self.pt[0] if len(self.nt) == 1 else self.nt[0]
            tll = self.pt if pn else self.nt
            
            i = 0
            outb = None

            #for debug, check the demo in bboxes mode
            # self.use_layer = True
            # if not hasattr(self, "bboxes"):
                
            #     self.bboxes=[]
            #     sumout = 0
            #     for drow in self.aratios:
            #         sumin = 0
            #         for dcell in drow.cols:
            #             addout = 0
            #             addin = 0
            #             sumin = sumin + int(dsin*dcell.ed) - int(dsin*dcell.st)
            #             if dcell.ed >= 0.999:
            #                 addin = sumin - dsin
            #                 sumout = sumout + int(dsout*drow.ed) - int(dsout*drow.st)
            #                 if drow.ed >= 0.999:
            #                     addout = sumout - dsout
            #             if "Horizontal" in self.mode:
            #                 self.bboxes.append([int(dsh*drow.st) + addout,int(dsw*dcell.st) + addin,int(dsh*drow.ed),int(dsw*dcell.ed)])
            #             elif "Vertical" in self.mode:
            #                 self.bboxes.append([int(dsh*dcell.st) + addin,int(dsw*drow.st) + addout,int(dsh*dcell.ed),int(dsw*drow.ed)])

            #end of debug



            if height!=dsh or width!=dsw:
                bboxes=[]
                scale_h=float(dsh)/height
                scale_w=float(dsw)/width
                for bbox in self.bboxes:
                    bbox_resize=[int(bbox[0]*scale_h),int(bbox[1]*scale_w),int(bbox[2]*scale_h),int(bbox[3]*scale_w)]
                    if bbox_resize[0]>=bbox_resize[2]:
                        if bbox_resize[0]>0:
                            bbox_resize[0]-=1
                        else:   
                            bbox_resize[2]+=1
                    if bbox_resize[1]>=bbox_resize[3]:
                        if bbox_resize[1]>0:
                            bbox_resize[1]-=1
                        else:
                            bbox_resize[3]+=1  
                    #for test
                    if len(bbox)==8:
                        target_bbox_resize=[int(bbox[4]*scale_h),int(bbox[5]*scale_w),int(bbox[6]*scale_h),int(bbox[7]*scale_w)]
                        if target_bbox_resize[0]>=target_bbox_resize[2]:
                            if target_bbox_resize[0]>0:
                                target_bbox_resize[0]-=1
                            else:   
                                target_bbox_resize[2]+=1
                        if target_bbox_resize[1]>=target_bbox_resize[3]:
                            if target_bbox_resize[1]>0:
                                target_bbox_resize[1]-=1
                            else:
                                target_bbox_resize[3]+=1  
                        bbox_resize=bbox_resize+target_bbox_resize

                    #test end
                    bboxes.append(bbox_resize)
                
                    

            else:
                bboxes=self.bboxes

            
            
            #for debug, generate different layers
            debug=False
            if debug:
                i=0
                ox = torch.zeros_like(x).reshape(x.size()[0], dsh, dsw, x.size()[2])
                if self.use_layer:
                    for j in range(len(bboxes)):
                        
                        context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                        # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                        cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                        if cnet_ext > 0:
                            context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                            
                        negpip = negpipdealer(i,pn)

                        db(self,f"tokens : {tll[i][0]*TOKENSCON}-{tll[i][1]*TOKENSCON}")
                        out = main_forward(module, x, context, mask, divide, self.isvanilla,userpp = self.pn, step = self.step, isxl = self.isxl,negpip = negpip)
                        if len(self.nt) == 1 and not pn:
                            db(self,"return out for NP")
                            return out
                        out = out.reshape(1, dsh, dsw, out.size()[2]) # convert to main shape.
                        # ox[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:] = out[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:]
                        ox=out

                        i+=1


            else:
                if self.usebase:
                    context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                    # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                    cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                    if cnet_ext > 0:
                        context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    negpip = negpipdealer(i,pn)

                    i = i + 1

                    out = main_forward(module, x, context, mask, divide, self.isvanilla,userpp =True,step = self.step, isxl = self.isxl, negpip = negpip)

                    if len(self.nt) == 1 and not pn:
                        db(self,"return out for NP")
                        return out
                    # if self.usebase:
                    outb = out.clone()
                    outb = outb.reshape(outb.size()[0], dsh, dsw, outb.size()[2]) if "Ran" not in self.mode else outb
                    

                sumout = 0
                db(self,f"tokens : {tll},pn : {pn}")
                db(self,[r for r in self.aratios])
                if self.use_layer:
                    merge_mask=torch.zeros(x.size()[0], dsh, dsw, x.size()[2]).to(x.device)
                    merge_ratio=self.merge_ratio
                    text_mask=torch.zeros(x.size()[0], dsh, dsw, x.size()[2]).to(x.device)
                    ox = torch.zeros_like(x).reshape(x.size()[0], dsh, dsw, x.size()[2])
                    for j in range(len(bboxes)):
                        attention_mask=mask
                        if i in self.pglyph:
                            context=self.byt5_prompt_embeds[self.pglyph.index(i)]
                            real_length=len(torch.where(self.byt5_attention_masks[self.pglyph.index(i)])[0])
                            context=context[:,:real_length,:]
                            
                            # attention_mask=self.byt5_attention_masks[self.pglyph.index(i)].repeat(1,x.size()[1],1)
                            # i+=1
                            # continue
                        else:
                            context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                            # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                            cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                            if cnet_ext > 0:
                                context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                            if j!=0:
                                merge_mask[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:]=1
                            
                            
                        negpip = negpipdealer(i,pn)

                        db(self,f"tokens : {tll[i][0]*TOKENSCON}-{tll[i][1]*TOKENSCON}")

                        out = main_forward(module, x, context, mask, divide, self.isvanilla,userpp = self.pn, step = self.step, isxl = self.isxl,negpip = negpip)
                        if len(self.nt) == 1 and not pn:
                            db(self,"return out for NP")
                            return out
                        out = out.reshape(out.size()[0], dsh, dsw, out.size()[2]) # convert to main shape.
                        # Resize, put all the layer latent in the bbox area
                        # out = out[:,bboxes[j][4]:bboxes[j][6],bboxes[j][5]:bboxes[j][7],:]
                        # out=out.permute(0,3,1,2)
                        # from torchvision.transforms import Resize 
                        # torch_resize=Resize((bboxes[j][2]-bboxes[j][0],bboxes[j][3]-bboxes[j][1]), interpolation = InterpolationMode("nearest"))
                        # out = torch_resize(out)
                        # out=out.permute(0,2,3,1)
                        # ox[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:] = out
                        # Resize end
                        if i in self.pglyph:
                            ox[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:] = out[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:]
                            text_mask[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:]=1
                        else:
                            ox[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:] = torch.where(merge_mask[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:]==1,out[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:]*merge_ratio+ox[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:]*(1-merge_ratio),out[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:])

                            if j!=0:
                                merge_mask[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:]=1

                        i+=1

                    if self.usebase : 
                        ox = torch.where(text_mask==1,ox,ox * (1 - self.bratios[0][0]) + outb * self.bratios[0][0])




                else:
                    for drow in self.aratios:
                        v_states = []
                        sumin = 0
                        for dcell in drow.cols:
                            # Grabs a set of tokens depending on number of unrelated breaks.
                            context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                            # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                            cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                            if cnet_ext > 0:
                                context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                                
                            negpip = negpipdealer(i,pn)

                            db(self,f"tokens : {tll[i][0]*TOKENSCON}-{tll[i][1]*TOKENSCON}")
                            i = i + 1 + dcell.breaks
                            # if i >= contexts.size()[1]: 
                            #     indlast = True

                            out = main_forward(module, x, context, mask, divide, self.isvanilla,userpp = self.pn, step = self.step, isxl = self.isxl,negpip = negpip)
                            db(self,f" dcell.breaks : {dcell.breaks}, dcell.ed : {dcell.ed}, dcell.st : {dcell.st}")
                            if len(self.nt) == 1 and not pn:
                                db(self,"return out for NP")
                                return out
                            # Actual matrix split by region.
                            if "Ran" in self.mode:
                                v_states.append(out)
                                continue
                            
                            out = out.reshape(out.size()[0], dsh, dsw, out.size()[2]) # convert to main shape.
                            # if indlast:
                            addout = 0
                            addin = 0
                            sumin = sumin + int(dsin*dcell.ed) - int(dsin*dcell.st)
                            if dcell.ed >= 0.999:
                                addin = sumin - dsin
                                sumout = sumout + int(dsout*drow.ed) - int(dsout*drow.st)
                                if drow.ed >= 0.999:
                                    addout = sumout - dsout
                            if "Horizontal" in self.mode:
                                out = out[:,int(dsh*drow.st) + addout:int(dsh*drow.ed),
                                            int(dsw*dcell.st) + addin:int(dsw*dcell.ed),:]
                                if self.debug : print(f"{int(dsh*drow.st) + addout}:{int(dsh*drow.ed)},{int(dsw*dcell.st) + addin}:{int(dsw*dcell.ed)}")
                                if self.usebase : 
                                    # outb_t = outb[:,:,int(dsw*drow.st):int(dsw*drow.ed),:].clone()
                                    outb_t = outb[:,int(dsh*drow.st) + addout:int(dsh*drow.ed),
                                                    int(dsw*dcell.st) + addin:int(dsw*dcell.ed),:].clone()
                                    out = out * (1 - dcell.base) + outb_t * dcell.base
                            elif "Vertical" in self.mode: # Cols are the outer list, rows are cells.
                                out = out[:,int(dsh*dcell.st) + addin:int(dsh*dcell.ed),
                                        int(dsw*drow.st) + addout:int(dsw*drow.ed),:]
                                db(self,f"{int(dsh*dcell.st) + addin}:{int(dsh*dcell.ed)}-{int(dsw*drow.st) + addout}:{int(dsw*drow.ed)}")
                                if self.usebase : 
                                    # outb_t = outb[:,:,int(dsw*drow.st):int(dsw*drow.ed),:].clone()
                                    outb_t = outb[:,int(dsh*dcell.st) + addin:int(dsh*dcell.ed),
                                                int(dsw*drow.st) + addout:int(dsw*drow.ed),:].clone()
                                    out = out * (1 - dcell.base) + outb_t * dcell.base
                            db(self,f"sumin:{sumin},sumout:{sumout},dsh:{dsh},dsw:{dsw}")
                    
                            v_states.append(out)
                            if self.debug : 
                                for h in v_states:
                                    print(h.size())
                                    
                        if "Horizontal" in self.mode:
                            ox = torch.cat(v_states,dim = 2) # First concat the cells to rows.
                        elif "Vertical" in self.mode:
                            ox = torch.cat(v_states,dim = 1) # Cols first mode, concat to cols.
                        elif "Ran" in self.mode:
                            if self.usebase:
                                ox = outb * makerrandman(self.ranbase,dsh,dsw).view(-1, 1)
                            ox = torch.zeros_like(v_states[0])
                            for state, filter in zip(v_states, self.ransors):
                                filter = makerrandman(filter,dsh,dsw)
                                ox = ox + state * filter.view(-1, 1)
                            return ox

                        h_states.append(ox)
                    if "Horizontal" in self.mode:
                        ox = torch.cat(h_states,dim = 1) # Second, concat rows to layer.
                    elif "Vertical" in self.mode:
                        ox = torch.cat(h_states,dim = 2) # Or cols.
            ox = ox.reshape(x.size()[0],x.size()[1],x.size()[2]) # Restore to 3d source.  
            return ox

        def masksepcalc(x,contexts,mask,pn,divide):
            db(self,f"in MaskSepCalc")
            xs = x.size()[1]
            (dsh,dsw) = split_dims(xs, height, width, self)

            tll = self.pt if pn else self.nt
            
            # Base forward.
            i = 0
            outb = None
            if self.usebase:
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)

                negpip = negpipdealer(i,pn) 

                i = i + 1
                out = main_forward(module, x, context, mask, divide, self.isvanilla, isxl = self.isxl, negpip = negpip)

                if len(self.nt) == 1 and not pn:
                    db(self,"return out for NP")
                    return out
                # if self.usebase:
                outb = out.clone()
                outb = outb.reshape(outb.size()[0], dsh, dsw, outb.size()[2]) 

            db(self,f"tokens : {tll},pn : {pn}")
            
            ox = torch.zeros_like(x)
            ox = ox.reshape(ox.shape[0], dsh, dsw, ox.shape[2])
            ftrans = Resize((dsh, dsw), interpolation = InterpolationMode("nearest"))
            for rmask in self.regmasks:
                # Need to delay mask tensoring so it's on the correct gpu.
                # Dunno if caching masks would be an improvement.
                if self.usebase:
                    bweight = self.bratios[0][i - 1]
                # Resize mask to current dims.
                # Since it's a mask, we prefer a binary value, nearest is the only option.
                rmask2 = ftrans(rmask.reshape([1, *rmask.shape])) # Requires dimensions N,C,{d}.
                rmask2 = rmask2.reshape(1, dsh, dsw, 1)
                
                # Grabs a set of tokens depending on number of unrelated breaks.
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    
                db(self,f"tokens : {tll[i][0]*TOKENSCON}-{tll[i][1]*TOKENSCON}")
                i = i + 1
                # if i >= contexts.size()[1]: 
                #     indlast = True
                out = main_forward(module, x, context, mask, divide, self.isvanilla, isxl = self.isxl)
                if len(self.nt) == 1 and not pn:
                    db(self,"return out for NP")
                    return out
                    
                out = out.reshape(out.size()[0], dsh, dsw, out.size()[2]) # convert to main shape.
                if self.usebase:
                    out = out * (1 - bweight) + outb * bweight
                ox = ox + out * rmask2

            if self.usebase:
                rmask = self.regbase
                rmask2 = ftrans(rmask.reshape([1, *rmask.shape])) # Requires dimensions N,C,{d}.
                rmask2 = rmask2.reshape(1, dsh, dsw, 1)
                ox = ox + outb * rmask2
            ox = ox.reshape(x.size()[0],x.size()[1],x.size()[2]) # Restore to 3d source.  
            return ox

        def promptsepcalc(x, contexts, mask, pn,divide):
            h_states = []

            tll = self.pt if pn else self.nt
            db(self,f"in PromptSepCalc")
            db(self,f"tokens : {tll},pn : {pn}")

            for i, tl in enumerate(tll):
                context = contexts[:, tl[0] * TOKENSCON : tl[1] * TOKENSCON, :]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                
                db(self,f"tokens3 : {tl[0]*TOKENSCON}-{tl[1]*TOKENSCON}")
                db(self,f"extra-tokens : {cnet_ext}")

                userpp = self.pn and i == 0 and self.pfirst

                negpip = negpipdealer(self.condi,pn) if "La" in self.calc else negpipdealer(i,pn)

                out = main_forward(module, x, context, mask, divide, self.isvanilla, userpp = userpp, width = dsw, height = dsh,
                                                 tokens = self.pe, step = self.step, isxl = self.isxl, negpip = negpip, inhr = self.in_hr)

                if (len(self.nt) == 1 and not pn) or ("Pro" in self.mode and "La" in self.calc):
                    db(self,"return out for NP or Latent")
                    return out

                db(self,[scale, dsh, dsw, dsh * dsw, x.size()[1]])

                if i == 0:
                    outb = out.clone()
                    continue
                else:
                    h_states.append(out)

            if self.debug:
                for h in h_states :
                    print(f"divided : {h.size()}")
                print(pmaskshw)

            if pmaskshw == []:
                return outb

            ox = outb.clone() if self.ex else outb * 0

            db(self,[pmaskshw,maskready,(dsh,dsw) in pmaskshw and maskready,len(pmasksf),len(h_states)])

            if (dsh,dsw) in pmaskshw and maskready:
                depth = pmaskshw.index((dsh,dsw))
                maskb = None
                for masks , state in zip(pmasksf.values(),h_states):
                    mask = masks[depth]
                    masked = torch.multiply(state, mask)
                    if self.ex:
                        ox = torch.where(masked !=0 , masked, ox)
                    else:
                        ox = ox + masked
                    maskb = maskb + mask if maskb is not None else mask
                maskb = 1 - maskb
                if not self.ex : ox = ox + torch.multiply(outb, maskb)
                return ox
            else:
                return outb

        if self.eq:
            db(self,"same token size and divisions")
            if "Mas" in self.mode:
                ox = masksepcalc(x, contexts, mask, True, 1)
            elif "Pro" in self.mode:
                ox = promptsepcalc(x, contexts, mask, True, 1)
            else:
                ox = matsepcalc(x, contexts, mask, True, 1)
        elif x.size()[0] == 1 * self.batch_size:
            db(self,"different tokens size")
            if "Mas" in self.mode:
                ox = masksepcalc(x, contexts, mask, self.pn, 1)
            elif "Pro" in self.mode:
                ox = promptsepcalc(x, contexts, mask, self.pn, 1)
            else:
                ox = matsepcalc(x, contexts, mask, self.pn, 1)
        else:
            db(self,"same token size and different divisions")
            # SBM You get 2 layers of x, context for pos/neg.
            # Each should be forwarded separately, pairing them up together.
            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                nx, px = x.chunk(2)
                conn,conp = contexts.chunk(2)
            else:
                px, nx = x.chunk(2)
                conp,conn = contexts.chunk(2)
            if "Mas" in self.mode:
                opx = masksepcalc(px, conp, mask, True, 2)
                onx = masksepcalc(nx, conn, mask, False, 2)
            elif "Pro" in self.mode:
                opx = promptsepcalc(px, conp, mask, True, 2)
                onx = promptsepcalc(nx, conn, mask, False, 2)
            else:
                # SBM I think division may have been an incorrect patch.
                # But I'm not sure, haven't tested beyond DDIM / PLMS.
                opx = matsepcalc(px, conp, mask, True, 2)
                onx = matsepcalc(nx, conn, mask, False, 2)
            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                ox = torch.cat([onx, opx])
            else:
                ox = torch.cat([opx, onx])  

        self.count += 1

        limit =70 if self.isxl else 16

        if self.count == limit:
            self.pn = not self.pn
            self.count = 0
            self.pfirst = False
            self.condi += 1
        db(self,f"output : {ox.size()}")
        return ox

    return forward


def hook_self_attn_forward(self, module):
    def forward(x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        if self.xsize == 0: self.xsize = x.shape[1]
        if "input" in getattr(module, self.layer_name,""):
            if x.shape[1] > self.xsize:
                self.in_hr = True

        height = self.hr_h if self.in_hr and self.hr else self.h 
        width = self.hr_w if self.in_hr and self.hr else self.w

        xs = x.size()[1]
        scale = round(math.sqrt(height * width / xs))

        dsh = round(height / scale)
        dsw = round(width / scale)
        ha, wa = xs % dsh, xs % dsw
        if ha == 0:
            dsw = int(xs / dsh)
        elif wa == 0:
            dsh = int(xs / dsw)
        if dsh!=64 or dsw!=64:
            return module.original_forward(x, context=context, mask=None, additional_tokens=additional_tokens, n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self)
        if str(dsh) in self.masks:
            mask= self.masks[str(dsh)].to(x.device)
            return module.original_forward(x, context=context, mask=mask, additional_tokens=additional_tokens, n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self)
        else:
            mask = torch.ones(x.size()[0],x.size()[1],x.size()[1]).to(x.device)
            if self.bboxes[0][2]!=dsh or self.bboxes[0][3]!=dsw:
                bboxes=[]
                scale_h=float(dsh)/self.bboxes[0][2]
                scale_w=float(dsw)/self.bboxes[0][3]
                for bbox in self.bboxes:
                    bbox_resize=[int(bbox[0]*scale_h),int(bbox[1]*scale_w),int(bbox[2]*scale_h),int(bbox[3]*scale_w)]
                    if bbox_resize[0]>=bbox_resize[2]:
                        if bbox_resize[0]>0:
                            bbox_resize[0]-=1
                        else:   
                            bbox_resize[2]+=1
                    if bbox_resize[1]>=bbox_resize[3]:
                        if bbox_resize[1]>0:
                            bbox_resize[1]-=1
                        else:
                            bbox_resize[3]+=1  
                    bboxes.append(bbox_resize)
            else:
                bboxes=self.bboxes

            for bbox in self.bboxes:
                if float((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))/dsh/dsw>self.restrict_selfattn_threshold:
                    continue
                bbox_tool=torch.zeros(dsh,dsw)
                bbox_tool[bbox[0]:bbox[2],bbox[1]:bbox[3]]=1
                bbox_tool=bbox_tool.view(dsh*dsw)
                bbox_tool=torch.nonzero(bbox_tool).view(-1)
                mask[:,bbox_tool.tolist(),:]=0
                mask[:,bbox_tool.tolist(),bbox_tool.tolist()]=1
            mask=mask.to(x.device).bool()
            self.masks[str(dsh)] = mask
            return module.original_forward(x, context=context, mask=mask, additional_tokens=additional_tokens, n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self)
    return forward

def hook_debug_forward(self, module):
    def forward(x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        if self.debug:
            print("input : ", x.size())
            print("tokens : ", context.size())
            print("module : ", getattr(module, self.layer_name,None))
        if "conds" in self.log:
            if self.log["conds"] != context.size():
                self.log["conds2"] = context.size()
        else:
            self.log["conds"] = context.size()

        if self.xsize == 0: self.xsize = x.shape[1]
        if "input" in getattr(module, self.layer_name,""):
            if x.shape[1] > self.xsize:
                self.in_hr = True

        height = self.hr_h if self.in_hr and self.hr else self.h 
        width = self.hr_w if self.in_hr and self.hr else self.w

        xs = x.size()[1]
        scale = round(math.sqrt(height * width / xs))

        dsh = round(height / scale)
        dsw = round(width / scale)
        ha, wa = xs % dsh, xs % dsw
        if ha == 0:
            dsw = int(xs / dsh)
        elif wa == 0:
            dsh = int(xs / dsw)

        contexts = context.clone()
        

        # SBM Matrix mode.
        def matsepcalc(x,contexts,mask,pn,divide):
            db(self,f"in MatSepCalc")
            h_states = []
            xs = x.size()[1]
            (dsh,dsw) = split_dims(xs, height, width, self)
            
            if "Horizontal" in self.mode: # Map columns / rows first to outer / inner.
                dsout = dsw
                dsin = dsh
            elif "Vertical" in self.mode:
                dsout = dsh
                dsin = dsw
            # if pn:
            #     tll = self.pt
            # else:

            #     tll = self.nt
                # tll[0] = self.pt[0] if len(self.nt) == 1 else self.nt[0]
            tll = self.pt if pn else self.nt
            
            i = 0
            outb = None



            if self.bboxes[0][2]!=dsh or self.bboxes[0][3]!=dsw:
                bboxes=[]
                scale_h=float(dsh)/self.bboxes[0][2]
                scale_w=float(dsw)/self.bboxes[0][3]
                for bbox in self.bboxes:
                    bbox_resize=[int(bbox[0]*scale_h),int(bbox[1]*scale_w),int(bbox[2]*scale_h),int(bbox[3]*scale_w)]
                    if bbox_resize[0]>=bbox_resize[2]:
                        if bbox_resize[0]>0:
                            bbox_resize[0]-=1
                        else:   
                            bbox_resize[2]+=1
                    if bbox_resize[1]>=bbox_resize[3]:
                        if bbox_resize[1]>0:
                            bbox_resize[1]-=1
                        else:
                            bbox_resize[3]+=1  
                    #for test
                    if len(bbox)==8:
                        target_bbox_resize=[int(bbox[4]*scale_h),int(bbox[5]*scale_w),int(bbox[6]*scale_h),int(bbox[7]*scale_w)]
                        if target_bbox_resize[0]>=target_bbox_resize[2]:
                            if target_bbox_resize[0]>0:
                                target_bbox_resize[0]-=1
                            else:   
                                target_bbox_resize[2]+=1
                        if target_bbox_resize[1]>=target_bbox_resize[3]:
                            if target_bbox_resize[1]>0:
                                target_bbox_resize[1]-=1
                            else:
                                target_bbox_resize[3]+=1  
                        bbox_resize=bbox_resize+target_bbox_resize

                    #test end
                    bboxes.append(bbox_resize)
                
                    

            else:
                bboxes=self.bboxes


            if self.usebase:
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    
                negpip = negpipdealer(i,pn)

                i = i + 1

                out = main_debug_forward(module, x, context, mask, divide, self.isvanilla,userpp =True,step = self.step, isxl = self.isxl, negpip = negpip)

                if len(self.nt) == 1 and not pn:
                    db(self,"return out for NP")
                    return out
                # if self.usebase:
                outb = out.clone()
                outb = outb.reshape(outb.size()[0], dsh, dsw, outb.size()[2]) if "Ran" not in self.mode else outb
            #使用mask方式添加glyph control
            encoder_hidden_states=contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
            cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
            if cnet_ext > 0:
                encoder_hidden_states = torch.cat([encoder_hidden_states,contexts[:,-cnet_ext:,:]],dim = 1)
            glyph_encoder_hidden_states=[]
            bbox=[]
            for n,t in enumerate(self.byt5_prompt_embeds):
                glyph_encoder_hidden_states.append(t)
                bbox.append(bboxes[self.pglyph[n]-1] if self.usebase else bboxes[self.pglyph[n]])
            bg_attn_mask=torch.ones((1,x.size()[1],encoder_hidden_states.size(1)),device=encoder_hidden_states.device).reshape(1,dsh,dsw,encoder_hidden_states.size(1))
            for b in bbox:
                bg_attn_mask[:,b[0]:b[2],b[1]:b[3],:]=0
            bg_attn_mask=bg_attn_mask.reshape(1,x.size()[1],encoder_hidden_states.size(1))
            glyph_attn_mask=[]
            for n, glyph in enumerate(glyph_encoder_hidden_states):
                glyph_attn_mask.append(torch.zeros((1,x.size()[1],glyph.size(1)),device=encoder_hidden_states.device).reshape(1,dsh,dsw,glyph.size(1)))
                glyph_attn_mask[n][:,bbox[n][0]:bbox[n][2],bbox[n][1]:bbox[n][3],:]=1
                glyph_attn_mask[n]=glyph_attn_mask[n].reshape(1,x.size()[1],glyph.size(1))
            glyph_encoder_hidden_states=torch.cat(glyph_encoder_hidden_states,dim=1)
            glyph_attn_mask=torch.cat(glyph_attn_mask,dim=-1)
            content=torch.cat([encoder_hidden_states, glyph_encoder_hidden_states], dim=1)
            mask=torch.cat([bg_attn_mask, glyph_attn_mask], dim=-1).bool()
            out = main_debug_forward(module, x, content, mask, divide, self.isvanilla,userpp = self.pn, step = self.step, isxl = self.isxl,negpip = negpip)
            return out




            # sumout = 0
            # db(self,f"tokens : {tll},pn : {pn}")
            # db(self,[r for r in self.aratios])
            # if self.use_layer:
            #     ox = torch.zeros_like(x).reshape(x.size()[0], dsh, dsw, x.size()[2])
            #     first_glyph_layer=False
            #     for j in range(len(bboxes)):
            #         if i in self.pglyph:
            #             context=self.byt5_prompt_embeds[self.pglyph.index(i)]
            #             first_glyph_layer=True
            #             # i+=1
            #             # continue
            #         else:
            #             context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
            #             # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
            #             cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
            #             if cnet_ext > 0:
            #                 context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
            #         if j!=0 and not first_glyph_layer: 
            #             i+=1
            #             continue
            #         negpip = negpipdealer(i,pn)

            #         db(self,f"tokens : {tll[i][0]*TOKENSCON}-{tll[i][1]*TOKENSCON}")
            #         out = main_debug_forward(module, x, context, mask, divide, self.isvanilla,userpp = self.pn, step = self.step, isxl = self.isxl,negpip = negpip)
            #         if len(self.nt) == 1 and not pn:
            #             db(self,"return out for NP")
            #             return out
            #         out = out.reshape(out.size()[0], dsh, dsw, out.size()[2]) # convert to main shape.
            #         # Resize, put all the layer latent in the bbox area
            #         # out = out[:,bboxes[j][4]:bboxes[j][6],bboxes[j][5]:bboxes[j][7],:]
            #         # out=out.permute(0,3,1,2)
            #         # from torchvision.transforms import Resize 
            #         # torch_resize=Resize((bboxes[j][2]-bboxes[j][0],bboxes[j][3]-bboxes[j][1]), interpolation = InterpolationMode("nearest"))
            #         # out = torch_resize(out)
            #         # out=out.permute(0,2,3,1)
            #         # ox[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:] = out
            #         # Resize end
            #         ox[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:] = out[:,bboxes[j][0]:bboxes[j][2],bboxes[j][1]:bboxes[j][3],:]

            #         i+=1
            #         if first_glyph_layer:
            #             break

            #     if self.usebase : 
            #         ox = ox * (1 - self.bratios[0][0]) + outb * self.bratios[0][0]



            #     else:
            #         ox=None
            # ox = ox.reshape(x.size()[0],x.size()[1],x.size()[2]) # Restore to 3d source.  
            # return ox

        
        if self.eq:
            db(self,"same token size and divisions")
            ox = matsepcalc(x, contexts, mask, True, 1)
        elif x.size()[0] == 1 * self.batch_size:
            ox = matsepcalc(x, contexts, mask, self.pn, 1)
        else:
            db(self,"same token size and different divisions")
            # SBM You get 2 layers of x, context for pos/neg.
            # Each should be forwarded separately, pairing them up together.
            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                nx, px = x.chunk(2)
                conn,conp = contexts.chunk(2)
            else:
                px, nx = x.chunk(2)
                conp,conn = contexts.chunk(2)

            # SBM I think division may have been an incorrect patch.
            # But I'm not sure, haven't tested beyond DDIM / PLMS.
            opx = matsepcalc(px, conp, mask, True, 2)
            onx = matsepcalc(nx, conn, mask, False, 2)
            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                ox = torch.cat([onx, opx])
            else:
                ox = torch.cat([opx, onx])  

        self.count += 1

        limit =70 if self.isxl else 16

        if self.count == limit:
            self.pn = not self.pn
            self.count = 0
            self.pfirst = False
            self.condi += 1
        db(self,f"output : {ox.size()}")
        return ox

    return forward
         


def split_dims(xs, height, width, self = None):
    """Split an attention layer dimension to height + width.
    
    Originally, the estimate was dsh = sqrt(hw_ratio*xs),
    rounding to the nearest value. But this proved inaccurate.
    What seems to be the actual operation is as follows:
    - Divide h,w by 8, rounding DOWN. 
      (However, webui forces dims to be divisible by 8 unless set explicitly.)
    - For every new layer (of 4), divide both by 2 and round UP (then back up)
    - Multiply h*w to yield xs.
    There is no inverse function to this set of operations,
    so instead we mimic them sans the multiplication part with orig h+w.
    The only alternative is brute forcing integer guesses,
    which might be inaccurate too.
    No known checkpoints follow a different system of layering,
    but it's theoretically possible. Please report if encountered.
    """
    # OLD METHOD.
    # scale = round(math.sqrt(height*width/xs))
    # dsh = round_dim(height, scale)
    # dsw = round_dim(width, scale) 
    scale = math.ceil(math.log2(math.sqrt(height * width / xs)))
    dsh = repeat_div(height,scale)
    dsw = repeat_div(width,scale)
    if xs > dsh * dsw and hasattr(self,"nei_multi"):
        dsh, dsw = self.nei_multi[1], self.nei_multi[0] 
        while dsh*dsw != xs:
            dsh, dsw = dsh//2, dsw//2

    if self is not None:
        if self.debug : print(scale,dsh,dsw,dsh*dsw,xs, height, width)

    return dsh,dsw

def repeat_div(x,y):
    """Imitates dimension halving common in convolution operations.
    
    This is a pretty big assumption of the model,
    but then if some model doesn't work like that it will be easy to spot.
    """
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x

#################################################################################
##### for Prompt mode
pmasks = {}              #maked from attention maps
pmaskshw =[]            #height,width set of u-net blocks
pmasksf = {}             #maked from pmasks for regions
maskready = False
hiresfinished = False

def reset_pmasks(self): # init parameters in every batch
    global pmasks, pmaskshw, pmasksf, maskready, hiresfinished, pmaskshw_o
    self.step = 0
    pmasks = {}
    pmaskshw =[]
    pmaskshw_o =[]
    pmasksf = {}
    maskready = False
    hiresfinished = False
    self.x = None
    self.rebacked = False

def savepmasks(self,processed):
    for mask ,th in zip(pmasks.values(),self.th):
        img, _ , _= makepmask(mask, self.h, self.w,th, self.step)
        processed.images.append(img)
    return processed

def hiresscaler(new_h,new_w,attn):
    global pmaskshw,pmasks,pmasksf,pmaskshw_o, hiresfinished
    nset = (new_h,new_w)
    (old_h, old_w) = pmaskshw[0]
    if new_h > pmaskshw[0][0]:
        pmaskshw_o = pmaskshw.copy()
        del pmaskshw
        pmaskshw = [nset]
        hiresmask(pmasks,old_h, old_w, new_h, new_w,at = attn[:,:,0])
        hiresmask(pmasksf,old_h, old_w, new_h, new_w,i = 0)
    if nset not in pmaskshw:
        index = len(pmaskshw)
        pmaskshw.append(nset)
        old_h, old_w = pmaskshw_o[index]
        hiresmask(pmasksf,old_h, old_w, new_h, new_w,i = index)
        if index == 3: hiresfinished = True

def hiresmask(masks,oh,ow,nh,nw,at = None,i = None):
    for key in masks.keys():
        mask = masks[key] if i is None else masks[key][i]
        mask = mask.view(8 if i is None else 1,oh,ow)
        mask = F.resize(mask,(nh,nw))
        mask = mask.reshape_as(at) if at is not None else mask.reshape(1,mask.shape[1] * mask.shape[2],1)
        if i is None:
            masks[key] = mask
        else:
            masks[key][i] = mask

def makepmask(mask, h, w, th, step, bratio = 1): # make masks from attention cache return [for preview, for attention, for Latent]
    th = th - step * 0.005
    bratio = 1 - bratio
    mask = torch.mean(mask,dim=0)
    mask = mask / mask.max().item()
    mask = torch.where(mask > th ,1,0)
    mask = mask.float()
    mask = mask.view(1,pmaskshw[0][0],pmaskshw[0][1]) 
    img = torchvision.transforms.functional.to_pil_image(mask)
    img = img.resize((w,h))
    mask = F.resize(mask,(h,w),interpolation=F.InterpolationMode.NEAREST)
    lmask = mask
    mask = mask.reshape(h*w)
    mask = torch.where(mask > 0.1 ,1,0)
    return img,mask * bratio , lmask * bratio

def makerrandman(mask, h, w, latent = False): # make masks from attention cache return [for preview, for attention, for Latent]
    mask = mask.float()
    mask = mask.view(1,mask.shape[0],mask.shape[1]) 
    img = torchvision.transforms.functional.to_pil_image(mask)
    img = img.resize((w,h))
    mask = F.resize(mask,(h,w),interpolation=F.InterpolationMode.NEAREST)
    if latent: return mask
    mask = mask.reshape(h*w)
    mask = torch.round(mask).long()
    return mask

def negpipdealer(i,pn):
    negpip = None
    from modules.scripts import scripts_txt2img
    for script in scripts_txt2img.alwayson_scripts:
        if "negpip.py" in script.filename:
            negpip = script

    if negpip:
        conds = negpip.conds if pn else negpip.unconds
        tokens = negpip.contokens if pn else negpip.untokens
        if conds and len(conds) >= i + 1:
            if conds[i] is not None:
                return [conds[i],tokens[i]]
        else:
            return None
    else:
        return None