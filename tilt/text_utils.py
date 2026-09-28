import abc

import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List
import torch.nn.functional as F
from .gaussian_smoothing import GaussianSmoothing
from matplotlib import pyplot as plt
import os



def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img

#! COPY
def build_normal(u_x, u_y, d_x, d_y, step, device):
    x, y = torch.meshgrid(torch.linspace(0,1,step), torch.linspace(0,1,step))
    x = x.to(device)
    y = y.to(device)
    mean_x = u_x / step
    mean_y = u_y / step
    std_x = d_x / step
    std_y = d_y / step
    out_prob = (1/2/torch.pi/std_x/std_y)*torch.exp(-1/2*(torch.square((x-mean_x)/std_x)+torch.square((y-mean_y)/std_y)))
    return out_prob
#! COPY
def uniq_masks(all_masks, zero_masks=None, scale=1.0):
    uniq_masks = torch.stack(all_masks)
    # num = all_masks.shape[0]
    uniq_mask = torch.argmax(uniq_masks, dim=0)
    if zero_masks is None:
        all_masks = [((uniq_mask==i)*mask*scale).float().clamp(0, 1.0) for i, mask in enumerate(all_masks)]
    else:
        all_masks = [((uniq_mask==i)*mask*scale).float().clamp(0, 1.0) for i, mask in enumerate(zero_masks)]

    return all_masks

#! COPY
def build_masks(bboxes, size, mask_mode="gaussian_zero_one", focus_rate=1.0, device='cpu'):
    all_masks = []
    zero_masks = []
    for bbox in bboxes:
        x0,y0,spread_x,spread_y = bbox
        mask = build_normal(y0, x0, spread_y, spread_x, size, device)
        zero_mask = torch.zeros_like(mask)
        # zero_mask[int(y0 * size):min(int(y1 * size)+1, size), int(x0 * size):min(int(x1 * size)+1, size)] = 1.0
        zero_masks.append(zero_mask)
        all_masks.append(mask)
    if mask_mode == 'zero_one':
        return zero_masks
    elif mask_mode == 'gaussian':
        all_masks = uniq_masks(all_masks, scale=focus_rate)
        return all_masks
    elif mask_mode == 'gaussian_zero_one':
        all_masks = uniq_masks(all_masks, zero_masks, scale=focus_rate)
        return all_masks
    else:
        raise ValueError("Not supported mask_mode.")
    
def centroid_from_mask(mask: torch.Tensor):
    # mask: H x W (float, non-negative)
    H, W = mask.shape
    ys = torch.arange(0, H, dtype=mask.dtype, device=mask.device).view(H, 1)
    xs = torch.arange(0, W, dtype=mask.dtype, device=mask.device).view(1, W)
    mass = mask.sum()
    if mass == 0:
        return torch.tensor([W/2.0, H/2.0], device=mask.device)  # fallback center
    x_cent = (mask * xs).sum() / mass
    y_cent = (mask * ys).sum() / mass
    return torch.stack([x_cent, y_cent])  # (x, y)

def centroid_and_spread(mask: torch.Tensor, eps: float = 1e-8):
    """
    Compute weighted centroid and 2x2 covariance of a mask (H x W).
    Returns (x_cent, y_cent, cov_matrix) where x,y are in pixel coordinates.
    """
    H, W, _ = mask.shape
    device = mask.device
    dtype = mask.dtype

    ys = torch.arange(0, H, device=device, dtype=dtype).view(H, 1)
    xs = torch.arange(0, W, device=device, dtype=dtype).view(1, W)

    mass = mask.sum()
    m = mask.squeeze()
    if mass <= eps:
        # fallback: center of image, tiny covariance
        x_cent = (W - 1) / 2.0
        y_cent = (H - 1) / 2.0
        cov = torch.tensor([[1e-3, 0.0], [0.0, 1e-3]], device=device, dtype=dtype)
        return x_cent, y_cent, cov

    x_cent = (m * xs).sum() / mass
    y_cent = (m * ys).sum() / mass

    dx = xs - x_cent
    dy = ys - y_cent

    x_var = (m * (dx ** 2)).sum() / mass
    y_var = (m * (dy ** 2)).sum() / mass
    xy_cov = (m * (dx * dy)).sum() / mass

    cov = torch.tensor([[x_var, xy_cov], [xy_cov, y_var]], device=device, dtype=dtype)
    return x_cent.item(), y_cent.item(), (x_var, y_var)


class MoLECrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.bboxes = []
        self.soft_mask_rate = 0.2
        self.current_timestep = None  # Add timestep tracking
        self.viz = False  # Whether to visualize masks
    

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)
        # (Added): Updated for later diffusers versions to add batch_size and dtype
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, hidden_states.dtype)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        cond_attention_probs = attention_probs
        
        if not is_cross and self.bboxes: # If it is not cross, it is self attention
            size = int(np.sqrt(sequence_length))
            all_masks = build_masks(self.bboxes, size, "gaussian", device=hidden_states.device)
            for mask_idx, img_mask in enumerate(all_masks):
                if img_mask.sum() <= 0:
                    continue
                
                img_mask = img_mask.reshape(sequence_length)
                mask_index = img_mask.nonzero().squeeze(-1)
                mask = torch.ones(sequence_length, sequence_length).to(hidden_states.device)

                mask[:, mask_index] = mask[:, mask_index] * img_mask.unsqueeze(-1)
                # save mask visualization with timestep and concept index for correlation
                if self.viz:
                    step = getattr(self.attnstore, "cur_step", 0)
                    t = self.current_timestep if self.current_timestep is not None else step
                    
                    # reshape back to image grid
                    vis_mask = img_mask.reshape(size, size).cpu().numpy()
                    # normalize to 0-255
                    mn, mx = vis_mask.min(), vis_mask.max()
                    vis_norm = (vis_mask - mn) / (mx - mn + 1e-8)
                    vis_img = (vis_norm * 255).astype(np.uint8)
                    
                    # Save with timestep and concept index for easy matching with self_attn visualizations
                    if t > 500:
                        filename = f"mask_images/gaussian_mask_{self.place_in_unet}_t{int(t)}_conceptidx_{mask_idx}_step{step}.png"
                        os.makedirs("mask_images", exist_ok=True)
                        cv2.imwrite(filename, vis_img)
                # except Exception as e:
                #     print(f"Failed to save mask: {e}")
                #     pass
                cond_attention_probs = cond_attention_probs * mask + cond_attention_probs * (1-mask) * self.soft_mask_rate
        
        # normalize again
        cond_attention_probs = cond_attention_probs / (cond_attention_probs.sum(-1, keepdim=True) + 1e-12)
        attention_probs = cond_attention_probs

        self.attnstore(cond_attention_probs, is_cross, self.place_in_unet)
        # self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# # Taken from AttendExcite
# def register_attention_control(model, controller):

#     attn_procs = {}
#     cross_att_count = 0
#     for name in model.unet.attn_processors.keys():
#         # print("name:", name)
#         cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
#         if name.startswith("mid_block"):
#             hidden_size = model.unet.config.block_out_channels[-1]
#             place_in_unet = "mid"
#         elif name.startswith("up_blocks"):
#             block_id = int(name[len("up_blocks.")])
#             hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
#             place_in_unet = "up"
#         elif name.startswith("down_blocks"):
#             block_id = int(name[len("down_blocks.")])
#             hidden_size = model.unet.config.block_out_channels[block_id]
#             place_in_unet = "down"
#         else:
#             continue

#         cross_att_count += 1
#         # attn_procs[name] = AttendExciteCrossAttnProcessor(
#         #     attnstore=controller, place_in_unet=place_in_unet
#         # )
#         attn_procs[name] = MoLECrossAttnProcessor(     
#             attnstore=controller, place_in_unet=place_in_unet
#         )
#     model.unet.set_attn_processor(attn_procs)
#     controller.num_att_layers = cross_att_count


# Taken from ToMe --> modified for MoLE
def register_attention_control(model, controller):
    attn_greenlist = ["up_blocks.0.attentions.1.transformer_blocks.1.attn2.processor",  # the ones from ToMe
                    "up_blocks.0.attentions.1.transformer_blocks.2.attn2.processor",
                    "up_blocks.0.attentions.1.transformer_blocks.3.attn2.processor",
                    "up_blocks.0.attentions.1.transformer_blocks.1.attn1.processor",
                    "up_blocks.0.attentions.1.transformer_blocks.2.attn1.processor",
                    "up_blocks.0.attentions.1.transformer_blocks.3.attn1.processor",
                    
                    # # extra up blocks for sa restriction
                    # "up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor",
                    # #"up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor",
                    # "up_blocks.1.attentions.0.transformer_blocks.1.attn1.processor",
                    # #"up_blocks.1.attentions.0.transformer_blocks.1.attn2.processor",
                    # "up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor",
                    # #"up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor",
                    # "up_blocks.1.attentions.1.transformer_blocks.1.attn1.processor",
                    # #"up_blocks.1.attentions.1.transformer_blocks.1.attn2.processor",
                    # "up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor",
                    # #"up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor",
                    # "up_blocks.1.attentions.2.transformer_blocks.1.attn1.processor",
                    # #"up_blocks.1.attentions.2.transformer_blocks.1.attn2.processor",
                    
                    # # extra up blocks for sa restriction
                    # "up_blocks.0.attentions.2.transformer_blocks.0.attn2.processor",
                    # "up_blocks.0.attentions.2.transformer_blocks.1.attn1.processor",
                    # "up_blocks.0.attentions.2.transformer_blocks.1.attn2.processor",
                    # "up_blocks.0.attentions.2.transformer_blocks.2.attn1.processor",
                    # "up_blocks.0.attentions.2.transformer_blocks.2.attn2.processor",
                    # "up_blocks.0.attentions.2.transformer_blocks.3.attn1.processor",
                    # "up_blocks.0.attentions.2.transformer_blocks.3.attn2.processor",

                    # extra down blocks for sa restriction
                    "down_blocks.0.attentions.1.transformer_blocks.1.attn2.processor",
                    "down_blocks.0.attentions.1.transformer_blocks.2.attn2.processor",
                    "down_blocks.0.attentions.1.transformer_blocks.3.attn2.processor",
                    "down_blocks.0.attentions.1.transformer_blocks.1.attn1.processor",
                    "down_blocks.0.attentions.1.transformer_blocks.2.attn1.processor",
                    "down_blocks.0.attentions.1.transformer_blocks.3.attn1.processor",
                    
                    # Keep: needed for masks
                    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor",
                    "down_blocks.2.attentions.1.transformer_blocks.1.attn2.processor",
                    "down_blocks.2.attentions.1.transformer_blocks.2.attn2.processor",
                    "down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor",
                    "down_blocks.2.attentions.1.transformer_blocks.1.attn1.processor",
                    "down_blocks.2.attentions.1.transformer_blocks.2.attn1.processor",
                    
                    # extra --> these make results bad
                    # "down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor",
                    # #"down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor",
                    # "down_blocks.1.attentions.0.transformer_blocks.1.attn1.processor",
                    # #"down_blocks.1.attentions.0.transformer_blocks.1.attn2.processor",
                    # "down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor",
                    # #"down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor",
                    # "down_blocks.1.attentions.1.transformer_blocks.1.attn1.processor",
                    # #"down_blocks.1.attentions.1.transformer_blocks.1.attn2.processor",

                    "mid_block.attentions.0.transformer_blocks.0.attn2.processor",
                    "mid_block.attentions.0.transformer_blocks.1.attn2.processor",
                    "mid_block.attentions.0.transformer_blocks.2.attn2.processor",
                    "mid_block.attentions.0.transformer_blocks.0.attn1.processor",
                    "mid_block.attentions.0.transformer_blocks.1.attn1.processor",
                    "mid_block.attentions.0.transformer_blocks.2.attn1.processor"
                    ]
    # TODO: add a check to see if all items in greenlist are being loaded. May change due to diffusers version?
    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        # print(name)
        if name not in attn_greenlist:
            attn_procs[name] = model.unet.attn_processors[name]
            continue
        #     # if name.startswith('mid_block') and name.endswith("attn1.processor"):
        #     #     attn_procs[name] = CompactAttnProcessor(controller, 'mid')
        #     # else:
        #     #     attn_procs[name] = model.unet.attn_processors[name]
        #     # continue
        #     if name.endswith("attn2.processor"):
        #         attn_procs[name] = CompactAttnProcessor(controller, name.split("_")[0])
        #     else:
        #         attn_procs[name] = model.unet.attn_processors[name]
        #     continue
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = MoLECrossAttnProcessor(     
            attnstore=controller, place_in_unet=place_in_unet
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead #! TODO: change this to 64 else self-attn on shallow layers won't work
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0


def select_low_entropy_heads(attention_map, top_k=None, entropy_threshold=None, concept_indices_list=None):
    """
    Select attention heads with low entropy (more focused attention).
    
    Low entropy = more peaked/focused attention distribution across tokens
    High entropy = more diffuse/spread attention distribution
    
    Usage:
        # Select top 5 most focused heads
        filtered, indices, entropies = select_low_entropy_heads(attn, top_k=5)
        
        # Select heads with entropy < 2.0
        filtered, indices, entropies = select_low_entropy_heads(attn, entropy_threshold=2.0)
        
        # Calculate entropy only over concept tokens
        filtered, indices, entropies = select_low_entropy_heads(
            attn, top_k=5, concept_indices_list=[[2, 3], [5, 6]]
        )
    
    
    Args:
        attention_map: Tensor of shape (num_heads, spatial_dim, token_length)
                      or (num_heads, H, W, token_length)
        top_k: Number of lowest entropy heads to keep (if None, use entropy_threshold)
        entropy_threshold: Keep heads with entropy below this value (if None, use top_k)
        concept_indices_list: List of token indices for each concept. If provided, entropy is 
                             calculated only over these concept tokens. E.g., [[2, 3], [5, 6]]
    
    Returns:
        Filtered attention map with selected heads, selected indices, all entropies
    """
    if attention_map.dim() == 4:
        # Shape: (num_heads, H, W, token_length)
        num_heads, H, W, token_length = attention_map.shape
        # Reshape to (num_heads, spatial_dim, token_length)
        attn_flat = attention_map.reshape(num_heads, H * W, token_length)
    else:
        # Shape: (num_heads, spatial_dim, token_length)
        attn_flat = attention_map
        num_heads = attn_flat.shape[0]
    
    # Compute entropy for each head
    # Entropy per spatial location: -sum(p * log(p)) across tokens
    # Then average across spatial locations for each head
    entropies = []
    
    if concept_indices_list is not None:
        # Calculate entropy only over concept tokens
        # Average attention across all concept tokens for each head
        for head_idx in range(num_heads):
            head_attn = attn_flat[head_idx]  # (spatial_dim, token_length)
            
            # Collect attention values for all concepts
            concept_attns = []
            for concept_indices in concept_indices_list:
                # Extract attention for this concept's tokens: (spatial_dim, num_concept_tokens)
                concept_attn = head_attn[:, concept_indices]
                # Average across concept tokens: (spatial_dim,)
                concept_attns.append(concept_attn.mean(dim=-1, keepdim=True))
            
            # Stack to get (spatial_dim, num_concepts)
            concept_stack = torch.cat(concept_attns, dim=-1)
            
            # Normalize to probabilities across concepts
            concept_probs = concept_stack / (concept_stack.sum(dim=-1, keepdim=True) + 1e-10)
            
            # Compute entropy: -sum(p * log(p)) across concepts
            log_probs = torch.log(concept_probs + 1e-10)
            entropy = -(concept_probs * log_probs).sum(dim=-1)  # (spatial_dim,)
            
            # Average entropy across spatial locations
            avg_entropy = entropy.mean().item()
            entropies.append(avg_entropy)
    else:
        # Original behavior: calculate entropy across all tokens
        # for head_idx in range(num_heads):
        #     head_attn = attn_flat[head_idx]  # (spatial_dim, token_length)
            
        #     # Normalize to probabilities if not already
        #     head_attn = head_attn / (head_attn.sum(dim=-1, keepdim=True) + 1e-10)
            
        #     # Compute entropy: -sum(p * log(p))
        #     log_attn = torch.log(head_attn + 1e-10)
        #     entropy = -(head_attn * log_attn).sum(dim=-1)  # (spatial_dim,)
        #
        #     # Average entropy across spatial locations
        #     avg_entropy = entropy.mean().item()
        #     entropies.append(avg_entropy)
        raise ValueError("concept_indices_list must be provided for entropy calculation.")
    
    entropies = torch.tensor(entropies)
    
    # Select heads based on criteria
    if top_k is not None:
        # Select top_k lowest entropy heads
        k = min(top_k, num_heads)
        _, indices = torch.topk(entropies, k, largest=False)  # lowest entropy
    # elif entropy_threshold is not None:
    #     # Select heads below entropy threshold
    #     indices = torch.where(entropies < entropy_threshold)[0]
    #     if len(indices) == 0:
    #         # Fallback: select at least the lowest entropy head
    #         indices = torch.tensor([entropies.argmin()])
    # else:
    #     # Default: select top 50% lowest entropy heads
    #     k = max(1, num_heads // 2)
    #     _, indices = torch.topk(entropies, k, largest=False)
    else:
        raise ValueError("top_k must be > 0 .")
    
    # Filter attention map
    if attention_map.dim() == 4:
        selected = attention_map[indices]
    else:
        selected = attn_flat[indices]
    
    return selected, indices, entropies


def select_bidir_entropy_heads(attention_map, concept_indices_list, top_k=None):
    """
    Select attention heads with low bi-directional entropy.
    
    Bi-directional entropy = channel_entropy × spatial_entropy
    - Channel entropy: For each pixel, how clearly does it belong to one concept?
    - Spatial entropy: For each concept, how localized is it (not spread everywhere)?
    
    Good heads have BOTH low channel AND low spatial entropy:
    - Each pixel clearly belongs to one concept (not mixed)
    - Each concept is localized in specific regions (not uniform everywhere)
    
    This avoids both dominance (uniform spatial distribution) and confusion (mixed concepts).
    
    Args:
        attention_map: Tensor of shape (num_heads, H, W, token_length) or (num_heads, spatial_dim, token_length)
        concept_indices_list: List of token indices for each concept. E.g., [[2, 3], [5, 6]]
        top_k: Number of lowest score heads to keep
    
    Returns:
        Filtered attention map, selected indices, scores (lower is better)
    """
    if attention_map.dim() == 4:
        num_heads, H, W, token_length = attention_map.shape
        attn_flat = attention_map.reshape(num_heads, H * W, token_length)
    else:
        attn_flat = attention_map
        num_heads = attn_flat.shape[0]
    
    # Ensure float dtype for operations
    if attn_flat.dtype not in (torch.float32, torch.float64):
        attn_flat = attn_flat.float()
    
    scores = []
    
    for head_idx in range(num_heads):
        head_attn = attn_flat[head_idx]  # (spatial_dim, token_length)
        
        # Extract per-concept attention
        concept_attns = []
        for concept_indices in concept_indices_list:
            concept_attn = head_attn[:, concept_indices].mean(dim=-1)  # (spatial_dim,)
            concept_attns.append(concept_attn)
        
        # Stack to get (num_concepts, spatial_dim)
        concept_stack = torch.stack(concept_attns, dim=0)
        
        # 1. Channel Entropy (concept direction): For each spatial location, entropy across concepts
        # Transpose to (spatial_dim, num_concepts)
        spatial_concept = concept_stack.t()  # (spatial_dim, num_concepts)
        
        # Normalize to probabilities across concepts at each location
        spatial_probs = spatial_concept / (spatial_concept.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Compute entropy at each spatial location
        log_spatial_probs = torch.log(spatial_probs + 1e-10)
        channel_entropy_per_loc = -(spatial_probs * log_spatial_probs).sum(dim=-1)  # (spatial_dim,)
        
        # Average channel entropy across all spatial locations
        avg_channel_entropy = channel_entropy_per_loc.mean().item()
        
        # 2. Spatial Entropy (location direction): For each concept, entropy across spatial locations
        # concept_stack is already (num_concepts, spatial_dim)
        
        # Normalize to probabilities across spatial locations for each concept
        concept_probs = concept_stack / (concept_stack.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Compute entropy for each concept across spatial locations
        log_concept_probs = torch.log(concept_probs + 1e-10)
        spatial_entropy_per_concept = -(concept_probs * log_concept_probs).sum(dim=-1)  # (num_concepts,)
        
        # Average spatial entropy across all concepts
        avg_spatial_entropy = spatial_entropy_per_concept.mean().item()
        
        # 3. Bi-directional score: product of both entropies (lower is better)
        bidir_score = avg_channel_entropy * avg_spatial_entropy
        scores.append(bidir_score)
    
    scores = torch.tensor(scores)
    
    # Select top_k lowest scoring heads
    if top_k is not None:
        k = min(top_k, num_heads)
        _, indices = torch.topk(scores, k, largest=False)  # lowest scores
    else:
        raise ValueError("top_k must be > 0 for bidir_entropy selection.")
    
    # Filter attention map
    if attention_map.dim() == 4:
        selected = attention_map[indices]
    else:
        selected = attn_flat[indices]
    
    return selected, indices, scores



def select_balanced_heads(attention_map, concept_indices_list, top_k=None, 
                         selection_metric='separation', min_concept_presence=0.1):
    """
    Select attention heads that balance all concepts (avoid dominant concept heads).
    
    Corrected understanding of dominance:
        Dominant concept = its 10th percentile value is high (close to or exceeds other concepts' typical values)
        Example (BAD - lion dominates):
            bird:  p10=0.10, p90=0.35, mean=0.225, median=0.22
            lion:  p10=0.30, p90=0.90, mean=0.60, median=0.58
            → lion's floor (0.30) > bird's median (0.22) → lion is always high, even where bird should be
        
        Example (GOOD - spatially separated):
            bird:  p10=0.05, p90=0.90, mean=0.40, median=0.35
            lion:  p10=0.05, p90=0.85, mean=0.45, median=0.42
            → Both concepts can "turn off" (go to ~0.05) in the other's region
    
    Args:
        attention_map: Tensor of shape (num_heads, H, W, token_length) or (num_heads, spatial_dim, token_length)
        concept_indices_list: List of token indices for each concept. E.g., [[2, 3], [5, 6]]
        top_k: Number of best heads to keep
        selection_metric: 
            'variance' - high spatial variance per concept (localized patterns)
            'separation' - concepts' value ranges don't overlap (no dominance) [RECOMMENDED]
            'combined' - variance * separation_score
            'combined_v2' - (1 - spatial_entropy) * separation (localization + depth-based separation)
            'balance' - balanced total attention + separation
        min_concept_presence: Minimum 90th percentile attention to consider (filters heads that ignore concepts)
    
    Returns:
        Filtered attention map, selected indices, scores
    """
    if attention_map.dim() == 4:
        num_heads, H, W, token_length = attention_map.shape
        attn_flat = attention_map.reshape(num_heads, H * W, token_length)
    else:
        attn_flat = attention_map
        num_heads = attn_flat.shape[0]
    
    scores = []
    
    for head_idx in range(num_heads):
        head_attn = attn_flat[head_idx]  # (spatial_dim, token_length)
        
        # Extract per-concept attention
        concept_attns = []
        for concept_indices in concept_indices_list:
            concept_attn = head_attn[:, concept_indices].mean(dim=-1)  # (spatial_dim,)
            concept_attns.append(concept_attn)
        
        # Stack: (num_concepts, spatial_dim)
        concept_stack = torch.stack(concept_attns, dim=0)
        # Convert to float if needed (FIX for quantile error)
        if concept_stack.dtype not in (torch.float32, torch.float64):
            concept_stack = concept_stack.float()
        
        # Compute percentiles for each concept (more robust than min/max)
        concept_p10 = torch.quantile(concept_stack, 0.1, dim=1)   # (num_concepts,) 10th percentile
        concept_p90 = torch.quantile(concept_stack, 0.9, dim=1)   # (num_concepts,) 90th percentile
        concept_means = concept_stack.mean(dim=1)                 # (num_concepts,)
        concept_medians = concept_stack.median(dim=1)[0]          # (num_concepts,)
        
        # Check if all concepts are present (not ignored)
        # Use 90th percentile to check presence (more robust than max)
        if (concept_p90 < min_concept_presence).any():
            # Penalize heads that ignore some concepts
            scores.append(-1e6)
            continue
        
        if selection_metric == 'variance':
            # High spatial variance = concept is localized (good)
            variances = concept_stack.var(dim=1)  # (num_concepts,)
            score = variances.min().item()
            
        elif selection_metric == 'separation':
            # Key insight: Good heads have LOW 10th percentile for all concepts
            # (concepts can "turn off" to yield space to others)
            
            num_concepts = len(concept_indices_list)
            separation_scores = []
            
            for i in range(num_concepts):
                for j in range(i + 1, num_concepts):
                    # Check dominance in both directions using percentiles
                    # Concept i dominates j if: p10(i) > median(j)
                    i_p10 = concept_p10[i].item()
                    j_p10 = concept_p10[j].item()
                    i_p90 = concept_p90[i].item()
                    j_p90 = concept_p90[j].item()
                    i_mean = concept_means[i].item()
                    j_mean = concept_means[j].item()
                    i_median = concept_medians[i].item()
                    j_median = concept_medians[j].item()
                    
                    # Dominance test: if p10(A) > median(B), A never yields to B
                    i_dominates = i_p10 > j_median
                    j_dominates = j_p10 > i_median
                    
                    if i_dominates or j_dominates:
                        # Strong dominance detected
                        separation_scores.append(0.0)
                    else:
                        # Compute separation quality:
                        # Good: both concepts have low p10 (can turn off)
                        # Score based on how much lower the p10 are compared to means
                        
                        # Normalized depth: (mean - p10) / mean
                        # High value = concept can go very low (good for yielding)
                        i_depth = (i_mean - i_p10) / (i_mean + 1e-6)
                        j_depth = (j_mean - j_p10) / (j_mean + 1e-6)
                        
                        # Both should have good depth (low p10)
                        pair_score = min(i_depth, j_depth)
                        separation_scores.append(pair_score)
            
            # Worst-case separation across all pairs
            score = min(separation_scores) if separation_scores else 0.0
            
        elif selection_metric == 'combined':
            # Variance + separation
            variances = concept_stack.var(dim=1)
            min_variance = variances.min().item()
            
            # Separation component
            num_concepts = len(concept_indices_list)
            separation_scores = []
            
            for i in range(num_concepts):
                for j in range(i + 1, num_concepts):
                    i_p10 = concept_p10[i].item()
                    j_p10 = concept_p10[j].item()
                    i_median = concept_medians[i].item()
                    j_median = concept_medians[j].item()
                    i_mean = concept_means[i].item()
                    j_mean = concept_means[j].item()
                    
                    i_dominates = i_p10 > j_median
                    j_dominates = j_p10 > i_median
                    
                    if i_dominates or j_dominates:
                        separation_scores.append(0.0)
                    else:
                        i_depth = (i_mean - i_p10) / (i_mean + 1e-6)
                        j_depth = (j_mean - j_p10) / (j_mean + 1e-6)
                        separation_scores.append(min(i_depth, j_depth))
            
            separation = min(separation_scores) # if separation_scores else 0.0
            score = min_variance * separation
            
        elif selection_metric == 'balance':
            # Total attention balance + separation
            total_attns = concept_stack.sum(dim=1)
            balance_score = 1.0 - total_attns.std() / (total_attns.mean() + 1e-10)
            
            # Separation
            num_concepts = len(concept_indices_list)
            separation_scores = []
            
            for i in range(num_concepts):
                for j in range(i + 1, num_concepts):
                    i_p10 = concept_p10[i].item()
                    j_p10 = concept_p10[j].item()
                    i_median = concept_medians[i].item()
                    j_median = concept_medians[j].item()
                    i_mean = concept_means[i].item()
                    j_mean = concept_means[j].item()
                    
                    i_dominates = i_p10 > j_median
                    j_dominates = j_p10 > i_median
                    
                    if i_dominates or j_dominates:
                        separation_scores.append(0.0)
                    else:
                        i_depth = (i_mean - i_p10) / (i_mean + 1e-6)
                        j_depth = (j_mean - j_p10) / (j_mean + 1e-6)
                        separation_scores.append(min(i_depth, j_depth))
            
            separation = min(separation_scores) if separation_scores else 0.0
            score = balance_score * separation
            
        elif selection_metric == 'combined_v2':
            # Spatial entropy (low localization = high entropy = bad) + Separation (depth-based)
            # This combines: how localized each concept is + how well concepts can yield to each other
            
            # 1. Spatial entropy component: for each concept, entropy across spatial locations
            # Normalize to probabilities across spatial locations for each concept
            concept_probs = concept_stack / (concept_stack.sum(dim=-1, keepdim=True) + 1e-10)
            
            # Compute entropy for each concept across spatial locations
            log_concept_probs = torch.log(concept_probs + 1e-10)
            spatial_entropy_per_concept = -(concept_probs * log_concept_probs).sum(dim=-1)  # (num_concepts,)
            
            # Average spatial entropy across all concepts (lower = better localized)
            avg_spatial_entropy = spatial_entropy_per_concept.mean().item()
            
            # Invert spatial entropy for scoring (we want LOW entropy = HIGH score)
            # Max possible entropy ≈ log(spatial_dim), normalize to [0, 1] range
            spatial_dim = concept_stack.shape[1]
            max_entropy = torch.log(torch.tensor(spatial_dim, dtype=torch.float32)).item()
            spatial_score = 1.0 - (avg_spatial_entropy / (max_entropy + 1e-6))  # 1.0 = perfectly localized
            
            # 2. Separation component (depth-based, as in 'separation' metric)
            num_concepts = len(concept_indices_list)
            separation_scores = []
            
            for i in range(num_concepts):
                for j in range(i + 1, num_concepts):
                    i_p10 = concept_p10[i].item()
                    j_p10 = concept_p10[j].item()
                    i_median = concept_medians[i].item()
                    j_median = concept_medians[j].item()
                    i_mean = concept_means[i].item()
                    j_mean = concept_means[j].item()
                    
                    i_dominates = i_p10 > j_median
                    j_dominates = j_p10 > i_median
                    
                    if i_dominates or j_dominates:
                        separation_scores.append(0.0)
                    else:
                        i_depth = (i_mean - i_p10) / (i_mean + 1e-6)
                        j_depth = (j_mean - j_p10) / (j_mean + 1e-6)
                        separation_scores.append(min(i_depth, j_depth))
            
            separation = min(separation_scores) if separation_scores else 0.0
            
            # Combined score: both spatial localization AND separation (both in [0, 1] range)
            score = spatial_score * separation
            
        else:
            raise ValueError(f"Unknown selection_metric: {selection_metric}")
        
        scores.append(score)
        # print(score)
    
    scores = torch.tensor(scores)
    
    # Select top_k highest scoring heads
    if top_k is not None:
        k = min(top_k, num_heads)
        _, indices = torch.topk(scores, k, largest=True)
    else:
        # Keep all heads with positive scores
        indices = torch.where(scores > 0)[0]
        if len(indices) == 0:
            indices = torch.tensor([scores.argmax()])
    # print(f"scores: {scores[indices].tolist()}")

    # Filter attention map
    if attention_map.dim() == 4:
        selected = attention_map[indices]
    else:
        selected = attn_flat[indices]
    
    return selected, indices, scores


## Version 2 (can select attentions from specific block_index)
def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int,
                        use_deepest_only: bool = True,
                        keep_n: int = 2,
                        block_index: int = 0,
                        use_low_entropy_heads: bool = False,
                        top_k_heads: int = None,
                        entropy_threshold: float = None,
                        concept_indices_list: List[List[int]] = None,
                        head_selection_metric: str = 'entropy') -> torch.Tensor:
    """
    Aggregates attention maps from the deepest layers in SDXL.
    If use_deepest_only is True, selects only the last `keep_n` layers per block.
    If use_deepest_only is False, selects only the attention map at `block_index` from each location.
    
    Args:
        attention_store: Store containing attention maps
        res: Resolution for reshaping
        from_where: List of locations ("up", "down", "mid")
        is_cross: Whether to use cross-attention or self-attention
        select: Index for selecting specific batch/head
        use_deepest_only: If True, use deepest layers; if False, use specific block_index
        keep_n: Number of deepest layers to keep (only used when use_deepest_only=True)
        block_index: Index of specific block to select (only used when use_deepest_only=False)
        use_low_entropy_heads: If True, select heads with low entropy (focused attention)
        top_k_heads: Number of lowest entropy heads to keep (mutually exclusive with entropy_threshold)
        entropy_threshold: Keep heads with entropy below this value (mutually exclusive with top_k_heads)
        concept_indices_list: List of token indices for each concept (for entropy calculation)
        head_selection_metric: Metric for head selection when use_low_entropy_heads=True
                              'entropy' - low channel entropy only (can select dominant heads)
                              'bidir_entropy' - low channel × spatial entropy (avoids both dominance and confusion)
                              'separation' - percentile-based dominance detection (recommended)
                              'variance' - high spatial variance per concept
                              'combined' - variance * separation
                              'combined_v2' - (1 - spatial_entropy) * separation (localization + depth)
                              'balance' - balanced total attention + separation
    
    Returns:
        Aggregated attention tensor of shape (res, res, token_length)
    """
    out = []
    attention_maps = attention_store.get_average_attention()
    key_suffix = 'cross' if is_cross else 'self'
    
    for location in from_where:
        key = f"{location}_{key_suffix}"
        if key not in attention_maps:
            continue
        all_items = attention_maps[key]
        
        if location == "up":
            all_items = list(reversed(all_items))  # Fix: reverse for "up" since it's decoder order
        
        if use_deepest_only:
            
            selected_items = (
                all_items[-keep_n:] if len(all_items) > keep_n
                else all_items
            )
        else:
            
            if block_index >= len(all_items):
                print(f"Warning: block_index {block_index} >= available blocks {len(all_items)} for {location}")
                continue
            if block_index < 0:
                # Support negative indexing (e.g., -1 for last block)
                effective_index = len(all_items) + block_index
            else:
                effective_index = block_index
                
            if 0 <= effective_index < len(all_items):
                selected_items = [all_items[effective_index]]
            else:
                print(f"Warning: Invalid block_index {block_index} for {location} with {len(all_items)} blocks")
                continue
        
        for idx, item in enumerate(selected_items):
            # print(f"location: {location}, shape: {item.shape}, idx: {idx}, block_index: {block_index if not use_deepest_only else 'N/A'}")
            # if item.shape[1] == 4096:
            #     res = 64
            # elif item.shape[1] == 1024:
            #     res = 32
            # else:
            #     raise ValueError(f"Unexpected spatial size {item.shape[1]} in attention map.")
            cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select] #(num_heads, res, res, token_length)
            
            # Apply head selection if enabled
            if use_low_entropy_heads:
                if head_selection_metric == 'entropy':
                    # Original entropy-based selection (can select dominant heads)
                    cross_maps, selected_indices, scores = select_low_entropy_heads(
                        cross_maps,
                        top_k=top_k_heads,
                        entropy_threshold=entropy_threshold,
                        concept_indices_list=concept_indices_list
                    )
                elif head_selection_metric == 'bidir_entropy':
                    # Bi-directional entropy (channel × spatial)
                    cross_maps, selected_indices, scores = select_bidir_entropy_heads(
                        cross_maps,
                        concept_indices_list=concept_indices_list,
                        top_k=top_k_heads
                    )
                else:
                    # Balanced selection (avoids dominant heads)
                    cross_maps, selected_indices, scores = select_balanced_heads(
                        cross_maps,
                        concept_indices_list=concept_indices_list,
                        top_k=top_k_heads,
                        selection_metric=head_selection_metric,
                        min_concept_presence=0.05
                    )
            # print(f"  Selected {len(selected_indices)}/{cross_maps.shape[0]} heads with scores: {scores[selected_indices].tolist()}")
            
            out.append(cross_maps)


    if not out:
        raise ValueError("No attention maps matched the criteria.")
        
    out = torch.cat(out, dim=0)
    # print("out shape:", out.shape)
    out = out.sum(0) / out.shape[0] # Average across all selected attention heads/blocks
    return out

def find_subsequence_indices(tokens, phrase_tokens):
    #todo: Currently only returns the first occurrence of the phrase in the tokens.
    for i in range(len(tokens) - len(phrase_tokens) + 1):
        if tokens[i:i+len(phrase_tokens)] == phrase_tokens:
            return list(range(i, i + len(phrase_tokens)))
    raise ValueError(f"Phrase {' '.join(phrase_tokens)} not found in tokens: {' '.join(tokens)}")



def _percentile(x: torch.Tensor, q: float, dim=None, keepdim=False):
    # Works on PyTorch w/ torch.quantile; falls back if needed
    return torch.quantile(x, q, dim=dim, keepdim=keepdim)

def _sparsemax(logits, dim=-1, eps=1e-12):
    # Vectorized sparsemax (returns probs; sums to 1 on support)
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    num_classes = z.shape[dim]
    k = torch.arange(1, num_classes + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = -1
    k = k.view(shape)

    taus = (torch.cumsum(z_sorted, dim=dim) - 1) / k
    support = (z_sorted > taus).type_as(z)
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (torch.sum(z_sorted * support, dim=dim, keepdim=True) - 1) / k_z
    z_p = torch.clamp(z - tau, min=0)
    # Numerical safeguard for all-zero rows (shouldn't happen after centering)
    row_sum = z_p.sum(dim=dim, keepdim=True)
    z_p = torch.where(row_sum > eps, z_p, torch.full_like(z_p, 1.0 / z_p.size(dim)))
    return z_p



def process_attention_to_masks(
    concept_indices_list, # List of concept indices for each concept
    attention_maps,
    output_size=(128, 128),
    softmax_temperature=0.05,  # Lower default temperature
    kernel_size=3,
    sigma=0.5,
    use_complement_last=True,
    balance_concepts=True  # New parameter
):
    """
    Improved mask processing with better concept balance.
    """
    # index_offset = 1  # Skip <BOS> token
    
    # # Tokenize full prompt
    # input_ids = tokenize_prompt(tokenizer, full_prompt)[0]
    # tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # tokens = [t for t in tokens if t not in tokenizer.all_special_tokens and not t.startswith("<|")]
    # tokens = [t.lower().strip("Ġ").strip("##") for t in tokens]
    
    all_masks = []
    concept_weights = {'mean':[],'std':[], 'max':[], 'min':[]}  # Track average attention per concept
    
    for i, concept_indices in enumerate(concept_indices_list):
        # concept_ids = tokenize_prompt(tokenizer, concept)[0]
        # concept_tokens = tokenizer.convert_ids_to_tokens(concept_ids)
        # concept_tokens = [t for t in concept_tokens if t not in tokenizer.all_special_tokens and not t.startswith("<|")]
        # concept_tokens = [t.lower().strip("Ġ").strip("##") for t in concept_tokens]
        
        # # Find concept indices in the full prompt
        # try:
        #     concept_indices = find_subsequence_indices(tokens, concept_tokens)
        #     concept_indices = [idx + index_offset for idx in concept_indices]
        # except ValueError:
        #     print(f"Warning: Concept '{concept}' not found in prompt. Using fallback.")
        #     # Fallback: use attention from middle tokens
        #     concept_indices = list(range(len(tokens)//2 - 1, len(tokens)//2 + 1))
        
        # Extract attention for this concept
        concept_attention = attention_maps[:, :, concept_indices]
        
        # Average over concept tokens
        attn = concept_attention.mean(dim=2, keepdim=True)  # H x W x 1
        
        # Apply Gaussian smoothing to reduce noise
        if output_size[0] >= 32:  
            smoother = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma) #kernel size can be changed to smaller?

            attn_input = attn.permute(2, 0, 1).unsqueeze(0)
            attn_smooth = F.conv2d(attn_input, smoother.weight.to(attn_input.device, attn_input.dtype), 
                                groups=smoother.groups, padding=2)  # padding=kernel_size//2
            attn = attn_smooth.squeeze(0).permute(1, 2, 0)

        all_masks.append(attn)
        concept_weights['mean'].append(attn.mean().item())
        concept_weights['std'].append(attn.std().item())
        concept_weights['max'].append(attn.max().item())
        concept_weights['min'].append(attn.min().item())
    
    # Balance concept masks if requested
    if balance_concepts and len(all_masks) > 1:
        # Normalize concept weights
        total_weight = sum(concept_weights) #TODO: error here
        target_weight = 1.0 / len(concept_indices_list)  # Target weight for each concept
        
        # Adjust masks to balance concepts
        balanced_masks = []
        for i, (mask, weight) in enumerate(zip(all_masks, concept_weights)):
            if weight > 0:
                # Scale mask to achieve target weight
                scale_factor = target_weight * total_weight / weight
                scale_factor = np.clip(scale_factor, 0.5, 2.0)  # Limit scaling
                balanced_mask = mask * scale_factor
            else:
                balanced_mask = mask
            balanced_masks.append(balanced_mask)
        all_masks = balanced_masks
    
    # Concatenate all masks
    m_prime = torch.cat(all_masks, dim=2)  # H x W x N

    
    # Apply temperature scaling and softmax
    m_prime = m_prime / softmax_temperature
    
    # Add small epsilon to prevent numerical issues
    m_prime = m_prime - m_prime.max(dim=2, keepdim=True)[0]  # Numerical stability
    m = F.softmax(m_prime, dim=2)
    # Resize attention maps to target size
    attns_resized = []
    m = m.detach().cpu().numpy()    
    for i in range(m.shape[2]):
        # Use INTER_CUBIC for better quality when upsampling
        resized = cv2.resize(m[:, :, i], output_size, 
                        interpolation=cv2.INTER_CUBIC)
        attns_resized.append(resized)
    
    m = torch.tensor(np.stack(attns_resized, axis=2))  # H x W x T

    # calc. binary masks
    max_indices = torch.argmax(m, dim=2)  # Shape: (H, W)
    binary_mask = F.one_hot(max_indices, num_classes=m.shape[2]).float()  # Shape: (H, W, K)
    
    # # ensure minimum activation per concept
    # min_activation = 0.1  # Minimum 10% activation per concept
    # m = m * (1 - min_activation * len(concepts)) + min_activation #not sure if this is needed #Todo: remove?
    
    # # Ensure sum to 1 after adjustment
    # m = m / m.sum(dim=2, keepdim=True)

    return m.permute(2, 0, 1).unsqueeze(1), concept_weights,  binary_mask.permute(2, 0, 1).unsqueeze(1)  # (N, 1, H, W)

def _concept_attns(attns, concept_indices_list):
    attn_maps = []
    for concept_idx in concept_indices_list:
        attn = attns[:, :, concept_idx].mean(dim=2, keepdim=True)  # H x W x 1
        attn_maps.append(attn)
    attention_maps = torch.cat(attn_maps, dim=2).permute(2,0,1).contiguous()  # (C, H, W)
    return attention_maps

def _upsample_and_smooth(x, output_size, upsample=True, use_gaussian_smoothing=True):

    device = x.device
    C, H, W = x.shape
    if upsample:
        logits_up = F.interpolate(x.unsqueeze(0), size=output_size, mode="nearest").squeeze(0) # (C, output_H, output_W)

        if use_gaussian_smoothing:
            # Optional Gaussian smoothing after upsample
            sigma = 1.0
            kernel_size = 5
            for k in range(C):  
                attn = logits_up[k:k+1].unsqueeze(0)  # (1,1,output_H, output_W)
                gaussian_smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma).to(device=device, dtype=torch.float32)
                attn_smooth = F.conv2d(attn, gaussian_smoothing.weight.to(attn.device, attn.dtype),
                                        groups=gaussian_smoothing.groups, padding=2)
                logits_up[k:k+1] = attn_smooth.squeeze(0)
        logits_up = torch.clamp(logits_up, min=0)
        # print("logits_Up shape:", logits_up.shape)
        x = logits_up
    return x

def _individual_normalization(x, p_low=0.10, p_high=0.90):
    """
    x: (C,H,W) attention maps for each concept
    Returns x_norm: (C,H,W) normalized attention maps
    """
    C, H, W = x.shape
    # x = torch.clamp(x, min=0)

    # Compute percentiles across spatial dim for each concept
    x_flat = x.view(C, -1)
    p10 = _percentile(x_flat, p_low, dim=1, keepdim=True)
    p90 = _percentile(x_flat, p_high, dim=1, keepdim=True)
    scale = (p90 - p10).clamp_min(1e-6)
    x_norm = torch.clamp((x_flat - p10) / scale, 0.0, 1.0).view_as(x)  # (C,H,W)
    return x_norm

def _bg_blend(fg, bg_provided, fg_bg_blend=0.5):
    # 2) Background prior from residual free space (1 - max foreground)
    fg_max, _ = fg.max(dim=0, keepdim=True)  # (1,H,W)
    bg_residual = torch.clamp(1.0 - fg_max, 0.0, 1.0)

    # Process background: Blend provided BG (all tokens) and residual free space
    bg_blend = fg_bg_blend * bg_provided + (1.0 - fg_bg_blend) * bg_residual  # (1,H,W)
    return bg_blend

def _mutual_suppression(x, suppression_rho=0.30):
    # x: (C,H,W) attention maps for each concept

    fg = x[:-1]  # (C-1,H,W)
    if fg.size(0) > 1 and suppression_rho > 0:
        others_max = []
        # Compute max over "others" efficiently: max_all and second_max trick
        fg_all_max, idx_max = fg.max(dim=0, keepdim=True)  # (1,H,W)
        # Replace max channel temporarily to compute second max
        mask = torch.nn.functional.one_hot(idx_max.squeeze(0).long(), num_classes=fg.size(0)).permute(2,0,1)  # (C-1,H,W)
        replaced = torch.where(mask.bool(), torch.tensor(-1e9, device=x.device), fg)
        fg_second_max, _ = replaced.max(dim=0, keepdim=True)
        # Per-channel "others max"
        others_max = torch.where(mask.bool(), fg_second_max, fg_all_max)  # (C-1,H,W)
        fg_logits = fg - suppression_rho * others_max
    else:
        raise ValueError("Mutual suppression requires at least two foreground concepts and a positive suppression rho.")
    bg = x[-1:]  # (1,H,W)
    x_suppressed = torch.cat([fg_logits, bg], dim=0)  # (C,H,W)
    return x_suppressed

def _concepts_softmax(x, temp=1.0):
    C, H, W = x.shape
    x = x * temp
    probs_up = torch.softmax(x.permute(1,2,0).reshape(-1, C), dim=-1)
    probs_up = probs_up.view(H, W, C).permute(2,0,1).contiguous() # (C,H,W)
    return probs_up

def _concepts_sparsemax(x, temp=1.0):
    C, H, W = x.shape
    x = x * temp
    probs_up = _sparsemax(x.permute(1,2,0).reshape(-1, C), dim=-1)
    probs_up = probs_up.view(H, W, C).permute(2,0,1).contiguous() # (C,H,W)
    return probs_up

def _prepare_gray_guidance(denoised_x0, out_hw, device, dtype=torch.float32):
    """
    denoised_x0: np.ndarray or torch.Tensor, shape (H,W,3) or (3,H,W) or (H,W)
    Returns I_gray of shape (1,1,H_out,W_out) in [0,1], float32 on device.
    """
    # if isinstance(denoised_x0, torch.Tensor):
    I = denoised_x0.detach().to(device=device, dtype=dtype)
    if I.ndim == 2:
        I = I.unsqueeze(0)  # (1,H,W)
    if I.ndim == 3:
        if I.shape[0] in (1,3):  # (C,H,W)
            pass
        elif I.shape[-1] in (1,3):  # (H,W,C)
            I = I.permute(2,0,1)
        else:
            raise ValueError("Unsupported denoised_x0 shape.")
    elif I.ndim == 4:
        I = I.squeeze(0)  # (C,H,W) if accidentally batched

    # to gray (1,H,W)
    if I.shape[0] == 3:
        r, g, b = I[0:1], I[1:1+1], I[2:2+1]
        I = 0.2989 * r + 0.5870 * g + 0.1140 * b
    elif I.shape[0] == 1:
        pass
    else:
        # fallback: average channels
        I = I.mean(dim=0, keepdim=True)

    H_out, W_out = out_hw
    I = I.unsqueeze(0)  # (1,1,H,W)
    if I.shape[-2:] != (H_out, W_out):
        I = F.interpolate(I, size=(H_out, W_out), mode="bilinear", align_corners=False)
    return I  # (1,1,H_out,W_out)

def _guided_filter_gray_batch(p, I, radius=2, eps=1e-3):
    """
    p: (B,1,H,W) masks to refine
    I: (1,1,H,W) gray guidance image in [0,1]
    Returns q: (B,1,H,W)
    """
    # Repeat guidance over batch
    B, _, H, W = p.shape
    I = I.expand(B, 1, H, W)

    k = 2 * radius + 1
    # box means via avg_pool2d (stride=1, padding=radius)
    mean_I = F.avg_pool2d(I, kernel_size=k, stride=1, padding=radius)
    mean_p = F.avg_pool2d(p, kernel_size=k, stride=1, padding=radius)
    mean_Ip = F.avg_pool2d(I * p, kernel_size=k, stride=1, padding=radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = F.avg_pool2d(I * I, kernel_size=k, stride=1, padding=radius)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = F.avg_pool2d(a, kernel_size=k, stride=1, padding=radius)
    mean_b = F.avg_pool2d(b, kernel_size=k, stride=1, padding=radius)

    q = mean_a * I + mean_b
    return q

def guided_filter_masks(probs_CHW, denoised_x0, radius=2, eps=1e-3):
    """
    probs_CHW: (C,H,W) soft masks in [0,1], sum over C ~= 1 per pixel
    denoised_x0: guidance image (H,W) or (H,W,3) or (3,H,W), np or torch
    Returns refined_probs: (C,H,W), renormalized per pixel.
    """
    assert probs_CHW.ndim == 3, "Expected (C,H,W)"
    device = probs_CHW.device
    dtype = probs_CHW.dtype
    C, H, W = probs_CHW.shape

    I = _prepare_gray_guidance(denoised_x0, (H, W), device=device, dtype=torch.float32)  # (1,1,H,W)
    # print("I.shape:", I.shape)

    p = probs_CHW.to(dtype=torch.float32).unsqueeze(1)  # (C,1,H,W)
    # print("p.shape:", p.shape)
    q = _guided_filter_gray_batch(p, I, radius=radius, eps=eps)  # (C,1,H,W)
    # print("q.shape:", q.shape)
    q = q.squeeze(1)  # (C,H,W)

    # Clamp and renormalize per pixel
    q = q.clamp_min(0.0)
    denom = q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    q = q / denom

    return q.to(dtype=dtype)

@torch.no_grad()
def process_attention_to_masks_v3(
    concept_indices_list,
    attention_maps: torch.Tensor,   # (num_concepts, H, W), last concept is background ("all tokens")
    steps,
    output_size=(128, 128),
    denoised_x0_t = None,
    *,
    p_low=0.10, p_high=0.90,
    fg_bg_blend=0.5,        # blend weight for provided BG vs residual free-space
    suppression_rho=0.30,   # mutual suppression strength (0..1)
    sharpness=1.0 #3.0           # higher = sharper logits before sparsemax
):
    """
    Returns:
        masks: (num_concepts, output_H, output_W), soft, sharper, sum==1 per pixel.
    Notes:
        - concept_indices_list[-1] is treated as background (all tokens).
        - Handles small [~0.01, 0.3] ranges via percentile normalization and sharpening.
        - Uses sparsemax for sharper, sum-to-1 partitioning (soft, possibly sparse).
    """

    # aggregate concept attentions
    attention_maps = _concept_attns(attention_maps, concept_indices_list)

    attention_maps = attention_maps.to(dtype=torch.float32)
    attention_maps = torch.clamp(attention_maps, min=0)

    device = attention_maps.device
    dtype = attention_maps.dtype
    C, H, W = attention_maps.shape

    attention_masks = attention_maps  # (C,H,W)

    if steps["upsample_before_processing"]:
        attention_masks = _upsample_and_smooth(attention_masks, output_size=output_size, 
                                               upsample=True,
                                               use_gaussian_smoothing=steps["use_gaussian_smoothing"])
    
    if steps["use_indv_normalization"]:
        attention_masks = _individual_normalization(attention_masks, p_low=p_low, p_high=p_high)
    
    fg = attention_masks[:-1]           # (C-1,H,W)
    bg_provided = attention_masks[-1:]
    if steps["use_bg_blend"]:
        bg_blend = _bg_blend(fg, bg_provided, fg_bg_blend=fg_bg_blend)  # (1,H,W)
        attention_masks = torch.cat([fg, bg_blend], dim=0)  # (C,H,W)

    if steps["use_mutual_suppression"]:
        attention_masks = _mutual_suppression(attention_masks, suppression_rho=suppression_rho)

    if not steps["upsample_before_processing"]:
        attention_masks = _upsample_and_smooth(attention_masks, output_size=output_size, 
                                               upsample=True,
                                               use_gaussian_smoothing=steps["use_gaussian_smoothing"])
    
    if steps["use_sparsemax"]:
        probs_up = _concepts_sparsemax(attention_masks, temp=sharpness)
    else:
        probs_up = _concepts_softmax(attention_masks, temp=sharpness)   

    if steps["use_guided_filter"]:
        assert denoised_x0_t is not None, "denoised_x0_t is required for guided filtering."
        probs_up = guided_filter_masks(probs_up, denoised_x0_t, radius=2, eps=1e-3)

    probs_up = torch.clamp(probs_up, min=0)
    denom = probs_up.sum(dim=0, keepdim=True).clamp_min(1e-8)
    probs_up = probs_up / denom

    # #Todo: also return concept_attn stats and binary masks
    return probs_up.unsqueeze(1).to(dtype=dtype, device=device), None, None  # (C, output_H, output_W)
    


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def get_concept_indices(tokenizer, prompt, concepts):
    concept_indices_list = []
    index_offset = 1
    input_ids = tokenize_prompt(tokenizer, prompt)[0]  # shape: (77,)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    tokens = [t for t in tokens if t not in tokenizer.all_special_tokens and not t.startswith("<|")]
    tokens = [t.lower().strip("Ġ").strip("##") for t in tokens]  # Clean up for both CLIP/BERT styles

    for i, concept in enumerate(concepts):
        concept_ids = tokenize_prompt(tokenizer, concept)[0]
        concept_tokens = tokenizer.convert_ids_to_tokens(concept_ids)
        concept_tokens = [t for t in concept_tokens if t not in tokenizer.all_special_tokens and not t.startswith("<|")]
        concept_tokens = [t.lower().strip("Ġ").strip("##") for t in concept_tokens]
        concept_indices = find_subsequence_indices(tokens, concept_tokens) # Todo: only returns the first match, need to handle multiple matches
        concept_indices = [i + index_offset for i in concept_indices]  # Adjust indices to account for the first non-informative tokens
        concept_indices_list.append(concept_indices)

    return concept_indices_list


def postprocess_mask_for_viz(mask, h, w, mask_path):
    assert mask.ndim == 2, "Input Mask array should be 2D."
    mask = mask.cpu().numpy()
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    #mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    fig = plt.figure()
    plt.imshow(mask, cmap='magma')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(mask_path, bbox_inches='tight', dpi=300)
    plt.close()

def get_all_noun_chunks(text, parser):
    doc = parser(text)
    noun_chunks = []
    for sentence in doc.sentences:
        words = sentence.words
        for word in words:
            # A noun phrase head is often a NOUN or PROPN
            if word.upos in ("NOUN", "PROPN"):
                # Collect modifiers and the head noun
                chunk_tokens = []
                for w in words:
                    if (
                        w.head == word.id and w.deprel in ("amod", "compound", "det")
                    ) or w.id == word.id:
                        chunk_tokens.append(w.text)
                # Sort by token order
                chunk_tokens = sorted(chunk_tokens, key=lambda t: text.index(t))
                chunk_text = " ".join(chunk_tokens)
                if chunk_text not in noun_chunks:
                    noun_chunks.append(chunk_text)

    return noun_chunks

def get_non_head_noun_chunks(text, parser):
    """
    warning: head nouns are NOUNs before 1st
    """
    doc = parser(text)
    non_head_chunks = []

    for sentence in doc.sentences:
        # Find the main syntactic head noun of the sentence (if any)
        head_noun_id = None
        for w in sentence.words:
            if w.head == 0 and w.upos in ("NOUN", "PROPN"):
                head_noun_id = w.id
                break

        # Now extract noun chunks excluding that head noun
        for w in sentence.words:
            if w.upos in ("NOUN", "PROPN"):
                # Skip if this is the sentence head noun
                if w.id == head_noun_id:
                    continue

                # Collect modifiers + head noun
                chunk_tokens = []
                for ww in sentence.words:
                    if (
                        ww.head == w.id and ww.deprel in ("amod", "compound", "det")
                    ) or ww.id == w.id:
                        chunk_tokens.append(ww)

                # Sort tokens by their position in sentence
                chunk_tokens = sorted(chunk_tokens, key=lambda t: t.id)
                chunk_text = " ".join(t.text for t in chunk_tokens)

                if chunk_text not in non_head_chunks:
                    non_head_chunks.append(chunk_text)
    
    return non_head_chunks


def remove_adjectives(text, parser):
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                text[i] = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "ADJ"])
            else:
                # spaCy format
                text[i] = " ".join([token.text for token in doc if token.pos_ != "ADJ"])
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            text = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "ADJ"])
        else:
            # spaCy format
            text = " ".join([token.text for token in doc if token.pos_ != "ADJ"])
    return text

def remove_articles_from_beginning(text, parser):
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                tokens = [token.text for sent in doc.sentences for token in sent.words]
                pos_tags = [token.upos for sent in doc.sentences for token in sent.words]
                
                # Remove articles from beginning
                start_idx = 0
                while start_idx < len(pos_tags) and pos_tags[start_idx] == "DET":
                    start_idx += 1
                
                text[i] = " ".join(tokens[start_idx:])
            else:
                # spaCy format
                tokens = [token.text for token in doc]
                pos_tags = [token.pos_ for token in doc]
                
                # Remove articles from beginning
                start_idx = 0
                while start_idx < len(pos_tags) and pos_tags[start_idx] == "DET":
                    start_idx += 1
                
                text[i] = " ".join(tokens[start_idx:])
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            tokens = [token.text for sent in doc.sentences for token in sent.words]
            pos_tags = [token.upos for sent in doc.sentences for token in sent.words]
            
            # Remove articles from beginning
            start_idx = 0
            while start_idx < len(pos_tags) and pos_tags[start_idx] == "DET":
                start_idx += 1
            
            text = " ".join(tokens[start_idx:])
        else:
            # spaCy format
            tokens = [token.text for token in doc]
            pos_tags = [token.pos_ for token in doc]
            
            # Remove articles from beginning
            start_idx = 0
            while start_idx < len(pos_tags) and pos_tags[start_idx] == "DET":
                start_idx += 1
            
            text = " ".join(tokens[start_idx:])
    
    return text

def remove_articles(text, parser):
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                text[i] = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "DET"])
            else:
                # spaCy format
                text[i] = " ".join([token.text for token in doc if token.pos_ != "DET"])
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            text = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "DET"])
        else:
            # spaCy format
            text = " ".join([token.text for token in doc if token.pos_ != "DET"])
    return text

def remove_conjunctions(text, parser):
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                text[i] = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "CCONJ"])
            else:
                # spaCy format
                text[i] = " ".join([token.text for token in doc if token.pos_ != "CCONJ"])
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            text = " ".join([token.text for sent in doc.sentences for token in sent.words if token.upos != "CCONJ"])
        else:
            # spaCy format
            text = " ".join([token.text for token in doc if token.pos_ != "CCONJ"])
    return text

def remove_conjunctions_from_beginning(text, parser):
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                tokens = [token.text for sent in doc.sentences for token in sent.words]
                pos_tags = [token.upos for sent in doc.sentences for token in sent.words]
                
                # Remove conjunctions from beginning
                start_idx = 0
                while start_idx < len(pos_tags) and pos_tags[start_idx] == "CCONJ":
                    start_idx += 1
                
                text[i] = " ".join(tokens[start_idx:])
            else:
                # spaCy format
                tokens = [token.text for token in doc]
                pos_tags = [token.pos_ for token in doc]
                
                # Remove conjunctions from beginning
                start_idx = 0
                while start_idx < len(pos_tags) and pos_tags[start_idx] == "CCONJ":
                    start_idx += 1
                
                text[i] = " ".join(tokens[start_idx:])
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            tokens = [token.text for sent in doc.sentences for token in sent.words]
            pos_tags = [token.upos for sent in doc.sentences for token in sent.words]
            
            # Remove conjunctions from beginning
            start_idx = 0
            while start_idx < len(pos_tags) and pos_tags[start_idx] == "CCONJ":
                start_idx += 1
            
            text = " ".join(tokens[start_idx:])
        else:
            # spaCy format
            tokens = [token.text for token in doc]
            pos_tags = [token.pos_ for token in doc]
            
            # Remove conjunctions from beginning
            start_idx = 0
            while start_idx < len(pos_tags) and pos_tags[start_idx] == "CCONJ":
                start_idx += 1
            
            text = " ".join(tokens[start_idx:])
    
    return text

def remove_wh_words(text, parser):
    excluded = ["who", "whom", "whose", "what", "which", "when", "where", "why"]
    if isinstance(text, list):
        for i in range(len(text)):
            doc = parser(text[i])
            # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
            if hasattr(doc, 'sentences'):
                # Stanza format
                text[i] = " ".join([token.text for sent in doc.sentences for token in sent.words if token.text not in excluded])
            else:
                # spaCy format
                text[i] = " ".join([token.text for token in doc if token.text not in excluded])
            text[i] = text[i].strip()
    else:
        doc = parser(text)
        # Check if it's Stanza (has .sentences) or spaCy (direct iteration)
        if hasattr(doc, 'sentences'):
            # Stanza format
            text = " ".join([token.text for sent in doc.sentences for token in sent.words if token.text not in excluded])
        else:
            # spaCy format
            text = " ".join([token.text for token in doc if token.text not in excluded])
            text = text.strip()
        
    return text

def find_super_strings(st, chunks):

    super_strings = []
    
    for chunk in chunks:
        if st in chunk:
            super_strings.append(chunk)
    
    return super_strings

def find_sub_strings(st, chunks):

    sub_strings = []
    
    for chunk in chunks:
        if chunk in st:
            sub_strings.append(chunk)
    
    return sub_strings

# def get_unique_noun_chunks(text, parser):
#     all_chunks = get_all_noun_chunks(text, parser)
#     unique_chunks = []
#     for chunk in all_chunks:
#         if len(find_sub_strings(chunk, all_chunks)) == 1:
#             unique_chunks.append(chunk)
#     return unique_chunks

def remove_duplicate_chunks(chunks):
    unique_chunks = []
    for chunk in chunks:
        if len(find_sub_strings(chunk, chunks)) == 1:
            unique_chunks.append(chunk)
    return unique_chunks

def get_prompts_and_concepts_fine(prompt, parser,
                             remove_adj_from_contrastive_prompts=False):
    
    non_head_noun_chunks = get_non_head_noun_chunks(prompt, parser)
    non_head_noun_chunks =  remove_duplicate_chunks(non_head_noun_chunks)
    non_head_noun_chunks = remove_articles(non_head_noun_chunks, parser)
    concepts = non_head_noun_chunks


    all_noun_chunks = get_all_noun_chunks(prompt, parser)
    prompts = remove_duplicate_chunks(all_noun_chunks)
    if remove_adj_from_contrastive_prompts:
        prompts = remove_adjectives(prompts, parser)
    return prompts, concepts

def get_prompts_and_concepts_coarse(prompt, parser,
                             remove_adj_from_contrastive_prompts=False,
                             remove_art_from_contrastive_prompts=False,
                             remove_art_from_concepts=True):

    concepts = get_all_noun_chunks(prompt, parser)
    concepts =  remove_duplicate_chunks(concepts)
    if remove_art_from_concepts:
        concepts = remove_articles(concepts, parser)

    all_noun_chunks = get_all_noun_chunks(prompt, parser)
    prompts = remove_duplicate_chunks(all_noun_chunks)
    if remove_adj_from_contrastive_prompts:
        prompts = remove_adjectives(prompts, parser)
    if remove_art_from_contrastive_prompts:
        prompts = remove_articles(prompts, parser)
    return prompts, concepts


# to extract only the head nouns, excluding compound modifiers for cprompts 
def extract_base_nouns(text, nlp):
    """
    Extract only the head nouns, excluding compound modifiers
    
    For "a star shaped drum is being carried by a wooly monkfish"
    Returns: ['drum', 'monkfish']  (NOT 'star')
    """
    doc = nlp(text)
    base_nouns = []
    
    for chunk in doc.noun_chunks:
        # Get the syntactic head (root) of the noun phrase
        head = chunk.root
        
        # Only include if it's the main noun, not a compound modifier
        if head.pos_ == "NOUN":
            # Check if this noun is modifying another noun (compound)
            # If it's a compound modifier, skip it
            if head.dep_ != "compound":
                base_nouns.append(head.text)
    
    return base_nouns
