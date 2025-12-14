import torch
import torch.nn.functional as F
from torchvision import transforms, models

from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    SegformerFeatureExtractor,           
    SegformerForSemanticSegmentation,    
    AutoProcessor,                       
    AutoModel,
    pipeline,
    BitsAndBytesConfig,      
)
from PIL import Image
import numpy as np
import warnings

# Suppress specific warnings if needed, e.g., from transformers or torchvision
warnings.filterwarnings('ignore', category=UserWarning, message='.*?torchvision.*?')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*?transformers.*?')

# --- Model Loading Functions ---

SIMILARITY_DIMENSIONS = [
    "content similarity (objects, scenes, or subjects present)",
    "color similarity (dominant colors and tones)",
    "structural similarity (layout, shapes, and composition)",
    "texture similarity (patterns and surface details)"
]



def load_reward_model(reward_type, device=0):
    """Loads the specified reward model and its processor."""
    print(f"Loading reward model: {reward_type}")
    if reward_type == "depth":
        model, processor = load_depth_reward()
    elif reward_type == "inception":
        model, processor = load_inception_reward()
    elif reward_type == "segmentation":
        model, processor = load_segmentation_reward()
    elif reward_type == "image_psnr":
        model, processor = load_image_psnr_reward()
    elif reward_type == "clip":
        model, processor = load_clip_reward()
    elif reward_type == "llava":
        model, processor = load_llava_reward(device)
    elif reward_type == "qwen_vl":
        model, processor = load_qwen_vl_reward(device)
    elif reward_type == "dino":  # New case for DINO
        model, processor = load_dino_reward()
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    print(f"Reward model '{reward_type}' loaded successfully.")
    return model, processor

def load_dino_reward():
    """Loads DINOv2 model and processor."""
    model_name = "facebook/dinov2-base"  # Options: dinov2-small, dinov2-base, dinov2-large, dinov2-giant
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, processor


def load_llava_reward(device):
    """Loads LLaVA model with quantization or on CPU if necessary."""
    try:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = pipeline(
            "image-text-to-text",
            model="llava-hf/llama3-llava-next-8b-hf",
            quantization_config=quantization_config,
            device=device
        )
        print("LLaVA loaded with 4-bit quantization on GPU.")
    except RuntimeError as e:
        print(f"GPU loading failed: {e}. Falling back to CPU.")
        model = pipeline(
            "image-text-to-text",
            model="llava-hf/llama3-llava-next-8b-hf",
            device=-1 
        )
        print("LLaVA loaded on CPU.")
    processor = None
    return model, processor




def load_depth_reward():
    """Loads Depth Anything model and processor."""
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    return model, processor

def load_inception_reward():
    """Loads Inception V3 model and processor."""
    model = models.inception_v3(pretrained=True, aux_logits=True, transform_input=False) # Use aux_logits=False for simpler output
    # Define processor matching Inception V3 requirements
    processor = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC), # Match Inception input size
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, processor

def load_segmentation_reward():
    """Loads SegFormer model and processor for semantic segmentation."""
    model_name = "nvidia/segformer-b4-finetuned-ade-512-512"
    processor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    # Store num_classes for Dice calculation
    model.config.num_classes = model.config.num_labels # Store for later use
    return model, processor

def load_image_psnr_reward():
    """Loads processor for Image PSNR (no actual model needed)."""
    # PSNR compares raw pixel values, typically after converting to tensor [0, 1]
    processor = transforms.Compose([
        transforms.ToTensor() # Converts PIL image [0, 255] to Tensor [0, 1]
    ])
    # No network model required for PSNR calculation itself
    return None, processor # Return None for the model

def load_clip_reward():
    """Loads CLIP model and processor."""
    # Common choice: "openai/clip-vit-large-patch14" or "openai/clip-vit-base-patch32"
    model_name = "openai/clip-vit-large-patch14"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, processor

# --- Reward Calculation Functions ---

def calculate_reward(reward_type, reward_model, reward_model_processor, model_pred, target, device):
    """Calculates rewards based on the specified reward type."""
    # Ensure inputs are on the correct device and detached if necessary
    model_pred = model_pred.clamp(0, 1) if model_pred.min() < 0 else model_pred
    target = target.clamp(0, 1) if target.min() < 0 else target

    if reward_type == "depth":
        return calculate_depth_reward(reward_model, reward_model_processor, model_pred, target, device)
    elif reward_type == "inception":
        return calculate_inception_reward(reward_model, reward_model_processor, model_pred, target, device)
    elif reward_type == "segmentation":
        return calculate_segmentation_reward(reward_model, reward_model_processor, model_pred, target, device)
    elif reward_type == "image_psnr":
        return calculate_image_psnr_reward(reward_model_processor, model_pred, target, device)
    elif reward_type == "clip":
        return calculate_clip_reward(reward_model, reward_model_processor, model_pred, target, device)
    elif reward_type == "llava":
        return calculate_llava_reward(reward_model, reward_model_processor, model_pred, target, device)
    elif reward_type == "qwen_vl":
        return calculate_qwen_vl_reward(reward_model, reward_model_processor, model_pred, target, device)
    elif reward_type == "dino":  # New case for DINO
        return calculate_dino_reward(reward_model, reward_model_processor, model_pred, target, device)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")
    

def calculate_llava_reward(reward_model, reward_model_processor, model_pred, target, device):
    """Calculates reward based on LLaVA's multi-dimensional similarity assessment with retry logic."""
    batch_size = model_pred.shape[0]
    reward_scores = []
    to_pil = transforms.ToPILImage()
    max_retries = 5 

    for i in range(batch_size):
        pred_pil = to_pil(model_pred[i].cpu())
        target_pil = to_pil(target[i].cpu())

        dimension_scores = []
        for dimension in SIMILARITY_DIMENSIONS:
            prompt = (
                f"Evaluate the {dimension} between these two images on a scale from 0 to 100, "
                f"where 0 means completely dissimilar and 100 means identical. Provide only the numerical score."
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pred_pil},
                        {"type": "image", "image": target_pil},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            score = None
            for attempt in range(max_retries):
                out = reward_model(text=messages, max_new_tokens=10)
                generated_text = out[0]["generated_text"]
                print(generated_text)
                try:
                    score = float(generated_text)
                    score = max(0.0, min(100.0, score))
                    break  
                except (ValueError, IndexError):
                    print(f"Failed to parse LLaVA output for dimension '{dimension}', batch {i}, attempt {attempt + 1}/{max_retries}. Output: '{generated_text}'")
                    if attempt == max_retries - 1:
                        print(f"Max retries reached for dimension '{dimension}', batch {i}. Using fallback score 50.0.")
                        score = 50.0 
            
            dimension_scores.append(score)
        
        mean_score = np.mean(dimension_scores)
        reward_scores.append(mean_score)

    rewards = torch.tensor(reward_scores, dtype=torch.float32, device=device).unsqueeze(1)
    return rewards


def calculate_dino_reward(reward_model, reward_model_processor, model_pred, target, device):
    """Calculates reward based on DINOv2 image feature similarity."""
    batch_size = model_pred.shape[0]
    pred_features_list = []
    target_features_list = []
    to_pil = transforms.ToPILImage()

    reward_model.eval()
    reward_model.to(device)  # Ensure model is on the correct device

    for i in range(batch_size):
        pred_pil = to_pil(model_pred[i].cpu())
        target_pil = to_pil(target[i].cpu())

        # Process images with DINOv2 processor
        inputs = reward_model_processor(images=[pred_pil, target_pil], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            # Get DINOv2 features
            outputs = reward_model(**inputs)
            image_features = outputs.last_hidden_state[:, 0, :]  # Use CLS token (index 0) for global representation

        # Normalize features for cosine similarity
        image_features = F.normalize(image_features, p=2, dim=-1)

        # Separate features
        pred_features_list.append(image_features[0].unsqueeze(0))  # [1, embed_dim]
        target_features_list.append(image_features[1].unsqueeze(0))  # [1, embed_dim]

    # Concatenate features for the whole batch
    pred_features = torch.cat(pred_features_list, dim=0)  # [B, embed_dim]
    target_features = torch.cat(target_features_list, dim=0)  # [B, embed_dim]

    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(pred_features.float(), target_features.float(), dim=1)

    # Scale similarity from [-1, 1] to [0, 100]
    rewards = (cos_sim + 1.0) * 50.0

    # Return rewards reshaped to [B, 1]
    return rewards.unsqueeze(1)

def calculate_qwen_vl_reward(reward_model, reward_model_processor, model_pred, target, device):
    """Calculates reward based on Qwen2.5-VL-3B-Instruct's multi-dimensional similarity assessment."""
    batch_size = model_pred.shape[0]
    reward_scores = []
    to_pil = transforms.ToPILImage()
    max_retries = 5
    resize = transforms.Resize((224, 224)) 

    for i in range(batch_size):
        pred_pil = to_pil(resize(model_pred[i].cpu())) 
        target_pil = to_pil(resize(target[i].cpu()))

        dimension_scores = []
        for dimension in SIMILARITY_DIMENSIONS:
            prompt = (
                f"Evaluate the {dimension} between these two images on a scale from 0 to 100, "
                f"where 0 means completely dissimilar and 100 means identical. Provide only the numerical score."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pred_pil},
                        {"type": "image", "image": target_pil},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            score = None
            for attempt in range(max_retries):
                try:
                    text = reward_model_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, _ = process_vision_info(messages)
                    inputs = reward_model_processor(
                        text=[text],
                        images=image_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to(device)
                    with torch.no_grad():
                        generated_ids = reward_model.generate(**inputs, max_new_tokens=5)
                    generated_ids = generated_ids[0][len(inputs.input_ids[0]):]
                    generated_text = reward_model_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    # print(f"Generated text for '{dimension}', batch {i}: '{generated_text}'")
                    score = float(generated_text)
                    score = max(0.0, min(100.0, score))
                    break
                except (ValueError, IndexError) as e:
                    print(f"Failed to parse Qwen output for '{dimension}', batch {i}, attempt {attempt + 1}/{max_retries}. Error: {e}")
                    if attempt == max_retries - 1:
                        print(f"Max retries reached for '{dimension}', batch {i}. Using fallback score 50.0.")
                        score = 50.0
            dimension_scores.append(score)
        mean_score = np.mean(dimension_scores)
        reward_scores.append(mean_score)
        torch.cuda.empty_cache()

    rewards = torch.tensor(reward_scores, dtype=torch.float32, device=device).unsqueeze(1)
    return rewards

def calculate_inception_reward(reward_model, reward_model_processor, model_pred, target, device):
    """Calculates reward based on feature similarity using Inception V3."""
    batch_size = model_pred.shape[0]
    pred_features_list = []
    target_features_list = []
    to_pil = transforms.ToPILImage()

    reward_model.eval() # Ensure model is in eval mode

    for i in range(batch_size):
        # Convert tensor [C, H, W] (range [0, 1]) to PIL Image
        pred_pil = to_pil(model_pred[i].cpu())
        target_pil = to_pil(target[i].cpu())

        # Apply Inception V3 preprocessing
        pred_input = reward_model_processor(pred_pil).unsqueeze(0).to(device)
        target_input = reward_model_processor(target_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get features (output before final FC layer)
            pred_feature = reward_model(pred_input)
            target_feature = reward_model(target_input)

        pred_features_list.append(pred_feature)
        target_features_list.append(target_feature)

    # Concatenate features for the whole batch
    pred_features = torch.cat(pred_features_list, dim=0) # Shape: [B, num_features]
    target_features = torch.cat(target_features_list, dim=0) # Shape: [B, num_features]

    # Calculate cosine similarity (returns tensor of shape [B])
    cos_sim = F.cosine_similarity(pred_features, target_features, dim=1)

    # Scale similarity from [-1, 1] to [0, 100]
    rewards = (cos_sim + 1.0) * 50.0

    # Return rewards reshaped to [B, 1]
    return rewards.unsqueeze(1)


def calculate_depth_reward(reward_model, reward_model_processor, model_pred, target, device):
    """Calculates the PSNR-based reward using a depth estimation model."""
    batch_size = model_pred.shape[0]
    pred_depths_list = []
    target_depths_list = []
    to_pil = transforms.ToPILImage()

    reward_model.eval() # Ensure model is in eval mode

    for i in range(batch_size):
        # Convert tensor [C, H, W] (range [0, 1]) to PIL Image
        pred_pil = to_pil(model_pred[i].float().cpu())
        target_pil = to_pil(target[i].float().cpu())

        # Process images with Depth Anything processor
        # Note: Depth Anything expects standard ImageNet normalization internally if using AutoImageProcessor
        # Check if the processor handles normalization or if PIL image is sufficient.
        # Assuming processor handles conversion/normalization from PIL:
        inputs_pred = reward_model_processor(images=pred_pil, return_tensors="pt").to(device)
        inputs_target = reward_model_processor(images=target_pil, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs_pred = reward_model(**inputs_pred)
            outputs_target = reward_model(**inputs_target)

        # Post-process to get depth maps matching original image size
        # Use the shape from the *original* input tensors
        height, width = model_pred.shape[-2:]
        post_processed_pred = reward_model_processor.post_process_depth_estimation(
            outputs_pred, target_sizes=[(height, width)] # Use original H, W
        )[0]["predicted_depth"] # Shape [H, W]
        post_processed_target = reward_model_processor.post_process_depth_estimation(
            outputs_target, target_sizes=[(height, width)]
        )[0]["predicted_depth"] # Shape [H, W]

        # Normalize depth maps individually to range [0, 1] for consistent PSNR calculation
        pred_depth_norm = (post_processed_pred - post_processed_pred.min()) / \
                          (post_processed_pred.max() - post_processed_pred.min() + 1e-8)
        target_depth_norm = (post_processed_target - post_processed_target.min()) / \
                            (post_processed_target.max() - post_processed_target.min() + 1e-8)

        pred_depths_list.append(pred_depth_norm.unsqueeze(0)) # Add batch dim
        target_depths_list.append(target_depth_norm.unsqueeze(0)) # Add batch dim

    # Concatenate depth maps for the whole batch
    pred_depths = torch.cat(pred_depths_list, dim=0).float() # Shape: [B, H, W]
    target_depths = torch.cat(target_depths_list, dim=0).float() # Shape: [B, H, W]

    # Calculate MSE per image in the batch
    # Mean over spatial dimensions (H, W) -> shape [B]
    mse = torch.mean((pred_depths - target_depths) ** 2, dim=[1, 2])

    # Calculate PSNR using the formula: 10 * log10(max_val^2 / MSE)
    # Since depth maps are normalized to [0, 1], max_val = 1.0
    max_val_sq = 1.0**2
    # Add small epsilon to avoid log10(0) or division by zero
    psnr = 10 * torch.log10(max_val_sq / (mse + 1e-8))

    # Handle potential infinite PSNR if MSE is exactly 0 (perfect match)
    # You might clamp the reward or return a fixed high value
    psnr = torch.clamp(psnr, min=0) # Ensure PSNR is non-negative

    # Return rewards reshaped to [B, 1]
    return psnr.unsqueeze(1)


def _compute_dice_score(mask1, mask2, num_classes, smooth=1e-8):
    """Computes mean Dice score for a single pair of segmentation masks."""
    # Ensure masks are on the same device and are integer type
    mask1 = mask1.long()
    mask2 = mask2.long()

    return (mask1 == mask2).float().mean()

def calculate_segmentation_reward(reward_model, reward_model_processor, model_pred, target, device):
    """Calculates reward based on segmentation mask similarity (Dice score)."""
    batch_size = model_pred.shape[0]
    dice_scores_list = []
    to_pil = transforms.ToPILImage()
    num_classes = reward_model.config.num_classes # Get num_classes stored during loading

    reward_model.eval()

    for i in range(batch_size):
        pred_pil = to_pil(model_pred[i].cpu())
        target_pil = to_pil(target[i].cpu())

        # Process images with SegFormer feature extractor
        inputs_pred = reward_model_processor(images=pred_pil, return_tensors="pt").to(device)
        inputs_target = reward_model_processor(images=target_pil, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs_pred = reward_model(**inputs_pred)
            outputs_target = reward_model(**inputs_target)

        # Get logits [1, num_classes, H_feat, W_feat]
        logits_pred = outputs_pred.logits
        logits_target = outputs_target.logits

        mask_pred = torch.argmax(logits_pred, dim=1).squeeze(0)
        mask_target = torch.argmax(logits_target, dim=1).squeeze(0)

        # # Use the shape from the *original* input tensors
        # height, width = model_pred.shape[-2:]
        # upsampled_logits_pred = F.interpolate(logits_pred, size=(height, width), mode='bilinear', align_corners=False)
        # upsampled_logits_target = F.interpolate(logits_target, size=(height, width), mode='bilinear', align_corners=False)

        # # Get segmentation mask by taking argmax [1, H, W] -> [H, W]
        # mask_pred = torch.argmax(upsampled_logits_pred, dim=1).squeeze(0)
        # mask_target = torch.argmax(upsampled_logits_target, dim=1).squeeze(0)

        # Compute Dice score for the pair of masks
        dice_score = _compute_dice_score(mask_pred, mask_target, num_classes)
        dice_scores_list.append(dice_score)

    # Stack scores into a tensor [B]
    dice_scores = torch.stack(dice_scores_list)

    # Scale Dice score [0, 1] to reward range [0, 100]
    rewards = dice_scores * 100.0

    # Return rewards reshaped to [B, 1]
    return rewards.unsqueeze(1)


def calculate_image_psnr_reward(reward_model_processor, model_pred, target, device):
    """Calculates reward based on raw Image PSNR."""
    # model_pred, target are expected to be tensors [B, C, H, W] in range [0, 1]
    # Ensure they are on the correct device
    model_pred = model_pred.to(device)
    target = target.to(device)

    # Ensure images have the same dimensions
    if model_pred.shape != target.shape:
        # Resize target to match model_pred dimensions
        target = F.interpolate(target, size=model_pred.shape[-2:], mode='bilinear', align_corners=False)

    # Calculate MSE per image in the batch (mean over C, H, W)
    mse = torch.mean((model_pred - target) ** 2, dim=[1, 2, 3])

    # Calculate PSNR: 10 * log10(max_val^2 / MSE)
    # max_val is 1.0 for tensors in [0, 1] range
    max_val_sq = 1.0 ** 2
    psnr = 10 * torch.log10(max_val_sq / (mse + 1e-8)) # Add epsilon for stability

    # Clamp PSNR (e.g., max value if MSE is 0)
    # PSNR can technically be infinite if MSE=0. We can cap it.
    psnr = torch.clamp(psnr, min=0, max=100.0) # Cap at 100 for practical reward range

    # Return rewards reshaped to [B, 1]
    return psnr.unsqueeze(1)


def calculate_clip_reward(reward_model, reward_model_processor, model_pred, target, device):
    """Calculates reward based on CLIP image feature similarity."""
    batch_size = model_pred.shape[0]
    pred_features_list = []
    target_features_list = []
    to_pil = transforms.ToPILImage()

    reward_model.eval()

    for i in range(batch_size):
        pred_pil = to_pil(model_pred[i].cpu())
        target_pil = to_pil(target[i].cpu())

        # Process PIL images with CLIP processor
        # Processor handles resizing, normalization specific to CLIP
        inputs = reward_model_processor(images=[pred_pil, target_pil], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            # Get image features [2, embedding_dim]
            image_features = reward_model.get_image_features(**inputs)

        # Normalize features (important for cosine similarity)
        image_features = F.normalize(image_features, p=2, dim=-1)

        # Separate features
        pred_features_list.append(image_features[0].unsqueeze(0)) # Keep batch dim [1, embed_dim]
        target_features_list.append(image_features[1].unsqueeze(0)) # Keep batch dim [1, embed_dim]

    # Concatenate features for the whole batch
    pred_features = torch.cat(pred_features_list, dim=0) # Shape: [B, embed_dim]
    target_features = torch.cat(target_features_list, dim=0) # Shape: [B, embed_dim]

    # Calculate cosine similarity [B]
    # Ensure features are float32 for cosine similarity if needed
    cos_sim = F.cosine_similarity(pred_features.float(), target_features.float(), dim=1)

    # Scale similarity [-1, 1] to reward range [0, 100]
    rewards = (cos_sim + 1.0) * 50.0

    # Return rewards reshaped to [B, 1]
    return rewards.unsqueeze(1)


# --- Example Usage ---
if __name__ == '__main__':
    # Determine device
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {current_device}")

    # Create dummy input tensors (Batch=2, Channels=3, Height=64, Width=64)
    # Tensors should be in the range [0, 1] for PIL conversion and some models
    dummy_pred = torch.rand(2, 3, 64, 64, device=current_device)
    dummy_target = torch.rand(2, 3, 64, 64, device=current_device)
    # Make one target slightly different from its pred for non-zero scores
    dummy_target[1] = dummy_target[1] * 0.8 + 0.1

    # --- Test each reward function ---
    reward_types_to_test = ["depth", "inception", "segmentation", "image_psnr", "clip"]

    for r_type in reward_types_to_test:
        print(f"\n--- Testing Reward Type: {r_type} ---")
        try:
            # Load model and processor
            model, processor = load_reward_model(r_type, current_device)

            # Calculate reward
            rewards = calculate_reward(r_type, model, processor, dummy_pred, dummy_target, current_device)

            print(f"Reward calculation successful for '{r_type}'.")
            print(f"Input batch size: {dummy_pred.shape[0]}")
            print(f"Output rewards shape: {rewards.shape}")
            print(f"Sample rewards: {rewards.flatten().cpu().numpy()}")

            # Clean up GPU memory (optional, good practice in loops)
            del model
            del processor
            del rewards
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error testing reward type '{r_type}': {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            # Clean up even if error occurred
            if 'model' in locals() and model is not None : del model
            if 'processor' in locals() and processor is not None: del processor
            if 'rewards' in locals(): del rewards
            if torch.cuda.is_available():
                torch.cuda.empty_cache()