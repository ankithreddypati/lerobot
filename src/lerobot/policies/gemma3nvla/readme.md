configuration_gemma3nvla.py: Config class with @PreTrainedConfig.register_subclass("gemma3nvla")
modeling_gemma3nvla.py: Main policy wrapper (like SmolVLAPolicy → Gemma3nVLAPolicy)
gemma3n_with_expert.py: Core architecture (like SmolVLMWithExpertModel → Gemma3nWithExpertModel



# Training with new policy
python -m lerobot.scripts.train \
--policy.type=gemma3nvla \
--dataset.repo_id=your_dataset \
--batch_size=32 \
--steps=200000


# Gemma3nVLA: Vision-Language-Action Model

This implementation creates a new VLA (Vision-Language-Action) policy based on Google's Gemma3n-E2B model, replacing the SmolVLM2-500M backbone in SmolVLA with the more powerful 2B parameter Gemma3n model.

## What We Built

### **Architecture Overview**
```
┌──────────────────────────────┐
│                 actions      │
│                    ▲         │
│ ┌─────────┐      ┌─|────┐    │
│ |         │────► │      │    │
│ | Gemma3n │ kv   │      │    │
│ |  E2B    │────► │Action│    │
│ | (2B)    │cache │Expert│    |
│ │         │────► |      │    │
│ │         │      │      │    │
│ └▲──▲───▲─┘      └───▲──┘    |
│  │  |   |            │       |
│  |  |   |          noise     │
│  │  │ state                  │
│  │ language tokens           │
│  image(s)                    │
│  + audio          │
└──────────────────────────────┘
```

### **Key Improvements Over SmolVLA**
- **4x Larger Model**: 500M → 2B parameters
- **Better Vision**: MobileNet v5 encoder (768×768 native resolution)
- **Advanced Architecture**: AltUp, LAuReL, Per-Layer Embeddings, Activation Sparsity
- **Multimodal Ready**: Native support for audio inputs (can be added later)
- **Better Language Understanding**: Enhanced text processing capabilities

---

## File Structure

### **Folder**: `src/lerobot/policies/gemma3nvla/`

```
gemma3nvla/
├── configuration_gemma3nvla.py    # Configuration management
├── modeling_gemma3nvla.py         # Main policy wrapper  
└── gemma3n_with_expert.py        # VLM + Action Expert architecture
```

### **Factory Integration**: `src/lerobot/policies/factory.py`
Added three lines to enable `--policy.type=gemma3nvla`

---

## File Details

### **1. `configuration_gemma3nvla.py`**
**Purpose**: Manages all configuration parameters for the Gemma3nVLA policy

**Key Components**:
- **Model Selection**: `vlm_model_name = "google/gemma-3n-e2b-it"`
- **Architecture**: VLM layers, expert layers, attention modes
- **Training**: Optimizer settings, learning rates, freezing options
- **Data Processing**: Image sizes, tokenizer settings, normalization
- **Robot Specific**: Aloha adaptations, state/action dimensions

### **2. `modeling_gemma3nvla.py`**
**Purpose**: Main policy class that wraps the VLA model for LeRobot integration

**Key Components**:
- **`Gemma3nVLAPolicy`**: Main wrapper class (inherits from `PreTrainedPolicy`)
- **`VLAFlowMatching`**: Core neural network with flow matching for action generation
- **Data Processing**: Image preprocessing, language tokenization, state preparation
- **Action Sampling**: Diffusion-based action prediction with denoising
- **Training Loop**: Loss computation and optimization

### **3. `gemma3n_with_expert.py`**
**Purpose**: Core model architecture combining Gemma3n VLM with action expert

**Key Components**:
- **`Gemma3nWithExpertModel`**: Main architecture class
- **VLM Integration**: Loads and adapts Gemma3n for robotics
- **Action Expert**: Smaller transformer for action prediction
- **Cross-Attention**: Expert attends to VLM's representations
- **Layer Management**: Configurable VLM and expert layer counts

---

## Configuration Parameters

### **Model Architecture**

#### **VLM Configuration**
```python
# In configuration_gemma3nvla.py
vlm_model_name: str = "google/gemma-3n-e2b-it"  # Base model
num_vlm_layers: int = 30                          # Use first N layers of Gemma3n
load_vlm_weights: bool = False                    # Load pretrained weights
freeze_vision_encoder: bool = True                # Freeze vision components
```

#### **Expert Configuration**
```python
num_expert_layers: int = -1                       # -1 = same as VLM layers
expert_width_multiplier: float = 0.75            # Expert size relative to VLM
self_attn_every_n_layers: int = 2                # Self-attention frequency
attention_mode: str = "cross_attn"               # Attention pattern
```

#### **Training Configuration**
```python
train_expert_only: bool = True                    # Only train action expert
train_state_proj: bool = True                     # Train state projections
optimizer_lr: float = 5e-5                       # Lower LR for larger model
```

### **Data Processing**
```python
resize_imgs_with_padding: tuple[int, int] = (768, 768)  # Gemma3n native resolution
tokenizer_max_length: int = 48                           # Text sequence length
max_state_dim: int = 32                                  # Robot state dimension
max_action_dim: int = 32                                 # Action dimension
chunk_size: int = 50                                     # Action sequence length
```

---

## How to Modify Parameters

### **1. Change Model Size**
**File**: `configuration_gemma3nvla.py`
```python
# Use E4B instead of E2B (4B vs 2B parameters)
vlm_model_name: str = "google/gemma-3n-e4b-it"
num_vlm_layers: int = 35  # E4B has 35 layers

# Adjust learning rate for larger model
optimizer_lr: float = 2e-5
```

### **2. Adjust Expert Architecture**
**File**: `configuration_gemma3nvla.py`
```python
# Make expert larger/smaller
expert_width_multiplier: float = 1.0  # Same size as VLM
expert_width_multiplier: float = 0.5  # Half size of VLM

# Change number of expert layers
num_expert_layers: int = 15  # Explicit layer count
num_expert_layers: int = -1  # Match VLM layer count

# Modify attention pattern
self_attn_every_n_layers: int = 1  # More self-attention
attention_mode: str = "self_attn"  # Pure self-attention instead of cross
```

### **3. Training Configurations**
**File**: `configuration_gemma3nvla.py`
```python
# Fine-tune more of the model
train_expert_only: bool = False      # Train VLM too
freeze_vision_encoder: bool = False  # Train vision encoder

# Adjust learning rates
optimizer_lr: float = 1e-4           # Higher learning rate
scheduler_warmup_steps: int = 2_000  # Longer warmup
```

### **4. Image Processing**
**File**: `configuration_gemma3nvla.py`
```python
# Change image resolution
resize_imgs_with_padding: tuple[int, int] = (512, 512)  # Smaller images
resize_imgs_with_padding: tuple[int, int] = (1024, 1024)  # Larger images

# Add more cameras
empty_cameras: int = 2  # Support more camera views
```

### **5. Action Prediction**
**File**: `configuration_gemma3nvla.py`
```python
# Change action sequence length
chunk_size: int = 100        # Longer action sequences
n_action_steps: int = 25     # Steps per prediction

# Modify diffusion steps
num_steps: int = 20          # More denoising steps (slower but better)
```

---

## Advanced Modifications

### **1. Add Audio Support**
**File**: `modeling_gemma3nvla.py` - in `prepare_images()` method
```python
# Add audio processing (Gemma3n supports audio natively)
def prepare_audio(self, batch):
    # Use Gemma3n's audio encoder
    audio_features = self.vlm_with_expert.get_audio_features(batch["audio"])
    return audio_features
```

### **2. Custom Attention Patterns**
**File**: `gemma3n_with_expert.py` - modify attention mechanisms
```python
# Enable Gemma3n's sliding window attention for long sequences
def __init__(self, ...):
    # Configure sliding window attention
    self.config.use_sliding_window = True
    self.config.sliding_window = 512
```

### **3. Different Model Variants**
**File**: `configuration_gemma3nvla.py`
```python
# Use instruction-tuned vs base model
vlm_model_name: str = "google/gemma-3n-e2b"     # Base model
vlm_model_name: str = "google/gemma-3n-e2b-it"  # Instruction-tuned

# Use different sizes
vlm_model_name: str = "google/gemma-3n-e4b-it"  # Larger 4B model
```

### **4. Memory Optimization**
**File**: `gemma3n_with_expert.py`
```python
# Load with different precision
self.vlm = Gemma3nForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,      # Use float16 instead of bfloat16
    device_map="auto",
    attn_implementation="flash_attention_2",  # Use Flash Attention
)
```

### **5. Robot-Specific Adaptations**
**File**: `configuration_gemma3nvla.py`
```python
# Enable Aloha robot specific transformations
adapt_to_pi_aloha: bool = True
use_delta_joint_actions_aloha: bool = True

# Adjust for different robots
max_state_dim: int = 14   # 7-DOF arm x2
max_action_dim: int = 14  # Dual arm robot
```

---

## Usage Examples

### **Basic Training**
```bash
python -m lerobot.scripts.train \
--policy.type=gemma3nvla \
--dataset.repo_id=your_dataset \
--batch_size=32 \
--steps=200000
```

### **Custom Configuration**
```bash
python -m lerobot.scripts.train \
--policy.type=gemma3nvla \
--policy.num_vlm_layers=20 \
--policy.expert_width_multiplier=1.0 \
--policy.optimizer_lr=1e-4 \
--dataset.repo_id=your_dataset \
--batch_size=16 \
--steps=100000
```

### **Large Model (E4B)**
```bash
python -m lerobot.scripts.train \
--policy.type=gemma3nvla \
--policy.vlm_model_name="google/gemma-3n-e4b-it" \
--policy.num_vlm_layers=35 \
--policy.optimizer_lr=2e-5 \
--batch_size=16 \
--steps=200000
```

---

## Key Benefits

### **Performance Improvements**
- **4x More Parameters**: Better reasoning and understanding
- **Advanced Architecture**: AltUp, LAuReL enhance learning
- **Better Vision**: Native 768×768 resolution with MobileNet v5
- **Efficient Training**: Sparsity and KV cache sharing reduce memory

### **Flexibility**
- **Configurable Layers**: Adjust VLM and expert layer counts
- **Multiple Training Modes**: Expert-only, full fine-tuning, or hybrid
- **Attention Patterns**: Self-attention, cross-attention, or mixed
- **Robot Adaptation**: Built-in support for various robot platforms

### **Future Extensions**
- **Audio Integration**: Easy to add audio commands using Gemma3n's audio encoder
- **Larger Models**: Can easily switch to E4B or future Gemma3n variants
- **Custom Architectures**: Modular design allows easy modifications

This implementation provides a solid foundation for building state-of-the-art robotics policies with significantly enhanced capabilities compared to the original SmolVLA.