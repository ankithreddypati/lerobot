#  LeRobot with gemma3nvla and technical architecture



### Installation

```bash
# Basic installation

git clone https://github.com/ankithreddypati/lerobot.git
cd lerobot && git checkout adding_gemma3nvla
pip install -e .
pip install -e ".[gemma3nvla]"
pip install -e ".[async]"

```

## 🏗️ Gemma3nVLA Architecture

### Overview

**Gemma3nVLA** is a **Vision-Language-Action (VLA)** model that extends Google's **Gemma3n-E2B** vision-language model to perform robotic control tasks.

### Architecture Design

```
┌──────────────────────────────┐
│                 actions      │
│                    ▲         │
│ ┌─────────┐      ┌─|────┐    │
│ |         │────► │      │    │
│ | Gemma3n │ kv   │      │    │
│ |   VLM   │────► │Action│    │
│ |  E2B    │cache │Expert│    │
│ │         │────► |      │    │
│ │         │      │      │    │
│ └▲──▲───▲─┘      └───▲──┘    │
│  │  |   |            │       │
│  |  |   |          noise     │
│  │  │ state                  │
│  │ language tokens           │
│  image(s)                    │
└──────────────────────────────┘
```

### Core Components

#### 1. **Gemma3nWithExpertModel**
- **Base Model**: `google/gemma-3n-e2b-it` (~2B parameters)
- **Expert Architecture**: Smaller action expert with configurable width multiplier (default: 0.75x)
- **Layer Configuration**:
  - VLM layers: 30 (configurable, can be reduced for efficiency)
  - Expert layers: Same as VLM or configurable
  - Self-attention every N layers (default: every 2 layers)
- **Precision**: BFloat16 throughout for memory efficiency
- **Attention Modes**: Supports both self-attention and cross-attention

#### 2. **VLAFlowMatching**
- **Flow Matching**: Implements flow matching for action generation (faster than diffusion)
- **Prefix-Suffix Architecture**:
  - **Prefix**: Images + Language + State embeddings
  - **Suffix**: Noisy actions for denoising
- **Time Encoding**: Sinusoidal positional encoding for timesteps
- **Denoising Process**: Multi-step denoising with Euler integration

#### 3. **Gemma3nVLAPolicy** (Main Interface)
- **Input Processing**: Handles images, language, and robot state
- **Normalization**: Mean-std normalization for state/actions, identity for images
- **Action Chunking**: Generates sequences of actions (default: 50 steps)
- **ALOHA Adaptations**: Special handling for ALOHA robot coordinate systems

### Key Design Choices

#### **1. Flow Matching vs Diffusion**
- **Choice**: Flow matching for faster convergence
- **Steps**: 10 denoising steps (vs 100+ for diffusion)
- **Loss**: MSE on velocity field
- **Benefits**: Faster training and inference

#### **2. Expert Network Design**
- **Width**: 75% of VLM hidden size (efficiency)
- **Layers**: Same as VLM or configurable
- **Precision**: BFloat16 throughout
- **Purpose**: Action generation while VLM handles vision/language

#### **3. Multi-modal Integration**
- **Images**: Gemma3n native 768x768 processing
- **Language**: Gemma3n tokenizer with special tokens (BOI/EOI)
- **State**: Linear projection to VLM hidden size
- **Actions**: Expert network output

#### **4. Memory Efficiency**
- **Precision**: BFloat16
- **Attention**: SDPA implementation
- **Caching**: KV cache for inference
- **Layer reduction**: Configurable VLM layers

### Parameter Counts

| Configuration | Total Parameters | Trainable Parameters | Use Case |
|---------------|------------------|---------------------|----------|
| **Full Model** | ~2.06B | ~62M | Inference, full capability |
| **Efficient** | ~285M | ~15M | Training, development |
| **Minimal** | ~100M | ~10M | Fast prototyping |

### Training Strategy

#### **Efficient Fine-tuning**
```python
# Default training configuration
freeze_vision_encoder: bool = True      # Freeze vision tower
train_expert_only: bool = True          # Only train expert network
train_state_proj: bool = True           # Train state projection
```

#### **Training Parameters**
```python
# Optimization
optimizer_lr: float = 5e-5              # Lower LR for larger model
optimizer_betas: tuple[float, float] = (0.9, 0.95)
optimizer_weight_decay: float = 1e-10
optimizer_grad_clip_norm: float = 10

# Scheduler
scheduler_warmup_steps: int = 1_000
scheduler_decay_steps: int = 30_000
scheduler_decay_lr: float = 1.25e-6
```

#### **Training Example**
```bash
python -m lerobot.scripts.train \
  --policy.type=gemma3nvla \
  --policy.num_vlm_layers=4 \
  --policy.repo_id=${HF_USER}/model_name \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --dataset.video_backend=pyav \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=10000 \
  --output_dir=outputs/train/gemma3nvla_v0 \
  --job_name=gemma3nvla_test_v0 \
  --policy.device=cuda \
  --wandb.enable=true
```

#### **🔗 Colab Fine-tuning Tutorial**
For an interactive fine-tuning experience, check out [Google Colab tutorial](https://colab.research.google.com/drive/1-gZRbKM1wiLcafGUIfzYxKi7QYprIuE2) 

### Optimization Techniques



##  Async Inference

### Real-time Robot Control

Gemma3nVLA supports async inference for real-time robot control. You'll need to run two terminals:

cd 
pip install -e ".[async]"

#### **Terminal 1: Policy Server**
```bash
pip install -e ".[async]"
python3 -m lerobot.scripts.server.policy_server --host=localhost --port=8080
```

#### **Terminal 2: Robot Client**
```bash
python3 -m lerobot.scripts.server.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{ images.wrist: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30}, images.top: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30}}' \
    --task="pick up screwdriver and put it in box" \
    --policy_type=gemma3nvla \
    --pretrained_name_or_path=ankithreddy/gemma3nvla_lerepairbot_v0 \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```



This project was developed as part of **The Gemma 3n Impact Challenge**, where it achieved **~80% success rate** on pick and place tasks using the Gemma3nVLA architecture.

###  Dataset
- **Dataset**: [SO101 Pick & Place Tools Dataset](https://huggingface.co/datasets/ankithreddy/so101_pickplace_tools)
- **Task**: Pick and place manipulation 

###  Trained Model
- **Model**: [Gemma3nVLA LeRepairBot v0](https://huggingface.co/ankithreddy/gemma3nvla_lerepairbot_v0)
- **Base Model**: Google Gemma-3n-E2B

###  Deployment Architecture
- **Primary Model**: Running on AMD pc with rocm 
- **RAG Server**: Separate server with retrieval-augmented generation capabilities
- **Integration**: RAG LLM server triggers this model for task execution
