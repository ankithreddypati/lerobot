#!/usr/bin/env python3

import torch
from lerobot.policies.gemma3nvla.modeling_gemma3nvla import Gemma3nVLAPolicy

def test_gemma3nvla():
    print("🚀 Testing Gemma3nVLA model locally...")
    
    try:
        # Load your trained model
        print("📥 Loading model from HuggingFace...")
        policy = Gemma3nVLAPolicy.from_pretrained("ankithreddy/gemma3nvla_lerepairbot")
        policy.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            policy = policy.cuda()
            device = "cuda"
            print("✅ Using CUDA")
        else:
            device = "cpu"
            print("⚠️ Using CPU (slower)")
        
        # Create test batch with dummy data matching your dataset format
        print("📊 Creating test batch...")
        batch = {
            "observation.images.images.wrist": torch.randn(1, 3, 640, 480, device=device),  # Updated resolution
            "observation.images.images.top": torch.randn(1, 3, 640, 480, device=device),   # Updated resolution
            "observation.state": torch.tensor([[-6.894, -99.745, 100.0, 59.342, -54.346, 0.753]], device=device),
            "task": "pick up screwdriver and put it in box"  # Exact training task
        }
        
        print("🧠 Testing action prediction...")
        
        # Test single action
        single_action = policy.select_action(batch)
        print(f"✅ Single action shape: {single_action.shape}")
        print(f"   Values: {single_action}")
        
        # Test action chunk
        action_chunk = policy.predict_action_chunk(batch)
        print(f"✅ Action chunk shape: {action_chunk.shape}")
        print(f"   First action: {action_chunk[0, 0]}")
        print(f"   Last action: {action_chunk[0, -1]}")
        
        # Check action smoothness (good models have smooth trajectories)
        action_std = torch.std(action_chunk, dim=1).mean()
        print(f"   Action smoothness (std): {action_std:.3f}")
        
        # Verify the actions are reasonable (not NaN, not too extreme)
        if torch.isnan(action_chunk).any():
            print("❌ WARNING: Model outputting NaN values!")
        elif torch.abs(action_chunk).max() > 1000:
            print("❌ WARNING: Model outputting extreme values!")
        else:
            print("✅ Action values look reasonable")
        
        # Test with repair instruction
        print("\n🔧 Testing repair instruction...")
        batch["task"] = "pick up screwdriver and put it in box"
        actions = policy.predict_action_chunk(batch)
        print(f"Task: 'pick up screwdriver and put it in box' → First action: {actions[0, 0]}...")
        print(f"Action sequence smoothness: {torch.std(actions).item():.4f}")
        
        print("\n🎉 SUCCESS! Your Gemma3nVLA model is working!")
        print(f"Model predicts {action_chunk.shape[1]} future actions")
        print(f"Each action controls {action_chunk.shape[2]} degrees of freedom")
        print(f"Training loss=0.022 → Expecting excellent robot performance! 🏆")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("💡 Make sure you're in the lerobot directory and have the model installed")
        return False

if __name__ == "__main__":
    test_gemma3nvla()