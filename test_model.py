#!/usr/bin/env python3
"""
Test script for the trained LFM2-VL GUI model

This script loads the trained model and tests it on sample images.
"""

import torch
import argparse
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from pathlib import Path


def load_trained_model(model_path: str):
    """Load the trained model and processor.
    
    Args:
        model_path: Path to the trained model directory
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"📚 Loading trained model from: {model_path}")
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    print("✅ Model loaded successfully!")
    return model, processor


def test_model_on_image(model, processor, image_path: str, question: str):
    """Test the model on a single image with a question.
    
    Args:
        model: The loaded model
        processor: The model processor
        image_path: Path to the test image
        question: Question to ask about the image
        
    Returns:
        Model's response
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Prepare the conversation
    conversation = [
        {
            "role": "system", 
            "content": [
                {
                    "type": "text", 
                    "text": "You are a GUI automation assistant. Analyze the screenshot and provide guidance on GUI interactions."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]
    
    # Process with the model
    inputs = processor.apply_chat_template(
        conversation, 
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode the response
    response = processor.tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response


def interactive_test(model, processor):
    """Interactive testing mode where user provides images and questions.
    
    Args:
        model: The loaded model
        processor: The model processor
    """
    print("\n🔍 Interactive Testing Mode")
    print("Enter image paths and questions to test the model.")
    print("Type 'quit' to exit.")
    
    while True:
        print("\n" + "="*50)
        
        # Get image path
        image_path = input("📁 Enter image path (or 'quit' to exit): ").strip()
        if image_path.lower() == 'quit':
            break
        
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            continue
        
        # Get question
        question = input("❓ Enter your question about the GUI: ").strip()
        if not question:
            print("⚠️  Question cannot be empty!")
            continue
        
        try:
            print("\n🤔 Thinking...")
            response = test_model_on_image(model, processor, image_path, question)
            print(f"\n🤖 Model Response:\n{response}")
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")


def batch_test(model, processor, test_cases):
    """Run batch tests on predefined test cases.
    
    Args:
        model: The loaded model
        processor: The model processor
        test_cases: List of (image_path, question) tuples
    """
    print(f"\n🧪 Running {len(test_cases)} batch tests...")
    
    for i, (image_path, question) in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/{len(test_cases)}")
        print(f"   🖼️  Image: {image_path}")
        print(f"   ❓ Question: {question}")
        
        if not Path(image_path).exists():
            print(f"   ❌ Image not found, skipping...")
            continue
        
        try:
            response = test_model_on_image(model, processor, image_path, question)
            print(f"   🤖 Response: {response}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test trained LFM2-VL GUI model")
    parser.add_argument("--model-path", default="./lfm2-vl-gui", help="Path to trained model")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--question", help="Question to ask about the image")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--batch", help="Path to batch test file (JSON format)")
    
    args = parser.parse_args()
    
    try:
        # Load the trained model
        model, processor = load_trained_model(args.model_path)
        
        if args.interactive:
            # Interactive mode
            interactive_test(model, processor)
            
        elif args.image and args.question:
            # Single test
            print(f"\n🧪 Testing single image...")
            print(f"   🖼️  Image: {args.image}")
            print(f"   ❓ Question: {args.question}")
            
            response = test_model_on_image(model, processor, args.image, args.question)
            print(f"\n🤖 Model Response:\n{response}")
            
        elif args.batch:
            # Batch testing
            import json
            with open(args.batch, 'r') as f:
                test_cases = json.load(f)
            batch_test(model, processor, test_cases)
            
        else:
            # Default: interactive mode
            print("ℹ️  No specific test mode selected. Starting interactive mode...")
            interactive_test(model, processor)
            
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()