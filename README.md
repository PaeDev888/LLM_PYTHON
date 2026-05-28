# MiniLLM Chatbot By Peadev888,ALPNAI

A Thai-English bilingual language model chatbot with GPU support and interactive training capabilities.

## Features

- **Bilingual Support**: Trained on Thai and English text corpus
- **GPU Acceleration**: Configurable GPU usage with memory management
- **Interactive Training**: Retrain the model with custom parameters
- **Command Interface**: Rich CLI with settings adjustment
- **Real-time Generation**: Streaming text generation with sampling controls

## Installation

```bash
pip install torch
```

## Usage

### Basic Execution

```bash
# Run with full GPU usage
python chat.py

# Run with specific GPU memory limit (e.g., 80%)
python chat.py --gpu 80

# Force CPU usage
python chat.py --gpu 0
```

### Interactive Commands

Once running, use these commands:

- `/set temp X` - Adjust temperature (0.1–2.0, default 0.9)
- `/set topk X` - Adjust top-k sampling (1–200, default 50)
- `/set topp X` - Adjust top-p sampling (0.1–1.0, default 0.95)
- `/set len X` - Set response length (10–500, default 120)
- `/set gpu X` - Change GPU usage percentage
- `/gpu` - View current GPU/VRAM status
- `/settings` - View all current settings
- `/retrain` - Retrain the model
- `/clear` - Clear the screen
- `/help` - Show all commands
- `/q` or `/quit` - Exit the application

## Training

The model can be trained from scratch or retrained:

```bash
# Train with default settings
python chat.py --retrain

# Train with custom epochs
python chat.py --retrain --train_epochs 20
```

Training progress shows:
- Loss and perplexity metrics
- Training speed (tokens/second)
- Estimated time remaining
- GPU memory usage (when applicable)
- Learning rate progression

## Model Architecture

- **Transformer-based** language model
- **Embedding dimension**: 128
- **Attention heads**: 4
- **Transformer layers**: 4
- **Feed-forward network**: 512 dimensions
- **Sequence length**: 64 tokens
- **Vocabulary**: Character-level tokenizer

## Technical Details

### GPU Management
The script dynamically manages GPU memory usage:
- Allows specifying GPU usage percentage (0-100%)
- Automatically falls back to CPU if GPU unavailable or set to 0%
- Displays real-time VRAM utilization during operation

### Training Features
- Mixed precision training (when using GPU)
- Gradient clipping for stability
- Cosine annealing learning rate scheduler
- AdamW optimizer
- Automatic checkpoint saving

## File Structure

```
chat.py          # Main application
model.py         # Model architecture (imported)
tokenizer.py     # Character-level tokenizer (imported)
train.py         # Training utilities (imported)
checkpoints/     # Saved models and tokenizers (created during training)
```

## Requirements

- PyTorch
- Python 3.8+

## Notes

- The model is trained on a diverse corpus covering topics from technology to general knowledge
- Supports code-switching between Thai and English
- Designed for experimentation and learning rather than production use
