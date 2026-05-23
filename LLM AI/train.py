"""
Training loop สำหรับ MiniLLM
รัน: python train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import math

from model import MiniLLM
from tokenizer import CharTokenizer


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TextDataset(Dataset):
    """Dataset ที่ตัดข้อความเป็น chunks ขนาด seq_len"""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Trainer:

    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        print(f"ใช้ device: {self.device}")

    def train(self, model: MiniLLM, dataloader: DataLoader):
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )

        # Cosine learning rate schedule
        total_steps = self.cfg["epochs"] * len(dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=self.cfg["lr"] * 0.1
        )

        model.train()
        step = 0
        best_loss = float("inf")

        for epoch in range(self.cfg["epochs"]):
            total_loss = 0.0
            t0 = time.time()

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward
                _, loss = model(batch_x, batch_y)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping — ป้องกัน gradient explode
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                step += 1

                if step % self.cfg["log_every"] == 0:
                    avg = total_loss / step
                    ppl = math.exp(min(avg, 10))  # perplexity
                    lr  = scheduler.get_last_lr()[0]
                    print(f"  step {step:5d} | loss {avg:.4f} | ppl {ppl:.1f} | lr {lr:.2e}")

            epoch_loss = total_loss / len(dataloader)
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1}/{self.cfg['epochs']} | loss {epoch_loss:.4f} | {elapsed:.1f}s")

            # บันทึก checkpoint ถ้า loss ดีขึ้น
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self._save(model, optimizer, epoch, best_loss)

        print(f"\nเทรนเสร็จ! Best loss: {best_loss:.4f}")
        return model

    def _save(self, model, optimizer, epoch, loss):
        path = Path(self.cfg["save_dir"]) / "checkpoint.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
            "model_config": {
                "vocab_size": model.token_emb.embedding.num_embeddings,
                "d_model": model.d_model,
                "max_len": model.max_len,
            },
        }, path)
        print(f"  💾 บันทึก checkpoint → {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # ข้อมูลตัวอย่างสำหรับ demo (ในของจริงใช้ข้อมูลขนาดใหญ่กว่านี้)
    sample_texts = [
        """The quick brown fox jumps over the lazy dog.
        To be or not to be, that is the question.
        All that glitters is not gold.
        The only way to do great work is to love what you do.
        In the beginning was the Word, and the Word was with God.
        It was the best of times, it was the worst of times.
        Call me Ishmael. Some years ago never mind how long precisely,
        having little or no money in my purse, and nothing particular to interest me on shore,
        I thought I would sail about a little and see the watery part of the world.
        """ * 50  # ซ้ำเพื่อให้มีข้อมูลมากขึ้น
    ]

    # Config
    config = {
        "seq_len": 64,
        "batch_size": 16,
        "epochs": 5,
        "lr": 3e-4,
        "weight_decay": 0.1,
        "log_every": 20,
        "save_dir": "checkpoints",
    }

    # Model config (ขนาดเล็กสำหรับ demo)
    model_config = {
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "d_ff": 512,
        "max_len": config["seq_len"],
        "dropout": 0.1,
    }

    # Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(sample_texts)

    # Tokenize ข้อมูล
    all_ids = []
    for text in sample_texts:
        all_ids.extend(tokenizer.encode(text, add_special=False))

    print(f"ข้อมูลทั้งหมด: {len(all_ids):,} tokens")

    # Dataset + DataLoader
    dataset = TextDataset(all_ids, config["seq_len"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    print(f"จำนวน batches: {len(loader)}")

    # สร้างโมเดล
    model = MiniLLM(vocab_size=tokenizer.vocab_size, **model_config)

    # เทรน
    trainer = Trainer(config)
    trained_model = trainer.train(model, loader)

    # บันทึก tokenizer
    tokenizer.save("checkpoints/tokenizer.json")

    # ทดสอบ generate
    print("\n--- ทดสอบการสร้างข้อความ ---")
    trained_model.eval()
    device = next(trained_model.parameters()).device

    prompt = "The quick"
    prompt_ids = torch.tensor([tokenizer.encode(prompt, add_special=False)], device=device)

    output_ids = trained_model.generate(
        prompt_ids,
        max_new_tokens=80,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
    )

    generated = tokenizer.decode(output_ids[0].tolist())
    print(f"Prompt  : {prompt}")
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
