"""
chat.py — MiniLLM CMD Chat พร้อม GPU + ตั้ง % VRAM ได้
ติดตั้ง : pip install torch
รัน     : python chat.py
GPU     : python chat.py --gpu 80        (ใช้ GPU 80%)
"""

import sys, os, argparse, math, torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from model import MiniLLM
from tokenizer import CharTokenizer
from train import TextDataset

# ─────────────────────────────────────────────
# ANSI Colors
# ─────────────────────────────────────────────
class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    CYAN  = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    RED   = "\033[91m"; MAGENTA = "\033[95m"

USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def c(text, *codes):
    return ("".join(codes) + text + C.RESET) if USE_COLOR else text

# ─────────────────────────────────────────────
# GPU Setup
# ─────────────────────────────────────────────
def setup_device(gpu_percent: float = 100.0):
    """
    เลือก device และจำกัด VRAM ตาม %
    gpu_percent: 0 = ใช้ CPU, 1-100 = ใช้ GPU ตาม %
    """
    if gpu_percent <= 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
        print(c("  Device : CPU", C.YELLOW))
        return device

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory  # bytes
    total_mb   = total_vram / 1024**2

    # จำกัด VRAM ที่ PyTorch ใช้ได้
    fraction = max(0.05, min(gpu_percent / 100.0, 1.0))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)

    allowed_mb = total_mb * fraction
    print(c(f"  Device : {gpu_name}", C.GREEN, C.BOLD))
    print(c(f"  VRAM   : ใช้ได้ {allowed_mb:.0f} MB จาก {total_mb:.0f} MB ({gpu_percent:.0f}%)", C.GREEN))

    return device

def get_gpu_stats():
    """ดึงข้อมูล GPU usage ปัจจุบัน"""
    if not torch.cuda.is_available():
        return None
    used  = torch.cuda.memory_allocated(0) / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    pct   = used / total * 100
    return {"used_mb": used, "total_mb": total, "pct": pct}

def banner(gpu_percent: float):
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_label = c(f"GPU: {gpu_name} ({gpu_percent:.0f}%)", C.GREEN)
    else:
        gpu_label = c("GPU: ไม่พบ — ใช้ CPU", C.YELLOW)

    print(c("╔══════════════════════════════════════════════════╗", C.CYAN, C.BOLD))
    print(c("║         MiniLLM — ภาษาไทย + อังกฤษ             ║", C.CYAN, C.BOLD))
    print(c("╚══════════════════════════════════════════════════╝", C.CYAN, C.BOLD))
    print(f"  {gpu_label}\n")

# ─────────────────────────────────────────────
# Corpus
# ─────────────────────────────────────────────
CORPUS = ("""
สวัสดีครับ ยินดีต้อนรับสู่โลกของปัญญาประดิษฐ์
การเรียนรู้ของเครื่องจักรคือหัวใจของ AI สมัยใหม่
ภาษาไทยเป็นภาษาที่มีความซับซ้อนและสวยงาม
คอมพิวเตอร์ช่วยให้มนุษย์ทำงานได้เร็วและแม่นยำยิ่งขึ้น
ปัญญาประดิษฐ์สามารถเรียนรู้จากข้อมูลจำนวนมาก
โมเดลภาษาขนาดใหญ่ได้รับการฝึกจากข้อความจำนวนมหาศาล
ประเทศไทยมีวัฒนธรรมที่หลากหลายและน่าสนใจมาก
ความรู้คือพลังที่ยิ่งใหญ่ที่สุดในโลก
การศึกษาเป็นรากฐานสำคัญของการพัฒนาประเทศ
เทคโนโลยีช่วยพัฒนาคุณภาพชีวิตของมนุษย์
Artificial intelligence is transforming the world.
Machine learning models learn patterns from large datasets.
Natural language processing helps computers understand humans.
The quick brown fox jumps over the lazy dog.
Knowledge is power and learning is a lifelong journey.
Technology connects people across the world instantly.
Deep learning uses neural networks with many layers.
Python is a popular language for building AI systems.
""" * 25).strip()

# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────
def quick_train(ckpt_dir: str, epochs: int, device: torch.device):
    print(c("⚙  เทรนโมเดลใหม่...", C.YELLOW))

    tokenizer = CharTokenizer()
    tokenizer.build_vocab([CORPUS])
    all_ids = tokenizer.encode(CORPUS, add_special=False)

    seq_len = 64
    loader  = DataLoader(TextDataset(all_ids, seq_len), batch_size=64, shuffle=True)

    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128, n_heads=4, n_layers=4,
        d_ff=512, max_len=seq_len, dropout=0.1,
    ).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*len(loader), eta_min=3e-5)

    # mixed precision (เฉพาะ GPU)
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    best = float("inf")
    for ep in range(epochs):
        total = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                _, loss = model(bx, by)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item()

        avg = total / len(loader)
        ppl = math.exp(min(avg, 10))
        bar = "█" * (ep+1) + "░" * (epochs-ep-1)

        # แสดง VRAM ถ้าใช้ GPU
        vram_str = ""
        if device.type == "cuda":
            s = get_gpu_stats()
            vram_str = c(f"  VRAM {s['used_mb']:.0f}/{s['total_mb']:.0f}MB ({s['pct']:.1f}%)", C.DIM)

        print(f"\r  [{bar}] {ep+1}/{epochs}  loss={avg:.3f}  ppl={ppl:.1f}{vram_str}   ",
              end="", flush=True)

        if avg < best:
            best = avg
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {
                    "vocab_size": tokenizer.vocab_size,
                    "d_model": 128, "n_heads": 4,
                    "n_layers": 4,  "d_ff": 512, "max_len": seq_len,
                }
            }, f"{ckpt_dir}/checkpoint.pt")
            tokenizer.save(f"{ckpt_dir}/tokenizer.json")

    print(f"\n  {c('✓ เทรนเสร็จ', C.GREEN)}  best loss={best:.3f}\n")
    return model, tokenizer

def load_model(ckpt_dir: str, device: torch.device):
    ckpt = torch.load(f"{ckpt_dir}/checkpoint.pt", map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]
    model = MiniLLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 4),
        d_ff=cfg.get("d_ff", 512),
        max_len=cfg.get("max_len", 64),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = CharTokenizer()
    tokenizer.load(f"{ckpt_dir}/tokenizer.json")
    return model, tokenizer

# ─────────────────────────────────────────────
# Generate
# ─────────────────────────────────────────────
@torch.no_grad()
def generate_stream(model, tokenizer, prompt, max_new, temperature, top_k, top_p, device) -> str:
    ids       = tokenizer.encode(prompt, add_special=False)
    generated = torch.tensor([ids], dtype=torch.long, device=device)

    print(c("AI › ", C.GREEN, C.BOLD), end="", flush=True)
    if prompt:
        print(c(prompt, C.DIM), end="", flush=True)

    result = []
    for _ in range(max_new):
        ctx = generated[:, -model.max_len:]
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            logits, _ = model(ctx)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = float("-inf")

        if top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            sl[cum - F.softmax(sl, dim=-1) > top_p] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(1, si, sl)

        probs     = F.softmax(logits, dim=-1)
        next_tok  = torch.multinomial(probs, 1)
        generated = torch.cat([generated, next_tok], dim=1)

        ch = tokenizer.decode([next_tok.item()], skip_special=True)
        print(ch, end="", flush=True)
        result.append(ch)

    print("\n")
    return "".join(result)

# ─────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────
HELP = """
─── คำสั่ง ────────────────────────────────────────
  /set temp X    — temperature  (0.1–2.0, default 0.9)
  /set topk X    — top-k        (1–200,  default 50)
  /set topp X    — top-p        (0.1–1.0, default 0.95)
  /set len  X    — ความยาวคำตอบ (10–500, default 120)
  /set gpu  X    — เปลี่ยน % GPU (0=CPU, 1-100=GPU%)
                   เช่น /set gpu 50  หรือ /set gpu 0

  /gpu           — ดูสถานะ GPU / VRAM ปัจจุบัน
  /settings      — ดูค่าทั้งหมด
  /retrain       — เทรนโมเดลใหม่
  /clear         — ล้างหน้าจอ
  /help          — แสดงคำสั่ง
  /q  /quit      — ออก
────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────
# Chat Loop
# ─────────────────────────────────────────────
def chat_loop(model, tokenizer, device, args):
    cfg = {
        "temperature": args.temperature,
        "top_k":       args.top_k,
        "top_p":       args.top_p,
        "max_new":     args.max_tokens,
        "gpu_pct":     args.gpu,
    }
    print(c("พิมพ์ข้อความแล้วกด Enter — AI จะต่อประโยคให้\n", C.DIM))

    while True:
        try:
            # แสดง VRAM ใน prompt ถ้าใช้ GPU
            if device.type == "cuda":
                s = get_gpu_stats()
                vram_hint = c(f" [{s['pct']:.0f}%VRAM]", C.DIM)
            else:
                vram_hint = ""
            user_input = input(c("คุณ", C.CYAN, C.BOLD) + vram_hint + c(" › ", C.CYAN, C.BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{c('ลาก่อน!', C.YELLOW)}")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd   = parts[0].lower()

            if cmd in ("/quit", "/q", "/exit"):
                print(c("ลาก่อน!", C.YELLOW)); break

            elif cmd == "/help":
                print(c(HELP, C.DIM))

            elif cmd == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                banner(cfg["gpu_pct"])

            elif cmd == "/gpu":
                if not torch.cuda.is_available():
                    print(c("  ไม่พบ GPU — กำลังใช้ CPU\n", C.YELLOW))
                else:
                    s = get_gpu_stats()
                    name = torch.cuda.get_device_name(0)
                    print(c(f"  GPU    : {name}", C.GREEN))
                    print(c(f"  VRAM   : {s['used_mb']:.1f} / {s['total_mb']:.1f} MB ({s['pct']:.1f}%)", C.GREEN))
                    print(c(f"  จำกัด  : {cfg['gpu_pct']:.0f}%\n", C.DIM))

            elif cmd == "/settings":
                print(c(f"  temperature : {cfg['temperature']}", C.DIM))
                print(c(f"  top_k       : {cfg['top_k']}", C.DIM))
                print(c(f"  top_p       : {cfg['top_p']}", C.DIM))
                print(c(f"  max_tokens  : {cfg['max_new']}", C.DIM))
                print(c(f"  gpu         : {cfg['gpu_pct']}%  (device={device.type})\n", C.DIM))

            elif cmd == "/set" and len(parts) >= 3:
                key, val = parts[1].lower(), parts[2]
                try:
                    if key == "temp":
                        cfg["temperature"] = float(val)
                        print(c(f"  ✓ temperature = {cfg['temperature']}\n", C.GREEN))
                    elif key == "topk":
                        cfg["top_k"] = int(val)
                        print(c(f"  ✓ top_k = {cfg['top_k']}\n", C.GREEN))
                    elif key == "topp":
                        cfg["top_p"] = float(val)
                        print(c(f"  ✓ top_p = {cfg['top_p']}\n", C.GREEN))
                    elif key == "len":
                        cfg["max_new"] = int(val)
                        print(c(f"  ✓ max_tokens = {cfg['max_new']}\n", C.GREEN))
                    elif key == "gpu":
                        pct = float(val)
                        cfg["gpu_pct"] = pct
                        device = setup_device(pct)
                        # ย้ายโมเดลไปยัง device ใหม่
                        model = model.to(device)
                        print(c(f"  ✓ GPU = {pct}%  device={device.type}\n", C.GREEN))
                    else:
                        print(c(f"  ✗ ไม่รู้จัก: {key}\n", C.RED))
                except ValueError:
                    print(c(f"  ✗ ค่าไม่ถูกต้อง: {val}\n", C.RED))

            elif cmd == "/retrain":
                model, tokenizer = quick_train(args.checkpoint, args.train_epochs, device)

            else:
                print(c(f"  ✗ ไม่รู้จัก '{cmd}' — ลอง /help\n", C.RED))

            continue

        generate_stream(
            model, tokenizer, user_input,
            cfg["max_new"], cfg["temperature"],
            cfg["top_k"], cfg["top_p"], device,
        )

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MiniLLM — GPU + Thai support")
    parser.add_argument("--checkpoint",   default="checkpoints")
    parser.add_argument("--temperature",  type=float, default=0.9)
    parser.add_argument("--top_k",        type=int,   default=50)
    parser.add_argument("--top_p",        type=float, default=0.95)
    parser.add_argument("--max_tokens",   type=int,   default=120)
    parser.add_argument("--retrain",      action="store_true")
    parser.add_argument("--train_epochs", type=int,   default=12)
    parser.add_argument(
        "--gpu", type=float, default=100.0,
        help="VRAM ที่จะใช้ 0=CPU, 1-100=GPU เช่น --gpu 80"
    )
    args = parser.parse_args()

    if os.name == "nt":
        os.system("color")

    banner(args.gpu)

    # ── เลือก device ──
    print(c("─── ตรวจสอบ Hardware ───", C.DIM))
    device = setup_device(args.gpu)
    print()

    ckpt  = Path(args.checkpoint) / "checkpoint.pt"
    tok_f = Path(args.checkpoint) / "tokenizer.json"

    if args.retrain or not (ckpt.exists() and tok_f.exists()):
        model, tokenizer = quick_train(args.checkpoint, args.train_epochs, device)
    else:
        print(c(f"โหลดโมเดลจาก {args.checkpoint}/ ...", C.DIM))
        model, tokenizer = load_model(args.checkpoint, device)
        params = sum(p.numel() for p in model.parameters())
        print(c(f"✓ พร้อมใช้งาน | {params:,} params | vocab {tokenizer.vocab_size} | device={device.type}\n", C.GREEN))

    chat_loop(model, tokenizer, device, args)

if __name__ == "__main__":
    main()