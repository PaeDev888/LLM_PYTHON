"""
Tokenizer แบบง่าย — Character-level BPE-inspired
สำหรับโปรเจกต์นี้ใช้ character tokenizer ก่อน
เพื่อไม่ต้องพึ่ง library ภายนอก
"""

import re
import json
from collections import Counter
from pathlib import Path


class CharTokenizer:
    """
    Tokenizer ระดับตัวอักษร (character-level)
    เรียบง่ายที่สุด — vocab = ตัวอักษรทั้งหมดในข้อมูล
    """

    SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self.vocab_size = 0

    def build_vocab(self, texts: list[str]) -> None:
        """สร้าง vocabulary จาก list ของข้อความ"""
        # ใส่ special tokens ก่อน
        self.token2id = dict(self.SPECIAL_TOKENS)
        self.id2token = {v: k for k, v in self.token2id.items()}

        # นับตัวอักษรทั้งหมด
        chars = Counter("".join(texts))
        next_id = len(self.SPECIAL_TOKENS)

        for char, _ in sorted(chars.items()):
            if char not in self.token2id:
                self.token2id[char] = next_id
                self.id2token[next_id] = char
                next_id += 1

        self.vocab_size = len(self.token2id)
        print(f"Vocabulary: {self.vocab_size} tokens")

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """แปลงข้อความ → list ของ token IDs"""
        ids = []
        if add_special:
            ids.append(self.token2id["<bos>"])
        for ch in text:
            ids.append(self.token2id.get(ch, self.token2id["<unk>"]))
        if add_special:
            ids.append(self.token2id["<eos>"])
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """แปลง token IDs → ข้อความ"""
        special_ids = set(self.SPECIAL_TOKENS.values())
        chars = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            chars.append(self.id2token.get(i, "<unk>"))
        return "".join(chars)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token2id": self.token2id}, f, ensure_ascii=False, indent=2)
        print(f"บันทึก tokenizer → {path}")

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        print(f"โหลด tokenizer → vocab size = {self.vocab_size}")
