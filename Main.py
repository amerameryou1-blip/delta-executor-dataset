"""
DELTA EXECUTOR DATASET GENERATOR — 100 KEYS + AUTO HF UPLOAD
- Keys from Cell 2 only
- Globals from your GitHub URL
- Uploads every 500 samples + FORCED upload at 11.5 hours
- Strong obfuscated filter
- Model rotation + RPM=15 for gemma
"""

import os
import re
import json
import time
import hashlib
import threading
import signal
import sys
import random
from pathlib import Path
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd
from tqdm import tqdm

from google import genai
from datasets import Dataset, load_dataset

# ====================== CONFIG ======================
TARGET_SAMPLES   = 50_000
UPLOAD_EVERY     = 500
KAGGLE_LIMIT_H   = 12
SAFETY_BUFFER_M  = 30          # safety margin before 12h

MODEL_VOLUME = "gemma-4-31b-it"
MODEL_FAST   = "gemini-3.1-flash"
MODEL_THINK  = "gemini-2.5-pro"

RPM_GEMMA = 15
RPM_OTHER = 8

# ====================== KEY LOADING FROM CELL 2 ======================
def load_all_keys():
    keys = []
    i = 1
    while True:
        key = os.environ.get(f"GEMINI_KEY_{i}")
        if not key:
            break
        keys.append(key.strip())
        i += 1
    print(f"✅ Loaded {len(keys)} Gemini API keys from Cell 2")
    if len(keys) == 0:
        raise ValueError("No GEMINI_KEY_* found. Check Cell 2.")
    return keys

API_KEYS = load_all_keys()
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("⚠️ HF_TOKEN not set — automatic uploads will fail!")

HF_DATASET  = "amer224/Luau"
GLOBALS_URL = "https://raw.githubusercontent.com/amerameryou1-blip/delta-executor-dataset/refs/heads/main/delta_globals.json"

# ====================== OUTPUT DIRS ======================
OUT = Path("./delta_out")
OUT.mkdir(exist_ok=True)
GOOD_JSONL  = OUT / "good_samples.jsonl"
STATE_FILE  = OUT / "state.json"
REJECTS_LOG = OUT / "rejects.log"

# ====================== RATE LIMITER & KEY POOL ======================
class KeyLimiter:
    def __init__(self, api_key: str, idx: int, rpm: int):
        self.key = api_key
        self.idx = idx
        self.rpm = rpm
        self._lk = threading.Lock()
        self._win = deque()
        self._today = 0
        self._day_t = time.time()

    def _prune(self):
        now = time.time()
        if now - self._day_t >= 86400:
            self._today = 0
            self._day_t = now
        while self._win and now - self._win[0] >= 60:
            self._win.popleft()

    def exhausted(self) -> bool:
        with self._lk:
            self._prune()
            return self._today >= 1490

    def wait_and_acquire(self) -> bool:
        while True:
            with self._lk:
                self._prune()
                if self._today >= 1490:
                    return False
                if len(self._win) < self.rpm:
                    self._win.append(time.time())
                    self._today += 1
                    return True
                wait = 60.0 - (time.time() - self._win[0]) + 0.1
            time.sleep(max(0.05, wait))

    def status(self) -> str:
        with self._lk:
            self._prune()
            return f"Key{self.idx+1}: {self._today}/1490  {len(self._win)}/{self.rpm}/min"

class KeyPool:
    def __init__(self, keys: list, rpm: int):
        self.keys = [KeyLimiter(k, i, rpm) for i, k in enumerate(keys)]
        self._rr = 0
        self._lk = threading.Lock()

    def get_key(self) -> KeyLimiter | None:
        for _ in range(len(self.keys) * 2):
            with self._lk:
                idx = self._rr % len(self.keys)
                self._rr += 1
            k = self.keys[idx]
            if not k.exhausted():
                return k
        return None

    def print_status(self):
        print("\n📊 Key Status (first 15):")
        for k in self.keys[:15]:
            print(f"  {k.status()}")
        print(f"  ... +{len(self.keys)-15} more keys\n")

# ====================== CALL GEMINI ======================
def call_gemini(limiter: KeyLimiter, system_prompt: str, user_msg: str,
                expect_json=True, retries=4, strip_code=False, model=None):
    if model is None:
        model = MODEL_VOLUME

    full_prompt = f"{system_prompt}\n\n{user_msg}"

    for attempt in range(retries):
        if not limiter.wait_and_acquire():
            print(f"   [Key {limiter.idx+1}] Exhausted — skipping")
            return None

        print(f"   [Key {limiter.idx+1}] → {model} | attempt {attempt+1}")
        try:
            client = genai.Client(api_key=limiter.key)
            response = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=3000,
                )
            )
            text = response.text.strip()
            print(f"   [Key {limiter.idx+1}] → SUCCESS ({len(text)} chars)")

            if expect_json or strip_code:
                text = re.sub(r'^```(?:json|lua|luau)?\s*', '', text, flags=re.M)
                text = re.sub(r'\s*```$', '', text, flags=re.M)
                text = text.strip()

            time.sleep(0.4)
            return text

        except Exception as e:
            err = str(e).lower()
            print(f"   [Key {limiter.idx+1}] → ERROR: {str(e)[:150]}")
            if "408" in err or "timeout" in err:
                time.sleep(10 * (attempt + 1))
            elif "429" in err or "quota" in err:
                time.sleep(60 * (attempt + 1))
            else:
                time.sleep(3)
    print(f"   [Key {limiter.idx+1}] → Failed after {retries} attempts")
    return None

# ====================== LOAD GLOBALS ======================
def load_globals() -> tuple[list[dict], set[str]]:
    print("📥 Fetching delta_globals.json from your GitHub URL...")
    r = requests.get(GLOBALS_URL, timeout=15)
    r.raise_for_status()
    data = r.json()

    funcs = data.get("functions", [])
    whitelist = set()

    for f in funcs:
        name = f.get("name")
        if name:
            whitelist.add(name)
        for alias in f.get("aliases", []):
            if alias:
                whitelist.add(alias)

    print(f"  ✅ Loaded {len(funcs)} Delta globals → whitelist has {len(whitelist)} entries")
    return funcs, whitelist

# ====================== OBFUSCATED FILTER (strengthened) ======================
_OBFUSC = [
    re.compile(r'^[A-Za-z0-9+/]{300,}={0,2}$', re.M),
    re.compile(r'\\[0-9]{2,3}\\[0-9]{2,3}\\[0-9]{2,3}'),
    re.compile(r'local\s+[A-Z_]{25,}\s*='),
    re.compile(r'--\s*obfuscated|base64|encrypted', re.I),
]

def _is_obfuscated(code: str) -> bool:
    if any(p.search(code) for p in _OBFUSC):
        return True
    if len(re.findall(r'\b[a-zA-Z]\b', code)) > 150:
        return True
    return False

def scrape_scriptblox(pages=200) -> list[dict]:
    results = []
    print("📥 Scraping ScriptBlox...")
    for pg in range(1, pages + 1):
        try:
            r = requests.get(
                "https://scriptblox.com/api/script/fetch",
                params={"page": pg, "max": 20},
                timeout=10,
            )
            scripts = r.json().get("result", {}).get("scripts", [])
            if not scripts:
                break
            for s in scripts:
                code = s.get("script", "")
                if not code or len(code) < 100 or _is_obfuscated(code):
                    continue
                results.append({
                    "source":    "scriptblox",
                    "name":      s.get("title", "Script"),
                    "game":      s.get("game", {}).get("name", "Unknown"),
                    "desc":      "",
                    "code_hint": code[:3000],
                })
            time.sleep(0.3)
        except Exception as e:
            print(f"  ⚠️ SB page {pg}: {e}")
            time.sleep(2)
    print(f"  ✅ {len(results)} clean ScriptBlox scripts")
    return results

try:
    from luaparser import ast as _lua_ast
    _LUAPARSER = True
except ImportError:
    _LUAPARSER = False
    print("⚠️ luaparser missing — run: pip install luaparser")

def ast_check(code: str) -> tuple[bool, str]:
    if not _LUAPARSER:
        return True, ""
    try:
        _lua_ast.parse(code)
        return True, ""
    except Exception as e:
        return False, str(e)[:300]

_LUA_KEYWORDS = {"if","then","else","elseif","end","while","do","for","in","repeat","until","function","local","return","break","continue","and","or","not","true","false","nil"}

def whitelist_check(code: str, wl: set) -> tuple[bool, list[str]]:
    top = re.findall(r'(?<![.:a-zA-Z0-9_])([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
    ns = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
    all_calls = set(top) | set(ns)
    unknown = [f for f in all_calls if f not in wl and f not in _LUA_KEYWORDS and len(f) > 2]
    return len(unknown) == 0, unknown

# System Prompts
_GEN_SYS = """You are an expert Roblox executor scripter specialising in Delta executor, UNC globals, and advanced Luau.

Generate ONE training sample as raw JSON — no markdown fences, no extra text:
{
  "prompt": "<1-2 sentence natural question a real scripter would ask>",
  "chain_of_thought": "<150-400 words: which APIs to use and exactly WHY, edge cases, how it works step-by-step>",
  "code": "<complete, runnable Luau executor script — no placeholder comments>",
  "explanation": "<3-6 sentences plain English of what the code does>"
}

Hard rules:
1. Use ONLY real Delta/UNC globals
2. Every function call must exist in a real executor
3. chain_of_thought must explain why each executor global was chosen
4. code must be complete"""

_REWRITE_SYS = """You are a Luau executor syntax expert.
Return ONLY the fixed Luau code. No JSON, no markdown fences, no explanation whatsoever."""

_REVIEW_SYS = """You are a strict quality reviewer for a Roblox executor AI training dataset.
Examine the sample and reply ONLY with JSON — nothing else:
{"pass": true|false, "score": 1-10, "issue": "brief note or empty string"}
Scoring guide:
  8-10 → correct globals, complete code, deep reasoning, explanation matches code
  5-7  → minor issues
  1-4  → wrong/fake globals, broken code, reasoning doesn't match code"""

_QC_SYS = """You are a final quality-control agent for a Roblox executor ML dataset.
Score each sample strictly on: prompt clarity, reasoning depth, code correctness, explanation accuracy.
Reply ONLY with a JSON array — no other text:
[{"idx": 0, "score": 8}, {"idx": 1, "score": 5}, ...]
Be strict: 9-10 = truly excellent; 7-8 = solid; below 7 = mediocre."""

def build_gen_prompt(entry: dict, whitelist: set) -> str:
    wl_sample = ", ".join(sorted(whitelist)[:50]) + " ..."
    if entry.get("source") == "scriptblox":
        return f"Here is a real executor script from ScriptBlox:\nTitle: {entry['name']} | Game: {entry['game']}\n\n```lua\n{entry['code_hint']}\n```\n\nGenerate a training sample where the prompt is the natural question that would lead to writing this kind of script. Improve the code. Valid executor globals include: {wl_sample}"
    else:
        variant_themes = ["ESP/wallhack overlay using Drawing API", "remote event spy", "anti-AFK", "speed hack", "auto-farm", "GUI utility", "stat reader", "hook-based anti-cheat bypass", "fly script", "aimbot", "noclip", "executor environment debugger"]
        theme = variant_themes[entry.get("variant", 0) % len(variant_themes)]
        return f"Generate a training sample showing how to use the executor global: `{entry['name']}`\nDescription: {entry.get('desc', 'Executor global function')}\nTheme: **{theme}**\nMake it a realistic, complete use case. Valid executor globals include: {wl_sample}"

def _reject(sid, reason, detail=""):
    with open(REJECTS_LOG, "a") as f:
        f.write(json.dumps({"id": sid, "reason": reason, "detail": str(detail)[:200]}) + "\n")

def process_entry(entry: dict, pool: KeyPool, whitelist: set, done_ids: set) -> dict | None:
    sid = hashlib.md5(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
    if sid in done_ids:
        return None

    w = pool.get_key()
    if w is None:
        return None

    raw = call_gemini(w, _GEN_SYS, build_gen_prompt(entry, whitelist), expect_json=True)
    if not raw:
        return None

    try:
        sample = json.loads(raw)
        assert all(k in sample for k in ("prompt","chain_of_thought","code","explanation"))
    except Exception:
        _reject(sid, "json_parse_fail")
        return None

    code = sample.get("code", "").strip()
    if len(code) < 30:
        _reject(sid, "code_too_short")
        return None

    ast_ok, ast_err = ast_check(code)
    if not ast_ok:
        w2 = pool.get_key()
        if not w2:
            return None
        fixed = call_gemini(w2, _REWRITE_SYS, f"Luau syntax error: {ast_err}\n\nCode to fix:\n{code}", expect_json=False, strip_code=True)
        if not fixed:
            _reject(sid, "ast_fail")
            return None
        code = fixed.strip()
        sample["code"] = code

    wl_ok, unknown = whitelist_check(code, whitelist)
    if not wl_ok:
        w3 = pool.get_key()
        if not w3:
            return None
        wl_hint = ", ".join(sorted(whitelist)[:60])
        fixed = call_gemini(w3, _REWRITE_SYS, f"These function names do NOT exist in Delta executor: {unknown}\nReplace them with correct equivalents from this list: {wl_hint}\n\nOriginal code:\n{code}", expect_json=False, strip_code=True)
        if not fixed:
            _reject(sid, "whitelist_fail")
            return None
        code = fixed.strip()
        sample["code"] = code

    w4 = pool.get_key()
    if w4 is None:
        return None

    review_msg = f"Prompt: {sample['prompt']}\n\nCode:\n{code}\n\nReasoning (first 400 chars): {sample['chain_of_thought'][:400]}"
    rev_raw = call_gemini(w4, _REVIEW_SYS, review_msg, expect_json=True, model=MODEL_THINK)
    review_score = 6

    if rev_raw:
        try:
            rev = json.loads(rev_raw)
            review_score = rev.get("score", 6)
            if not rev.get("pass", True) or review_score < 5:
                _reject(sid, "review_fail")
                return None
        except Exception:
            pass

    sample.update({"source": entry.get("source", "delta"), "source_id": sid, "review_score": review_score})
    return sample

def qc_and_upload(samples: list[dict], pool: KeyPool, is_final=False) -> list[dict]:
    if not samples:
        return []
    print(f"\n🔍 QC scoring {len(samples)} samples...")
    passed = [s for s in samples if s.get("review_score", 6) >= 7]
    print(f"  ✅ QC: {len(passed)}/{len(samples)} passed")
    if passed:
        _hf_upload(passed, is_final=is_final)
    return passed

def _hf_upload(samples: list[dict], is_final=False):
    label = "FINAL" if is_final else "batch"
    print(f"\n📤 Uploading {len(samples)} samples to {HF_DATASET} [{label}]...")
    try:
        try:
            existing_ds = load_dataset(HF_DATASET, token=HF_TOKEN, split="train")
            existing_df = existing_ds.to_pandas()
            new_df = pd.DataFrame(samples)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["source_id"], keep="last")
        except Exception:
            combined = pd.DataFrame(samples)

        COLS = ["prompt","chain_of_thought","code","explanation","source","source_id","review_score"]
        combined = combined[[c for c in COLS if c in combined.columns]]
        ds = Dataset.from_pandas(combined, preserve_index=False)
        ds.push_to_hub(HF_DATASET, token=HF_TOKEN, commit_message=f"{label} upload — {len(combined)} rows")
        print(f"  ✅ Successfully uploaded to Hugging Face! Now has {len(combined)} rows")
    except Exception as e:
        print(f"  ❌ HF upload error: {e}")

class State:
    def __init__(self):
        self.done_ids = set()
        self.total_good = 0
        self.upload_buf = []
        self._lk = threading.Lock()

    def load(self):
        if STATE_FILE.exists():
            try:
                d = json.loads(STATE_FILE.read_text())
                self.done_ids = set(d.get("done_ids", []))
                self.total_good = d.get("total_good", 0)
                print(f"▶ Resumed from checkpoint: {self.total_good} samples")
            except Exception:
                pass

    def save(self):
        with self._lk:
            STATE_FILE.write_text(json.dumps({
                "done_ids": list(self.done_ids),
                "total_good": self.total_good,
                "ts": datetime.now().isoformat(),
            }))

    def record(self, sample):
        with self._lk:
            self.done_ids.add(sample["source_id"])
            self.total_good += 1
            self.upload_buf.append(sample)
            with open(GOOD_JSONL, "a") as fh:
                fh.write(json.dumps(sample) + "\n")

    def flush(self):
        with self._lk:
            buf = self.upload_buf[:]
            self.upload_buf.clear()
            open(GOOD_JSONL, "w").close()
        return buf

# ====================== MAIN WITH FORCED 11.5 HOUR UPLOAD ======================
def main():
    t0 = time.time()
    deadline = t0 + KAGGLE_LIMIT_H * 3600 - SAFETY_BUFFER_M * 60
    forced_upload_time = t0 + (11.5 * 3600)   # 11.5 hours

    print("╔══════════════════════════════════════════╗")
    print("║  Delta Executor Dataset Generator        ║")
    print(f"║  Target: {TARGET_SAMPLES:,} samples with 100 keys  ║")
    print("╚══════════════════════════════════════════╝\n")

    pool = KeyPool(API_KEYS, RPM_GEMMA)
    pool.print_status()

    globals_list, whitelist = load_globals()

    queue = []
    for g in globals_list:
        for v in range(12):
            e = {**g, "source": "delta_global", "variant": v}
            queue.append(e)

    print(f"📋 Queue : {len(queue):,} items remaining")

    state = State()
    state.load()

    bar = tqdm(total=TARGET_SAMPLES, initial=state.total_good, desc="Good samples", unit="smp", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=30) as ex:
        pending = {}
        qi = 0

        while state.total_good < TARGET_SAMPLES:
            if time.time() > deadline:
                print("\n⏰ Reached safety deadline — stopping.")
                break

            # Forced upload at 11.5 hours
            if time.time() >= forced_upload_time and len(state.upload_buf) > 0:
                print("\n⏰ 11.5 hour mark reached — forcing final upload before Kaggle timeout")
                buf = state.flush()
                qc_and_upload(buf, pool, is_final=True)
                forced_upload_time = float('inf')  # only once

            while len(pending) < 30 and qi < len(queue):
                fut = ex.submit(process_entry, queue[qi], pool, whitelist, state.done_ids)
                pending[fut] = queue[qi]
                qi += 1

            completed = [f for f in pending if f.done()]
            for f in completed:
                pending.pop(f)
                try:
                    sample = f.result(timeout=0)
                except Exception as exc:
                    print(f"\n  ⚠️  Worker threw: {exc}")
                    continue

                if sample:
                    state.record(sample)
                    bar.update(1)
                    state.save()
                    if len(state.upload_buf) >= UPLOAD_EVERY:
                        buf = state.flush()
                        qc_and_upload(buf, pool)
                        bar.set_postfix({"HF_total": state.total_good})

            if not completed:
                time.sleep(0.15)

    bar.close()

    print(f"\n🏁 Generation done: {state.total_good:,} good samples total")
    remaining = state.flush()
    if remaining:
        print(f"📤 Final upload of {len(remaining)} remaining samples...")
        qc_and_upload(remaining, pool, is_final=True)

    state.save()
    pool.print_status()
    elapsed = (time.time() - t0) / 3600
    print(f"\n✅ All done in {elapsed:.1f}h.")
    print(f"   Dataset → https://huggingface.co/datasets/{HF_DATASET}")

if __name__ == "__main__":
    main()
