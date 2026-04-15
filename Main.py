"""
╔═══════════════════════════════════════════════════════════════╗
║  DELTA EXECUTOR DATASET GENERATOR — FULL 10-AGENT PIPELINE   ║
║  Keys 1-9 : Generate → AST Check → Whitelist → Rewrite → Review  ║
║  Key  10  : Quality Control + HuggingFace Upload              ║
║  Model    : gemma-4-31b-it  (thinking=high)                   ║
║  Limits   : 15 RPM / 1490 per day per key (sliding window)    ║
║  Target   : 50,000 samples → amer224/Luau on HuggingFace      ║
╚═══════════════════════════════════════════════════════════════╝

Kaggle setup (run these cells BEFORE downloading this script):

  Cell 1:
    !pip install -q google-genai luaparser requests pandas pyarrow datasets huggingface_hub tqdm

  Cell 2 (never push to GitHub — keep local):
    import os
    os.environ["GEMINI_KEY_1"]  = "..."
    os.environ["GEMINI_KEY_2"]  = "..."
    os.environ["GEMINI_KEY_3"]  = "..."
    os.environ["GEMINI_KEY_4"]  = "..."
    os.environ["GEMINI_KEY_5"]  = "..."
    os.environ["GEMINI_KEY_6"]  = "..."
    os.environ["GEMINI_KEY_7"]  = "..."
    os.environ["GEMINI_KEY_8"]  = "..."
    os.environ["GEMINI_KEY_9"]  = "..."
    os.environ["GEMINI_KEY_10"] = "..."
    os.environ["HF_TOKEN"]      = "..."

  Cell 3:
    !wget -q https://raw.githubusercontent.com/amerameryou1-blip/delta-executor-dataset/refs/heads/main/Main.py
    !python Main.py
"""

# ╔══════════════════════════════════════════════╗
# ║  CREDENTIALS — loaded from environment       ║
# ║  Set them in a local notebook cell (Cell 2)  ║
# ╚══════════════════════════════════════════════╝
import os

def _get_key(n):
    k = os.environ.get(f"GEMINI_KEY_{n}", "")
    if not k:
        raise EnvironmentError(
            f"GEMINI_KEY_{n} is not set. "
            "Run the credentials cell before downloading this script."
        )
    return k

GEMINI_KEYS = [_get_key(i) for i in range(1, 11)]

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN is not set. Run the credentials cell first.")

HF_DATASET  = "amer224/Luau"
GLOBALS_URL = "https://raw.githubusercontent.com/amerameryou1-blip/delta-executor-dataset/refs/heads/main/delta_globals.json"

# ╔══════════════════════════════════════════════╗
# ║  TUNING KNOBS (safe to leave as-is)          ║
# ╚══════════════════════════════════════════════╝
TARGET_SAMPLES   = 50_000
UPLOAD_EVERY     = 500
KAGGLE_LIMIT_H   = 12
SAFETY_BUFFER_M  = 35
RPM_LIMIT        = 15
RPD_LIMIT        = 1490
VARIANTS_PER_GLB = 12
MODEL            = "gemma-4-31b-it"

# ════════════════════════════════════════════════════════════
import re, json, time, hashlib, threading, signal, sys, random
from pathlib import Path
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd
from tqdm import tqdm

from google        import genai
from google.genai  import types
from datasets      import Dataset, load_dataset

# ── Output dirs ─────────────────────────────────────────────
OUT         = Path("./delta_out")
OUT.mkdir(exist_ok=True)
GOOD_JSONL  = OUT / "good_samples.jsonl"
STATE_FILE  = OUT / "state.json"
REJECTS_LOG = OUT / "rejects.log"


# ════════════════════════════════════════════════════════════
# 1.  RATE LIMITER
# ════════════════════════════════════════════════════════════
class KeyLimiter:
    def __init__(self, api_key: str, idx: int):
        self.key    = api_key
        self.idx    = idx
        self._lk    = threading.Lock()
        self._win   = deque()
        self._today = 0
        self._day_t = time.time()

    def _prune(self):
        now = time.time()
        if now - self._day_t >= 86400:
            self._today  = 0
            self._day_t  = now
        while self._win and now - self._win[0] >= 60:
            self._win.popleft()

    def exhausted(self) -> bool:
        with self._lk:
            self._prune()
            return self._today >= RPD_LIMIT

    def wait_and_acquire(self) -> bool:
        while True:
            with self._lk:
                self._prune()
                if self._today >= RPD_LIMIT:
                    return False
                if len(self._win) < RPM_LIMIT:
                    self._win.append(time.time())
                    self._today += 1
                    return True
                wait = 60.0 - (time.time() - self._win[0]) + 0.1
            time.sleep(max(0.05, wait))

    def status(self) -> str:
        with self._lk:
            self._prune()
            return f"W{self.idx+1}: {self._today}/{RPD_LIMIT}/day  {len(self._win)}/{RPM_LIMIT}/min"

    def save(self) -> dict:
        with self._lk:
            return {"today": self._today, "day_t": self._day_t}

    def restore(self, d: dict):
        with self._lk:
            if time.time() - d.get("day_t", 0) < 86400:
                self._today = d.get("today", 0)
                self._day_t = d.get("day_t", time.time())


# ════════════════════════════════════════════════════════════
# 2.  KEY POOL
# ════════════════════════════════════════════════════════════
class KeyPool:
    def __init__(self):
        self.workers = [KeyLimiter(GEMINI_KEYS[i], i) for i in range(9)]
        self.qc      = KeyLimiter(GEMINI_KEYS[9], 9)
        self._rr     = 0
        self._lk     = threading.Lock()

    def get_worker(self) -> "KeyLimiter | None":
        for _ in range(9):
            with self._lk:
                idx = self._rr % 9
                self._rr += 1
            k = self.workers[idx]
            if not k.exhausted():
                return k
        if all(k.exhausted() for k in self.workers):
            return None
        time.sleep(1)
        return self.get_worker()

    def get_qc(self) -> KeyLimiter:
        return self.qc

    def print_status(self):
        print("\n📊 Key usage:")
        for k in self.workers:
            print(f"  {k.status()}")
        print(f"  QC: {self.qc._today}/{RPD_LIMIT}/day\n")

    def save(self) -> dict:
        return {"workers": [k.save() for k in self.workers], "qc": self.qc.save()}

    def restore(self, d: dict):
        for i, s in enumerate(d.get("workers", [])):
            self.workers[i].restore(s)
        self.qc.restore(d.get("qc", {}))


# ════════════════════════════════════════════════════════════
# 3.  GEMINI CALLER
# ════════════════════════════════════════════════════════════
def call_gemini(limiter: KeyLimiter, system_prompt: str, user_msg: str,
                expect_json=True, retries=3, strip_code=False) -> "str | None":
    for attempt in range(retries):
        if not limiter.wait_and_acquire():
            return None
        try:
            client = genai.Client(api_key=limiter.key)
            resp   = client.models.generate_content(
                model    = MODEL,
                contents = user_msg,
                config   = types.GenerateContentConfig(
                    system_instruction = system_prompt,
                    thinking_config    = types.ThinkingConfig(thinking_budget=8192),
                    temperature        = 0.7,
                    max_output_tokens  = 3000,
                ),
            )
            text = resp.text.strip()
            if expect_json or strip_code:
                text = re.sub(r'^```(?:json|lua|luau)?\s*', '', text, flags=re.M)
                text = re.sub(r'\s*```$',                   '', text, flags=re.M)
                text = text.strip()
            return text
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "resource" in err:
                wait = 60 * (attempt + 1)
                print(f"\n  ⏳ Key {limiter.idx+1} 429 — waiting {wait}s")
                time.sleep(wait)
            elif any(x in err for x in ["500","503","unavailable"]):
                time.sleep(5 * (attempt + 1))
            else:
                time.sleep(2)
    return None


# ════════════════════════════════════════════════════════════
# 4.  LOAD DELTA GLOBALS
# ════════════════════════════════════════════════════════════
STANDARD_GLOBALS = {
    "print","warn","error","assert","pcall","xpcall","type","typeof",
    "pairs","ipairs","next","select","unpack","table","string","math",
    "tostring","tonumber","rawget","rawset","rawequal","rawlen","load",
    "setmetatable","getmetatable","require","dofile","collectgarbage",
    "coroutine","io","os","bit32","utf8",
    "task","wait","spawn","delay","game","workspace","script","plugin",
    "Instance","Vector3","Vector2","Vector3int16","Vector2int16",
    "CFrame","Color3","UDim","UDim2","Enum","Ray","Region3",
    "TweenInfo","NumberSequence","ColorSequence","NumberRange",
    "BrickColor","Rect","Random","DateTime","RaycastParams","OverlapParams",
    "PhysicalProperties","Font","shared","_G","_VERSION","tick","elapsedTime",
    "UserSettings","settings",
    "Players","RunService","UserInputService","TweenService","HttpService",
    "ReplicatedStorage","ServerStorage","ServerScriptService","StarterGui",
    "StarterPack","Lighting","SoundService","MarketplaceService",
}

def load_globals() -> tuple[list[dict], set[str]]:
    print("📥 Fetching delta_globals.json from GitHub...")
    r = requests.get(GLOBALS_URL, timeout=15)
    r.raise_for_status()
    data  = r.json()
    funcs = data.get("functions", [])
    whitelist = set(STANDARD_GLOBALS)
    for f in funcs:
        whitelist.add(f["name"])
        for a in f.get("aliases", []):
            if a:
                whitelist.add(a)
    print(f"  ✅ {len(funcs)} executor globals → whitelist has {len(whitelist)} entries")
    return funcs, whitelist


# ════════════════════════════════════════════════════════════
# 5.  SCRIPTBLOX SCRAPER
# ════════════════════════════════════════════════════════════
_OBFUSC = [
    re.compile(r'^[A-Za-z0-9+/]{300,}={0,2}$', re.M),
    re.compile(r'\\[0-9]{2,3}\\[0-9]{2,3}\\[0-9]{2,3}'),
    re.compile(r'local\s+[A-Z_]{25,}\s*='),
]

def _is_obfuscated(code: str) -> bool:
    for p in _OBFUSC:
        if p.search(code):
            return True
    return len(re.findall(r'\b[a-zA-Z]\b', code)) > 120

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
            print(f"  ⚠️  SB page {pg}: {e}")
            time.sleep(2)
    print(f"  ✅ {len(results)} clean ScriptBlox scripts")
    return results


# ════════════════════════════════════════════════════════════
# 6.  AST SYNTAX CHECK
# ════════════════════════════════════════════════════════════
try:
    from luaparser import ast as _lua_ast
    _LUAPARSER = True
except ImportError:
    _LUAPARSER = False
    print("⚠️  luaparser missing — run: pip install luaparser")

def ast_check(code: str) -> tuple[bool, str]:
    if not _LUAPARSER:
        return True, ""
    try:
        _lua_ast.parse(code)
        return True, ""
    except Exception as e:
        return False, str(e)[:300]


# ════════════════════════════════════════════════════════════
# 7.  WHITELIST CHECK
# ════════════════════════════════════════════════════════════
_LUA_KEYWORDS = {
    "if","then","else","elseif","end","while","do","for","in",
    "repeat","until","function","local","return","break","continue",
    "and","or","not","true","false","nil",
}

def whitelist_check(code: str, wl: set) -> tuple[bool, list[str]]:
    top  = re.findall(r'(?<![.:a-zA-Z0-9_])([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
    ns   = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
    all_calls = set(top) | set(ns)
    unknown = [
        f for f in all_calls
        if f not in wl and f not in _LUA_KEYWORDS and len(f) > 2
    ]
    return len(unknown) == 0, unknown


# ════════════════════════════════════════════════════════════
# 8.  SYSTEM PROMPTS
# ════════════════════════════════════════════════════════════
_GEN_SYS = """You are an expert Roblox executor scripter specialising in Delta executor, UNC globals, and advanced Luau.

Generate ONE training sample as raw JSON — no markdown fences, no extra text:
{
  "prompt": "<1-2 sentence natural question a real scripter would ask>",
  "chain_of_thought": "<150-400 words: which APIs to use and exactly WHY, edge cases, how it works step-by-step>",
  "code": "<complete, runnable Luau executor script — no placeholder comments>",
  "explanation": "<3-6 sentences plain English of what the code does>"
}

Hard rules:
1. Use ONLY real Delta/UNC globals (getgenv, hookfunction, getrawmetatable, Drawing.new, syn.request, etc.)
2. Every function call in the code must exist in a real executor
3. chain_of_thought must explain why each executor global was chosen over alternatives
4. code must be complete — no 'TODO', 'your logic here', or empty function bodies"""

_REWRITE_SYS = """You are a Luau executor syntax expert.
Return ONLY the fixed Luau code. No JSON, no markdown fences, no explanation whatsoever."""

_REVIEW_SYS = """You are a strict quality reviewer for a Roblox executor AI training dataset.
Examine the sample and reply ONLY with JSON — nothing else:
{"pass": true|false, "score": 1-10, "issue": "brief note or empty string"}
Scoring guide:
  8-10 → correct globals, complete code, deep reasoning, explanation matches code
  5-7  → minor issues (small logic bug, shallow reasoning, slightly off prompt)
  1-4  → wrong/fake globals, broken code, reasoning doesn't match code, incomplete"""

_QC_SYS = """You are a final quality-control agent for a Roblox executor ML dataset.
Score each sample strictly on: prompt clarity, reasoning depth, code correctness, explanation accuracy.
Reply ONLY with a JSON array — no other text:
[{"idx": 0, "score": 8}, {"idx": 1, "score": 5}, ...]
Be strict: 9-10 = truly excellent; 7-8 = solid; below 7 = mediocre."""


# ════════════════════════════════════════════════════════════
# 9.  BUILD PROMPTS
# ════════════════════════════════════════════════════════════
def build_gen_prompt(entry: dict, whitelist: set) -> str:
    wl_sample = ", ".join(sorted(whitelist)[:50]) + " ..."
    if entry["source"] == "scriptblox":
        return (
            f"Here is a real executor script from ScriptBlox:\n"
            f"Title: {entry['name']}  |  Game: {entry['game']}\n\n"
            f"```lua\n{entry['code_hint']}\n```\n\n"
            f"Generate a training sample where the prompt is the natural question "
            f"that would lead to writing this kind of script. Improve the code — "
            f"make it cleaner, more complete, and more educational.\n"
            f"Valid executor globals include: {wl_sample}"
        )
    else:
        variant_themes = [
            "ESP/wallhack overlay using Drawing API",
            "remote event spy that logs all FireServer calls",
            "anti-AFK that bypasses kick detection",
            "speed hack with smooth interpolation",
            "auto-farm with instance detection",
            "GUI utility with toggle hotkey",
            "stat reader that exposes hidden values",
            "hook-based anti-cheat bypass",
            "fly script with collision avoidance",
            "aimbot using camera CFrame manipulation",
            "noclip that re-enables smoothly",
            "executor environment debugger/inspector",
        ]
        theme = variant_themes[entry.get("variant", 0) % len(variant_themes)]
        return (
            f"Generate a training sample showing how to use the executor global: "
            f"`{entry['name']}`\n\n"
            f"Description: {entry.get('desc', 'Executor global function')}\n"
            f"Category: {entry.get('category', 'executor')}\n"
            f"Parameters: {entry.get('parameters', 'varies')}\n"
            f"Returns: {entry.get('returns', 'varies')}\n\n"
            f"Theme for this sample: **{theme}**\n"
            f"Make it a realistic, complete use case — NOT a hello world or demo.\n"
            f"Valid executor globals include: {wl_sample}"
        )


# ════════════════════════════════════════════════════════════
# 10.  PROCESS ONE ENTRY
# ════════════════════════════════════════════════════════════
def _reject(sid, reason, detail=""):
    with open(REJECTS_LOG, "a") as f:
        f.write(json.dumps({"id": sid, "reason": reason,
                             "detail": str(detail)[:200]}) + "\n")

def process_entry(entry: dict, pool: KeyPool,
                  whitelist: set, done_ids: set) -> "dict | None":
    sid = hashlib.md5(
        json.dumps(entry, sort_keys=True).encode()
    ).hexdigest()[:16]
    if sid in done_ids:
        return None

    w = pool.get_worker()
    if w is None:
        return None

    raw = call_gemini(w, _GEN_SYS, build_gen_prompt(entry, whitelist), expect_json=True)
    if not raw:
        return None

    try:
        sample = json.loads(raw)
        assert all(k in sample for k in ("prompt","chain_of_thought","code","explanation"))
    except Exception:
        _reject(sid, "json_parse_fail", raw[:200])
        return None

    code = sample.get("code", "").strip()
    if len(code) < 30:
        _reject(sid, "code_too_short")
        return None

    ast_ok, ast_err = ast_check(code)
    if not ast_ok:
        w2 = pool.get_worker()
        if not w2:
            return None
        fixed = call_gemini(w2, _REWRITE_SYS,
                            f"Luau syntax error: {ast_err}\n\nCode to fix:\n{code}",
                            expect_json=False, strip_code=True)
        if not fixed:
            _reject(sid, "ast_fail_no_rewrite", ast_err)
            return None
        code = fixed.strip()
        ast_ok2, ast_err2 = ast_check(code)
        if not ast_ok2:
            _reject(sid, "ast_fail_after_rewrite", ast_err2)
            return None
        sample["code"] = code

    wl_ok, unknown = whitelist_check(code, whitelist)
    if not wl_ok:
        w3 = pool.get_worker()
        if not w3:
            return None
        wl_hint = ", ".join(sorted(whitelist)[:60])
        fixed = call_gemini(
            w3, _REWRITE_SYS,
            f"These function names do NOT exist in Delta executor: {unknown}\n"
            f"Replace them with correct equivalents from this list: {wl_hint}\n\n"
            f"Original code:\n{code}",
            expect_json=False, strip_code=True)
        if not fixed:
            _reject(sid, "whitelist_fail_no_rewrite", str(unknown))
            return None
        code = fixed.strip()
        wl_ok2, still_bad = whitelist_check(code, whitelist)
        if not wl_ok2:
            _reject(sid, "whitelist_fail_after_rewrite", str(still_bad))
            return None
        sample["code"] = code

    w4 = pool.get_worker()
    if w4 is None:
        return None

    review_msg = (
        f"Prompt: {sample['prompt']}\n\n"
        f"Code:\n{code}\n\n"
        f"Reasoning (first 400 chars): {sample['chain_of_thought'][:400]}"
    )
    rev_raw = call_gemini(w4, _REVIEW_SYS, review_msg, expect_json=True)
    review_score = 6

    if rev_raw:
        try:
            rev = json.loads(rev_raw)
            review_score = rev.get("score", 6)
            if not rev.get("pass", True) or review_score < 5:
                _reject(sid, "semantic_review_fail", rev.get("issue", ""))
                return None
        except Exception:
            pass

    sample.update({"source": entry["source"], "source_id": sid, "review_score": review_score})
    return sample


# ════════════════════════════════════════════════════════════
# 11.  QC + HUGGINGFACE UPLOAD
# ════════════════════════════════════════════════════════════
def qc_and_upload(samples: list[dict], pool: KeyPool, is_final=False) -> list[dict]:
    if not samples:
        return []
    print(f"\n🔍 QC scoring {len(samples)} samples (Key 10)...")
    qc_key = pool.get_qc()
    passed = []
    BATCH  = 8

    for i in range(0, len(samples), BATCH):
        chunk = samples[i : i + BATCH]
        batch_str = "\n---\n".join(
            f"idx:{j} | prompt:{s['prompt'][:100]} | "
            f"code_len:{len(s.get('code',''))} | "
            f"review:{s.get('review_score',6)}"
            for j, s in enumerate(chunk)
        )
        if not qc_key.wait_and_acquire():
            passed.extend(s for s in chunk if s.get("review_score", 6) >= 6)
            continue
        raw = call_gemini(qc_key, _QC_SYS, batch_str, expect_json=True)
        score_map = {}
        try:
            score_map = {item["idx"]: item["score"] for item in json.loads(raw)}
        except Exception:
            pass
        for j, s in enumerate(chunk):
            sc = score_map.get(j, s.get("review_score", 6))
            if sc >= 7:
                s["qc_score"] = sc
                passed.append(s)

    print(f"  ✅ QC: {len(passed)}/{len(samples)} passed (score ≥ 7)")
    if passed:
        _hf_upload(passed, is_final=is_final)
    return passed


def _hf_upload(samples: list[dict], is_final=False):
    label = "FINAL" if is_final else "batch"
    print(f"\n📤 Uploading {len(samples)} new samples to {HF_DATASET}  [{label}]...")
    try:
        try:
            existing_ds = load_dataset(HF_DATASET, token=HF_TOKEN, split="train")
            existing_df = existing_ds.to_pandas()
            new_df      = pd.DataFrame(samples)
            combined    = pd.concat([existing_df, new_df], ignore_index=True)
            combined    = combined.drop_duplicates(subset=["source_id"], keep="last")
        except Exception:
            combined = pd.DataFrame(samples)

        COLS = ["prompt","chain_of_thought","code","explanation",
                "source","source_id","review_score","qc_score"]
        combined = combined[[c for c in COLS if c in combined.columns]]
        ds = Dataset.from_pandas(combined, preserve_index=False)
        ds.push_to_hub(HF_DATASET, token=HF_TOKEN,
                       commit_message=f"{label} upload — {len(combined)} total rows")
        print(f"  ✅ HuggingFace now has {len(combined)} rows")
    except Exception as e:
        print(f"  ❌ HF upload error: {e}")
        bak = OUT / f"backup_{int(time.time())}.parquet"
        pd.DataFrame(samples).to_parquet(bak, index=False)
        print(f"  💾 Local backup: {bak}")


# ════════════════════════════════════════════════════════════
# 12.  CHECKPOINT STATE
# ════════════════════════════════════════════════════════════
class State:
    def __init__(self):
        self.done_ids    : set[str]   = set()
        self.total_good  : int        = 0
        self.upload_buf  : list[dict] = []
        self._lk = threading.Lock()

    def load(self, pool: KeyPool):
        if STATE_FILE.exists():
            d = json.loads(STATE_FILE.read_text())
            self.done_ids   = set(d.get("done_ids", []))
            self.total_good = d.get("total_good", 0)
            pool.restore(d.get("keys", {}))
            print(f"▶  Resumed from checkpoint: {self.total_good} samples done")
        if GOOD_JSONL.exists():
            with open(GOOD_JSONL) as f:
                for ln in f:
                    try:
                        self.upload_buf.append(json.loads(ln))
                    except Exception:
                        pass

    def save(self, pool: KeyPool):
        with self._lk:
            STATE_FILE.write_text(json.dumps({
                "done_ids":   list(self.done_ids),
                "total_good": self.total_good,
                "keys":       pool.save(),
                "ts":         datetime.now().isoformat(),
            }))

    def record(self, sample: dict):
        with self._lk:
            self.done_ids.add(sample["source_id"])
            self.total_good += 1
            self.upload_buf.append(sample)
            with open(GOOD_JSONL, "a") as fh:
                fh.write(json.dumps(sample) + "\n")

    def flush(self) -> list[dict]:
        with self._lk:
            buf = self.upload_buf[:]
            self.upload_buf.clear()
            open(GOOD_JSONL, "w").close()
        return buf


# ════════════════════════════════════════════════════════════
# 13.  MAIN
# ════════════════════════════════════════════════════════════
def main():
    t0       = time.time()
    deadline = t0 + KAGGLE_LIMIT_H * 3600 - SAFETY_BUFFER_M * 60

    print("╔══════════════════════════════════════════╗")
    print("║  Delta Executor Dataset Generator        ║")
    print(f"║  Target: {TARGET_SAMPLES:,} samples → {HF_DATASET}  ║")
    print("╚══════════════════════════════════════════╝\n")

    pool  = KeyPool()
    state = State()
    state.load(pool)
    pool.print_status()

    globals_list, whitelist = load_globals()
    sb_data = scrape_scriptblox(pages=200)

    queue = []
    for g in globals_list:
        for v in range(VARIANTS_PER_GLB):
            e = {**g, "source": "delta_global", "variant": v}
            queue.append(e)
    queue.extend(sb_data)
    random.shuffle(queue)

    queue = [
        e for e in queue
        if hashlib.md5(
            json.dumps(e, sort_keys=True).encode()
        ).hexdigest()[:16] not in state.done_ids
    ]

    h_left = (deadline - time.time()) / 3600
    print(f"📋 Queue : {len(queue):,} items remaining")
    print(f"⏰ Time  : {h_left:.1f} h until safety stop\n")

    def _shutdown(sig=None, frame=None):
        print("\n⚠️  Shutdown — final upload in progress...")
        buf = state.flush()
        if buf:
            qc_and_upload(buf, pool, is_final=True)
        state.save(pool)
        print(f"✅ Saved {state.total_good:,} total samples.  Bye.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    bar = tqdm(total=TARGET_SAMPLES, initial=state.total_good,
               desc="Good samples", unit="smp", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=9) as ex:
        pending = {}
        qi      = 0

        while state.total_good < TARGET_SAMPLES:
            if time.time() > deadline:
                print(f"\n⏰ Reached safety deadline — stopping.")
                break

            while len(pending) < 9 and qi < len(queue):
                fut = ex.submit(process_entry, queue[qi], pool, whitelist, state.done_ids)
                pending[fut] = queue[qi]
                qi += 1

            if not pending:
                break

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
                    state.save(pool)
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
        print(f"📤 Final QC + upload of {len(remaining)} remaining samples...")
        qc_and_upload(remaining, pool, is_final=True)

    state.save(pool)
    pool.print_status()
    elapsed = (time.time() - t0) / 3600
    print(f"\n✅ All done in {elapsed:.1f}h.")
    print(f"   Dataset → https://huggingface.co/datasets/{HF_DATASET}")


if __name__ == "__main__":
    main()
