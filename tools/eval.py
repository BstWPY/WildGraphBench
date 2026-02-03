#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for Wikipedia-citation QA (Type-1 / Type-2 / Type-3)

- Type-1 / Type-2: binary grading (1 = correct, 0 = incorrect) against the reference answer.
- Type-3: fact-level binary grading - each key fact must be correctly stated

Prompts are in ENGLISH. Console logs are in CHINESE.

Input:
  --gold  Path to gold JSONL (qa.jsonl: id/question/answer/question_type)
  --pred  Path to predictions JSON/JSONL (predictions.json or local_answers.jsonl)
  --outdir Output directory for scored.jsonl and report.json
"""

import os, json, argparse, time, sys, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio

from dotenv import load_dotenv
load_dotenv()

# API configuration - set via environment variables or .env file
# Supports OpenAI-compatible APIs (OpenAI, Anthropic, Azure, etc.)
APY_KEY = os.environ.get("EVAL_API_KEY", os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE"))
BASE_URL = os.environ.get("EVAL_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
MODEL = os.environ.get("EVAL_MODEL", "gpt-5-mini")

# -----------------------
# Async Claude API helpers
# -----------------------
async def call_claude_api_async(
    session: aiohttp.ClientSession,
    messages: List[Dict],
    max_tokens: int = 512,
    temperature: float = 0.0,
    debug: bool = False,
    retry: int = 3
) -> str:
    """Async call to Claude API"""
    url = f"{BASE_URL}/messages"
    headers = {
        "Authorization": f"Bearer {APY_KEY}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages
    }
    
    for attempt in range(retry):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if debug:
                    print(f"\n[DEBUG] Request URL: {url}")
                    print(f"[DEBUG] Response status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"API error {response.status}: {error_text}"
                    print(f"  [Claude API Error] {error_msg}", file=sys.stderr)
                    if attempt < retry - 1:
                        await asyncio.sleep(2 ** attempt)  # exponential backoff
                        continue
                    return ""
                
                result = await response.json()
                
                if debug:
                    print(f"[DEBUG] Full Response JSON: {json.dumps(result, ensure_ascii=False, indent=2)}")
                
                # First try OpenAI compatible format
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice:
                        content = choice["message"].get("content", "")
                        if debug:
                            print(f"[DEBUG] Extracted from OpenAI format: {content}")
                        return content
                
                # Fallback: native Claude format
                if "content" in result and len(result["content"]) > 0:
                    content = result["content"][0].get("text", "")
                    if debug:
                        print(f"[DEBUG] Extracted from Claude format: {content}")
                    return content
                
                if debug:
                    print(f"[DEBUG] No recognized response format found")
                
                return ""
                
        except asyncio.TimeoutError:
            print(f"  [Claude API Timeout] Request timeout after 120s (attempt {attempt+1}/{retry})", file=sys.stderr)
            if attempt < retry - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return ""
        except aiohttp.ClientError as e:
            print(f"  [Claude API Request Error] {e} (attempt {attempt+1}/{retry})", file=sys.stderr)
            if attempt < retry - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return ""
        except Exception as e:
            print(f"  [Claude API Exception] {e}", file=sys.stderr)
            if debug:
                import traceback
                traceback.print_exc()
            return ""
    
    return ""

def extract_json(text: str) -> dict:
    """Extract JSON from text (supports markdown code blocks)"""
    if not text:
        return {}
    
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    
    # Extract ```json ... ``` code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # Extract first {...} object
    match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    # Try more relaxed extraction
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            potential_json = text[start:end+1]
            return json.loads(potential_json)
    except:
        pass
    
    return {}

async def json_chat_async(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    debug: bool = False,
    retry: int = 3
) -> dict:
    """Async call to Claude and parse JSON response"""
    system_prompt = """You are a careful grader. You must return ONLY a valid JSON object.
Do not include any explanatory text before or after the JSON.
Do not wrap the JSON in markdown code blocks.
Just return the raw JSON object directly."""

    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
    ]
    
    content = await call_claude_api_async(
        session, messages,
        max_tokens=max_tokens,
        temperature=temperature,
        debug=debug,
        retry=retry
    )
    
    if debug:
        print(f"\n[DEBUG] Claude Raw Response:\n{content}\n")
    
    if not content:
        return {}
    
    # Try to extract JSON
    data = extract_json(content)
    
    if not data and debug:
        print(f"[WARNING] Failed to parse JSON from response. Raw content:")
        print(f"{content[:500]}..." if len(content) > 500 else content)
    
    return data

# -----------------------
# IO helpers
# -----------------------
def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file"""
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def load_json_or_jsonl(path: Path) -> List[dict]:
    """Load JSON or JSONL file"""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
    except:
        pass
    
    return load_jsonl(path)

def normalize_type(t: str) -> str:
    """Normalize question type name"""
    t = (t or "").strip().lower()
    if t in ("type1", "single", "single_fact", "single-fact"): 
        return "single_fact"
    if t in ("type2", "multi", "multi_fact", "multi-fact"):   
        return "multi_fact"
    if t in ("type3", "summary", "summ"):       
        return "summary"
    return t or "unknown"

def extract_question_type(g: dict) -> str:
    """Extract question type from gold record"""
    if "type" in g:
        return normalize_type(g.get("type", ""))
    
    if "question_type" in g:
        qt = g.get("question_type", [])
        if isinstance(qt, list) and len(qt) > 0:
            return normalize_type(qt[0])
        elif isinstance(qt, str):
            return normalize_type(qt)
    
    return "unknown"

# -----------------------
# Judge prompts (EN)
# -----------------------
BIN_PROMPT = """Grade whether the CANDIDATE ANSWER is CORRECT (1) or INCORRECT (0) relative to the REFERENCE ANSWER.

Rules:
- Paraphrasing is allowed if meaning is identical.
- Grade 0 if candidate omits/distorts key facts.
- Grade 0 if candidate contradicts reference.
- Grade 1 if candidate matches all key facts.

Return this exact JSON format:
{{"score": 0, "reason": "brief justification"}}

QUESTION:
{question}

REFERENCE ANSWER:
{gold}

CANDIDATE ANSWER:
{pred}
""".strip()

SUMMARY_PROMPT = """Grade the CANDIDATE SUMMARY using FACT-LEVEL evaluation.

TASK:
1. Extract all key facts from the REFERENCE SUMMARY
2. Check each fact in the CANDIDATE SUMMARY:
   - Is the fact stated correctly? (no contradictions, no distortions)
   - Is the fact present? (not omitted)
3. Assign binary score:
   - Score 1: ALL key facts are correctly stated
   - Score 0: ANY key fact is missing, contradicted, or distorted

STRICT RULES:
- Numbers, names, dates must be EXACT (unless clearly paraphrased with same meaning)
- Any hallucinated facts that contradict reference → Score 0
- Missing critical facts → Score 0
- Paraphrasing is OK only if meaning is IDENTICAL
- Extra details not in reference are OK if they don't contradict
- Focus on FACTUAL CORRECTNESS, not writing style

Return this exact JSON format:
{{"score": 0, "reason": "brief explanation of which facts failed"}}
OR
{{"score": 1, "reason": "all key facts correctly stated"}}

QUESTION:
{question}

REFERENCE SUMMARY:
{gold}

CANDIDATE SUMMARY:
{pred}
""".strip()
# -----------------------
# Type-3: statement-level extraction & alignment
# -----------------------

MAX_STATEMENTS_PER_SIDE = 20       # max statements to extract per side (prevent too many fragments)
TOP_K_MATCH = 5                    # only look at top-k candidates per statement
STATEMENT_MATCH_THRESHOLD = 0.7    # similarity threshold for "covered/supported"


async def extract_statements_async(
    session: aiohttp.ClientSession,
    answer: str,
    question: str,
    snippet: Optional[str] = None,
    is_gold: bool = False,
    debug: bool = False,
) -> List[str]:
    """
    Use gpt-5-mini to decompose the summary answer into atomic factual statements.

    - For gold (wiki answer), will provide raw snippet as context, but explicitly require not to introduce facts from snippet that are not in the answer.
    - For pred, only look at the model answer itself.
    """
    answer = (answer or "").strip()
    if not answer:
        return []

    role_desc = "REFERENCE (Wikipedia) SUMMARY ANSWER" if is_gold else "CANDIDATE (model) SUMMARY ANSWER"

    snippet_part = ""
    if is_gold and snippet:
        snippet_part = f"""

OPTIONAL ORIGINAL WIKI SNIPPET (for context only, may include citations):
{snippet}

IMPORTANT:
- Use the snippet ONLY to better understand boundaries and context.
- DO NOT add any fact that is not clearly present in the SUMMARY ANSWER.
"""

    prompt = f"""
You will be given a QUESTION and a {role_desc}.

Your task is to extract a list of short, atomic factual statements from the SUMMARY ANSWER.

Requirements:
- Each statement must express exactly ONE factual claim (subject + predicate + key objects).
- Statements should be concise declarative sentences (no bullet markers needed).
- Do NOT invent new information that is not present in the SUMMARY ANSWER.
- You may lightly rewrite for clarity, but the meaning must stay the same.
- Return at most {MAX_STATEMENTS_PER_SIDE} statements.
- If the SUMMARY ANSWER is empty or purely non-factual, return an empty list.

QUESTION:
{question}

SUMMARY ANSWER:
{answer}
{snippet_part}

Return JSON ONLY in this format:
{{"statements": ["...", "..."]}}
""".strip()

    data = await json_chat_async(
        session,
        prompt,
        max_tokens=10000,
        temperature=0.0,
        debug=debug,
        retry=3,
    )

    if not data:
        return []

    statements = data.get("statements", [])
    out: List[str] = []

    if isinstance(statements, list):
        for s in statements:
            s = str(s).strip()
            if s:
                out.append(s)
    elif isinstance(statements, str):
        # Error tolerance: if a long string is returned directly, roughly split by period/semicolon
        for s in re.split(r'[。.;；]\s*', statements):
            s = s.strip()
            if s:
                out.append(s)

    # Deduplicate + truncate
    seen = set()
    unique_out = []
    for s in out:
        if s not in seen:
            seen.add(s)
            unique_out.append(s)

    return unique_out[:MAX_STATEMENTS_PER_SIDE]


async def match_statements_async(
    session: aiohttp.ClientSession,
    ref_answer: str,
    cand_statements: List[str],
    debug: bool = False,
) -> List[int]:
    """
    Given a REFERENCE ANSWER (complete paragraph) and several CANDIDATE STATEMENTS,
    determine whether each statement is factually supported by the REFERENCE ANSWER.

    Returns a 0/1 list of length len(cand_statements):
      - 1 means the statement can be clearly derived/supported from ref_answer
      - 0 means ref_answer does not clearly express that fact, or contradicts/partially disagrees with it
    """
    if not cand_statements:
        return []

    ref_answer = (ref_answer or "").strip()
    if not ref_answer:
        # When there is no reference answer, all statements are considered unsupported
        return [0] * len(cand_statements)

    cand_block = "\n".join(
        f"[{i+1}] {s}" for i, s in enumerate(cand_statements)
    )

    prompt = f"""
You will be given a REFERENCE ANSWER (a short paragraph) and a list of CANDIDATE STATEMENTS.

For EACH candidate statement, decide whether it is fully and factually supported by the REFERENCE ANSWER.

Guidelines:
- Return 1 only if the statement can be clearly and fully inferred from the REFERENCE ANSWER with no contradictions.
- If the statement is partially wrong, missing key constraints, or contradicted, return 0.
- If the REFERENCE ANSWER does not clearly state the fact, return 0.
- Ignore wording differences; focus on factual meaning (entities, numbers, dates, core relations).

REFERENCE ANSWER:
{ref_answer}

CANDIDATE STATEMENTS (indexed):
{cand_block}

Return JSON ONLY in this format:
{{"scores": [{{"idx": 1, "score": 1}}, {{\"idx\": 2, \"score\": 0}}]}}
""".strip()

    data = await json_chat_async(
        session,
        prompt,
        max_tokens=10000,
        temperature=0.0,
        debug=debug,
        retry=3,
    )

    # Default all to 0
    flags = [0] * len(cand_statements)
    if not data:
        return flags

    raw_scores = data.get("scores") or data.get("results") or data.get("items") or []

    if isinstance(raw_scores, dict):
        raw_scores = raw_scores.get("scores", [])

    if not isinstance(raw_scores, list):
        return flags

    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        idx = item.get("idx")
        sc = item.get("score", 0)
        try:
            idx_int = int(idx)
            if 1 <= idx_int <= len(cand_statements):
                # Compatible with 0/1 or 0.0~1.0: treat >0.5 as 1
                sc_f = float(sc)
                supported = 1 if sc_f >= 0.5 else 0
                flags[idx_int - 1] = supported
        except Exception:
            continue

    return flags



async def score_alignment_one_direction_async(
    session: aiohttp.ClientSession,
    source_statements: List[str],
    ref_answer: str,
    *,
    debug: bool = False,
    direction: str = "gold->pred",
) -> Tuple[float, List[Dict]]:
    """
    Unidirectional alignment (based on 0/1 judgment of "full answer vs single statement"):

    - source_statements: A set of statements that need to be covered/verified by ref_answer
      * For gold->pred: gold_statements
      * For pred->gold: pred_statements
    - ref_answer: The complete answer text used as "evidence"
      * For gold->pred: pred_answer
      * For pred->gold: gold_answer

    For each statement in source_statements, determine whether it is factually supported by ref_answer.
    hit_rate = number of supported statements / total number of statements
    """
    details: List[Dict] = []
    if not source_statements:
        return 0.0, details

    # Call the model once to score each statement in this batch with 0/1
    flags = await match_statements_async(
        session,
        ref_answer=ref_answer,
        cand_statements=source_statements,
        debug=debug,
    )

    hits = 0
    total = len(source_statements)
    for i, s in enumerate(source_statements):
        supported = 0
        if i < len(flags) and flags[i] in (1, True):
            supported = 1
        hits += supported

        details.append({
            "idx": i,
            "statement": s,
            "supported": supported,
            "direction": direction,
        })

    hit_rate = hits / total if total > 0 else 0.0
    return hit_rate, details


# -----------------------
# Async Graders
# -----------------------
async def grade_binary_async(
    session: aiohttp.ClientSession,
    q: str,
    gold: str,
    pred: str,
    debug: bool = False
) -> Tuple[int, str]:
    """Async binary grading"""
    if not pred.strip():
        return 0, "Empty answer"
    
    prompt = BIN_PROMPT.format(question=q, gold=gold, pred=pred)
    data = await json_chat_async(session, prompt, max_tokens=2048, temperature=0.0, debug=debug, retry=3)
    
    if not data:
        return 0, "Claude returned invalid JSON"
    
    score_raw = data.get("score", 0)
    if isinstance(score_raw, str):
        score_raw = score_raw.strip().lower()
        if score_raw in ("1", "true", "correct", "yes"):
            score = 1
        else:
            score = 0
    else:
        score = int(score_raw in (1, "1", True))
    
    reason = (data.get("reason") or "").strip() or "N/A"
    return score, reason

async def grade_summary_async(
    session: aiohttp.ClientSession,
    q: str,
    gold: str,
    pred: str,
    debug: bool = False
) -> Tuple[int, str]:
    """Async summary grading"""
    if not pred.strip():
        return 0, "Empty answer"
    
    prompt = SUMMARY_PROMPT.format(question=q, gold=gold, pred=pred)
    data = await json_chat_async(session, prompt, max_tokens=2048, temperature=0.0, debug=debug, retry=3)
    
    if not data:
        return 0, "Claude returned invalid JSON"
    
    score_raw = data.get("score", 0)
    if isinstance(score_raw, str):
        score_raw = score_raw.strip().lower()
        if score_raw in ("1", "true", "correct", "yes"):
            score = 1
        else:
            score = 0
    else:
        score = int(score_raw in (1, "1", True))
    
    reason = (data.get("reason") or "").strip() or "None"
    return score, reason

# -----------------------
# Extract answer from prediction
# -----------------------
def extract_pred_answer(p: dict) -> str:
    """Extract answer from prediction record"""
    if not p:
        return ""
    
    if "pred_answer" in p:
        return (p.get("pred_answer") or "").strip()
    
    if "answer" in p:
        return (p.get("answer") or "").strip()
    
    if "result" in p:
        result = p.get("result", {})
        if isinstance(result, dict):
            return (result.get("response") or "").strip()
    
    if "response" in p:
        return (p.get("response") or "").strip()
    
    return ""

# -----------------------
# Alignment
# -----------------------
def align_by_question(golds: List[dict], preds: List[dict]) -> List[Tuple[dict, Optional[dict]]]:
    """Align by question"""
    q2pred = {}
    for p in preds:
        question = (p.get("question") or "").strip()
        if question:
            q2pred[question] = p
    
    out = []
    for g in golds:
        question = (g.get("question") or "").strip()
        out.append((g, q2pred.get(question)))
    
    return out

def align_by_id(golds: List[dict], preds: List[dict]) -> List[Tuple[dict, Optional[dict]]]:
    """Align by id"""
    id2pred = {}
    for p in preds:
        pid = p.get("id")
        if pid is not None:
            id2pred[pid] = p
    
    out = []
    for g in golds:
        gid = g.get("id")
        out.append((g, id2pred.get(gid)))
    
    return out

def align(golds: List[dict], preds: List[dict]) -> List[Tuple[dict, Optional[dict]]]:
    """Smart alignment"""
    has_id = any(p.get("id") is not None for p in preds)
    
    if has_id:
        print("[INFO] Using ID alignment mode")
        return align_by_id(golds, preds)
    else:
        print("[INFO] Using Question alignment mode")
        return align_by_question(golds, preds)

# -----------------------
# Async evaluation worker
# -----------------------
async def evaluate_single(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    idx: int,
    g: dict,
    p: Optional[dict],
    debug: bool = False
) -> dict:
    """Evaluate a single sample (with concurrency control)"""
    async with semaphore:
        gid = g.get("id")
        gtype = extract_question_type(g)
        question = (g.get("question") or "").strip()
        gold_answer = (g.get("answer") or "").strip()
        pred_answer = extract_pred_answer(p)

        rec = {
            "id": gid,
            "idx": idx,
            "type": gtype,
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
        }

        # Enable detailed logging for first 3 samples
        debug_this = debug or idx <= 3

        try:
            if gtype in ("single_fact", "multi_fact"):
                score, reason = await grade_binary_async(
                    session, question, gold_answer, pred_answer, debug=debug_this
                )
                rec.update({"score": score, "reason": reason})

            elif gtype == "summary":
                # ---------- Type-3: Use pre-cached gold_statements ----------
                # 1) Read pre-extracted statements from gold record
                gold_statements = g.get("gold_statements") or g.get("statements") or []
                if isinstance(gold_statements, str):
                    # In case it's written as a big string, simply split by line/semicolon
                    tmp = []
                    for s in re.split(r'[\n;；]+', gold_statements):
                        s = s.strip()
                        if s:
                            tmp.append(s)
                    gold_statements = tmp

                if not isinstance(gold_statements, list):
                    gold_statements = []

                # Try to get wiki_snippet from gold (for debug/visualization only, no longer used for statement extraction)
                snippet = ""
                src_list = g.get("source") or g.get("sources")
                if isinstance(src_list, list) and src_list and isinstance(src_list[0], dict):
                    snippet = (src_list[0].get("wiki_snippet") or "").strip()

                # 2) Only extract atomic statements from pred once more
                pred_statements = await extract_statements_async(
                    session,
                    pred_answer,
                    question,
                    snippet=None,
                    is_gold=False,
                    debug=debug_this,
                )

                if not gold_statements or not pred_statements:
                    coverage = 0.0
                    stmt_acc = 0.0
                    f1 = 0.0
                    score = 0
                    reason = f"Statement extraction failed or empty (gold={len(gold_statements)}, pred={len(pred_statements)})"
                else:
                    # 3) gold -> pred: coverage (recall)
                    #    Check whether each gold_statement is supported by pred_answer
                    coverage, cov_details = await score_alignment_one_direction_async(
                        session,
                        source_statements=gold_statements,
                        ref_answer=pred_answer,
                        debug=debug_this,
                        direction="gold->pred",
                    )
                    # 4) pred -> gold: accuracy (precision)
                    #    Check whether each pred_statement is supported by gold_answer
                    stmt_acc, acc_details = await score_alignment_one_direction_async(
                        session,
                        source_statements=pred_statements,
                        ref_answer=gold_answer,
                        debug=debug_this,
                        direction="pred->gold",
                    )

                    if coverage > 0.0 and stmt_acc > 0.0:
                        f1 = 2 * coverage * stmt_acc / (coverage + stmt_acc)
                    else:
                        f1 = 0.0

                    # Strict binary classification unchanged
                    score = 1 if (coverage >= 1.0 and stmt_acc >= 1.0) else 0
                    reason = (
                        f"statement-level bidirectional alignment: "
                        f"coverage={coverage:.3f}, accuracy={stmt_acc:.3f}, f1={f1:.3f}"
                    )

                rec.update({
                    "score": score,
                    "reason": reason,
                    "coverage": coverage,
                    "statement_accuracy": stmt_acc,
                    "statement_f1": f1,
                    "gold_statements": gold_statements,
                    "pred_statements": pred_statements,
                })
            else:
                rec.update({"score": 0, "reason": "Unknown question type"})
                
        except Exception as e:
            rec.update({"score": 0, "reason": f"Evaluation error: {e}"})
            if debug:
                import traceback
                traceback.print_exc()

        return rec

# -----------------------
# Main async evaluation
# -----------------------
async def run_evaluation(
    pairs: List[Tuple[dict, Optional[dict]]],
    scored_path: Path,
    max_concurrent: int = 10,
    debug: bool = False
) -> List[dict]:
    """Concurrent evaluation of all samples"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    connector = aiohttp.TCPConnector(limit=max_concurrent * 2)
    timeout = aiohttp.ClientTimeout(total=120)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            evaluate_single(session, semaphore, i, g, p, debug)
            for i, (g, p) in enumerate(pairs, start=1)
        ]
        
        # Use tqdm to show progress
        results = []
        with open(scored_path, "w", encoding="utf-8") as fout:
            for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Evaluation Progress"):
                rec = await coro
                results.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                
                # Print summary info
                print(f"[{rec['idx']:04d}] ID={rec['id']} {rec['type']:12s} score={rec['score']} | {rec['reason'][:60]}")
        
        return results

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, required=True, 
                    help="Gold JSONL file (e.g., qa.jsonl)")
    ap.add_argument("--pred", type=str, required=True, 
                    help="Predictions JSON/JSONL file")
    ap.add_argument("--outdir", type=str, required=False, 
                    default="./eval_output",
                    help="Output directory for evaluation results")
    ap.add_argument("--max-concurrent", type=int, default=10,
                    help="Maximum concurrency (default 10)")
    ap.add_argument("--debug", action="store_true", help="Print detailed debug info")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scored_path = outdir / "scored2.jsonl"
    report_path = outdir / "report2.json"

    # Load data
    golds = load_jsonl(Path(args.gold))
    preds = load_json_or_jsonl(Path(args.pred))
    
    # Alignment
    pairs = align(golds, preds)

    print(f"\n===== Starting Evaluation =====")
    print(f"Gold count: {len(golds)} | Pred count: {len(preds)} | After alignment: {len(pairs)}")
    print(f"Evaluation model: Claude 4.5 Haiku ({MODEL})")
    print(f"Max concurrency: {args.max_concurrent}")
    print(f"Grading criteria: All question types use binary grading (fact-level strict matching)")
    print(f"Debug mode: {args.debug}\n")

    # Run async evaluation
    results = asyncio.run(run_evaluation(pairs, scored_path, args.max_concurrent, args.debug))

    # Compute statistics
    total = len(results)
    t1_correct = sum(1 for r in results if r.get("type") == "single_fact" and r.get("score") == 1)
    t2_correct = sum(1 for r in results if r.get("type") == "multi_fact" and r.get("score") == 1)
    t3_correct = sum(1 for r in results if r.get("type") == "summary" and r.get("score") == 1)

    n1 = sum(1 for g, _ in pairs if extract_question_type(g) == "single_fact")
    n2 = sum(1 for g, _ in pairs if extract_question_type(g) == "multi_fact")
    n3 = sum(1 for g, _ in pairs if extract_question_type(g) == "summary")

    acc1 = (t1_correct / n1) if n1 else 0.0
    acc2 = (t2_correct / n2) if n2 else 0.0
    acc3 = (t3_correct / n3) if n3 else 0.0

    # Added: macro-average coverage / accuracy / f1 for Type-3
    sum_cov = sum(r.get("coverage", 0.0) for r in results if r.get("type") == "summary")
    sum_stmt_acc = sum(r.get("statement_accuracy", 0.0) for r in results if r.get("type") == "summary")
    sum_stmt_f1 = sum(r.get("statement_f1", 0.0) for r in results if r.get("type") == "summary")

    avg_cov = (sum_cov / n3) if n3 else 0.0
    avg_stmt_acc = (sum_stmt_acc / n3) if n3 else 0.0
    avg_stmt_f1 = (sum_stmt_f1 / n3) if n3 else 0.0

    report = {
        "total_items": total,
        "single_fact": {
            "num": n1,
            "correct": t1_correct,
            "accuracy": round(acc1, 4)
        },
        "multi_fact":  {
            "num": n2,
            "correct": t2_correct,
            "accuracy": round(acc2, 4)
        },
        "summary":     {
            "num": n3,
            "correct": t3_correct,
            "accuracy_binary": round(acc3, 4),
            "coverage_avg": round(avg_cov, 4),
            "statement_accuracy_avg": round(avg_stmt_acc, 4),
            "statement_f1_avg": round(avg_stmt_f1, 4),
        },
        "overall_accuracy": round(
            (t1_correct + t2_correct + t3_correct) / total if total else 0.0,
            4
        ),
        "details": {
            "scored_file": str(scored_path),
            "judge_model": MODEL,
            "max_concurrent": args.max_concurrent,
            "grading_method": (
                "Type1/2: binary exact; "
                "Type3: statement-level coverage/accuracy via answer-vs-statement entailment + strict binary"
            )

        }
    }
    
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== Evaluation Complete =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nResult files:\n- Scored output: {scored_path}\n- Summary report: {report_path}\n")

if __name__ == "__main__":
    main()
