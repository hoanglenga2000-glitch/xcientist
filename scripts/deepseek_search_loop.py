"""
DeepSeek Iterative Search Loop — Fixed V2.
LLM generates feature code → train+eval → feedback → iterate.
"""
import json, os, re, time, subprocess, hashlib, warnings, sys
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
WORKSTATION_URL = "http://127.0.0.1:8088"
BEST_KNOWN = {"spaceship_titanic": 0.8163, "titanic": 0.8283, "digit_recognizer": 0.975}

def call_deepseek(prompt, max_tokens=1200):
    try:
        r = subprocess.run(["curl","-s","-X","POST",f"{WORKSTATION_URL}/api/llm/deepseek/smoke",
            "-H","Content-Type: application/json","-d",json.dumps({"prompt":prompt})],
            capture_output=True, text=True, timeout=60)
        d = json.loads(r.stdout)
        return d.get("content","") if d.get("ok") else f"ERR:{d.get('error')}"
    except Exception as e:
        return f"ERR:{e}"

EXEC_TEMPLATE = '''
import numpy as np, pandas as pd, json, os, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from datetime import datetime

train = pd.read_csv(r"D:/桌面/codex/科研港科技/tasks/{task_id}/data/train.csv")
test = pd.read_csv(r"D:/桌面/codex/科研港科技/tasks/{task_id}/data/test.csv")

# === V2 BASELINE FEATURES (keep the strong foundation) ===
def baseline_features(df):
    df = df.copy()
    df["Cabin_deck"] = df["Cabin"].str.split("/").str[0].fillna("Unknown")
    df["Cabin_num"] = pd.to_numeric(df["Cabin"].str.split("/").str[1], errors="coerce").fillna(-1)
    df["Cabin_side"] = df["Cabin"].str.split("/").str[2].fillna("Unknown")
    df["Group"] = df["PassengerId"].str[:4]
    for c in ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]: df[c]=df[c].fillna(0)
    df["TotalSpend"] = df["RoomService"]+df["FoodCourt"]+df["ShoppingMall"]+df["Spa"]+df["VRDeck"]
    df["HasSpend"] = (df["TotalSpend"]>0).astype(int)
    df["Age"] = pd.to_numeric(df["Age"],errors="coerce").fillna(27)
    for c in ["VIP","CryoSleep"]: df[c]=df[c].fillna(False)
    df["VIP_Age"] = df["VIP"].astype(int)*df["Age"]
    df["CryoSleep_Spend"] = df["CryoSleep"].astype(int)*df["TotalSpend"]
    df["GroupSize"] = df["Group"].map(df.groupby("Group").size())
    return df

train = baseline_features(train)
test = baseline_features(test)

{feature_code}

target = "{target_col}"
y = (train[target]==True).astype(int).values
drop = [target,"PassengerId","Name","Cabin"]

X = train.drop(columns=[c for c in drop if c in train.columns], errors="ignore")
Xt = test.drop(columns=[c for c in drop if c in test.columns], errors="ignore")

for col in X.columns:
    if X[col].dtype == object:
        le = LabelEncoder()
        combined = pd.concat([X[col].astype(str), Xt[col].astype(str)])
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        Xt[col] = le.transform(Xt[col].astype(str))

common = [c for c in X.columns if c in Xt.columns]
X_arr = X[common].fillna(-1).astype(float).values
Xt_arr = Xt[common].fillna(-1).astype(float).values
X_arr = StandardScaler().fit_transform(X_arr)
Xt_arr = StandardScaler().fit_transform(Xt_arr)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_cb = np.zeros(len(y)); tpred_cb = np.zeros(len(Xt_arr))
oof_hgb = np.zeros(len(y)); tpred_hgb = np.zeros(len(Xt_arr))

for tr, val in skf.split(X_arr, y):
    cb = CatBoostClassifier(iterations=800, learning_rate=0.02, depth=7, random_seed=42, verbose=False, thread_count=-1)
    cb.fit(X_arr[tr], y[tr])
    oof_cb[val] = cb.predict_proba(X_arr[val])[:, 1]
    tpred_cb += cb.predict_proba(Xt_arr)[:, 1] / 5

    hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=None, random_state=42)
    hgb.fit(X_arr[tr], y[tr])
    oof_hgb[val] = hgb.predict_proba(X_arr[val])[:, 1]
    tpred_hgb += hgb.predict_proba(Xt_arr)[:, 1] / 5

cb_acc = float(accuracy_score(y, (oof_cb>0.5).astype(int)))
hgb_acc = float(accuracy_score(y, (oof_hgb>0.5).astype(int)))
blend = 0.6*oof_cb + 0.4*oof_hgb
blend_acc = float(accuracy_score(y, (blend>0.5).astype(int)))
blend_test = 0.6*tpred_cb + 0.4*tpred_hgb

sub = pd.read_csv(r"D:/桌面/codex/科研港科技/tasks/{task_id}/data/sample_submission.csv")
sub.iloc[:,-1] = (blend_test > 0.5).astype(bool)
sub.to_csv(r"{output_dir}/submission.csv", index=False)

pd.DataFrame({{"oof_cb":oof_cb,"oof_hgb":oof_hgb,"true":y}}).to_csv(r"{output_dir}/oof.csv", index=False)

result = {{"cb":cb_acc,"hgb":hgb_acc,"blend":blend_acc,"features":X_arr.shape[1]}}
with open(r"{output_dir}/metrics.json","w") as f:
    json.dump(result,f)
print("LOOP_RESULT:"+json.dumps(result))
'''

def run_experiment(feature_code, task_id="spaceship_titanic", target_col="Transported"):
    exp_hash = hashlib.md5(feature_code.encode()).hexdigest()[:8]
    ts = datetime.now().strftime("%H%M%S")
    out_dir = ROOT / "workspace" / "search_loop" / f"run_{ts}_{exp_hash}"
    out_dir.mkdir(parents=True, exist_ok=True)

    script = EXEC_TEMPLATE.format(
        task_id=task_id, target_col=target_col,
        feature_code=feature_code, output_dir=str(out_dir).replace("\\","/")
    )
    script_path = out_dir / "experiment.py"
    script_path.write_text(script, encoding="utf-8")

    try:
        r = subprocess.run(["C:/codex-python/python.exe", str(script_path)],
                          capture_output=True, text=True, timeout=300)
        output = r.stdout + r.stderr
        m = re.search(r'LOOP_RESULT:(\{.*\})', output)
        if m:
            return json.loads(m.group(1)), str(out_dir)
        return {"error": "No LOOP_RESULT", "output": output[-300:]}, str(out_dir)
    except subprocess.TimeoutExpired:
        return {"error": "Timeout 300s"}, str(out_dir)
    except Exception as e:
        return {"error": str(e)}, str(out_dir)

def clean_code(raw_response):
    """Strip markdown and extract only the Python code."""
    code = raw_response
    code = re.sub(r'```python\s*', '', code)
    code = re.sub(r'```\s*', '', code)
    lines = code.strip().split('\n')
    result = []
    started = False
    for line in lines:
        if line.strip().startswith('def '):
            started = True
        if started:
            result.append(line)
    if result:
        return '\n'.join(result)
    return code.strip()

def run_search(task_id="spaceship_titanic", n_iterations=10):
    target_col = "Transported" if task_id == "spaceship_titanic" else "Survived"
    best = BEST_KNOWN.get(task_id, 0.5)
    memory = []
    improvements = 0

    print(f"DeepSeek Search Loop: {task_id}")
    print(f"Starting from: {best:.6f}, Target: {best+0.005:.6f}")
    print(f"Iterations: {n_iterations}")
    print("="*50)

    for i in range(n_iterations):
        print(f"\n--- Iteration {i+1}/{n_iterations} (best: {best:.6f}) ---")

        # Build prompt with memory
        success_stories = "\n".join(
            f"- {m['idea'][:100]} (score: {m.get('score',0):.6f})"
            for m in memory[-3:] if m.get("improved")
        ) or "None yet"
        failure_stories = "\n".join(
            f"- {m['idea'][:100]} (score: {m.get('score',0):.6f})"
            for m in memory[-3:] if not m.get("improved")
        ) or "None yet"

        prompt = f"""You optimize a Kaggle ML pipeline for {task_id} (target={target_col}).

Current best accuracy: {best:.6f}. Target: {best+0.005:.6f}.

SUCCESSFUL ideas (reuse patterns): {success_stories}

FAILED ideas (DO NOT repeat): {failure_stories}

Generate ONE Python function 'def add_features(df):' that adds 2-4 new features to the DataFrame.
Be creative but practical. Handle NaN values. Use vectorized pandas.

Output ONLY the Python function:"""

        # Generate
        print("  Asking DeepSeek...")
        response = call_deepseek(prompt)
        code = clean_code(response)
        idea_preview = code[:120].replace('\n',' ')
        print(f"  Idea: {idea_preview}")

        # Validate
        if 'def add_features' not in code:
            print("  SKIP: No valid function generated")
            memory.append({"iteration": i, "idea": idea_preview, "error": "no function", "improved": False})
            continue

        # Execute
        print("  Training...")
        result, out_dir = run_experiment(code, task_id, target_col)

        if "error" in result:
            print(f"  ERROR: {result['error'][:150]}")
            memory.append({"iteration": i, "idea": idea_preview, "error": result["error"], "improved": False})
            continue

        cb = result.get("cb", 0)
        hgb = result.get("hgb", 0)
        blend = result.get("blend", 0)
        features = result.get("features", 0)
        improved = blend > best

        if improved:
            best = blend
            improvements += 1
            marker = ">>> NEW BEST <<<"
        else:
            marker = f"(-{best-blend:.6f})"

        print(f"  CB={cb:.6f} HGB={hgb:.6f} Blend={blend:.6f} {marker} | {features} features")

        memory.append({
            "iteration": i, "idea": idea_preview, "code": code,
            "cb": cb, "hgb": hgb, "blend": blend, "features": features,
            "improved": improved, "output_dir": out_dir
        })

    print(f"\n{'='*50}")
    print(f"DONE: {improvements}/{n_iterations} improvements")
    print(f"Initial: {BEST_KNOWN.get(task_id)} -> Final: {best:.6f} ({best-BEST_KNOWN.get(task_id,0):+.6f})")

    # Save results
    report_path = ROOT / "workspace" / "search_loop" / f"report_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump({
            "task_id": task_id,
            "initial_best": BEST_KNOWN.get(task_id),
            "final_best": best,
            "improvements": improvements,
            "iterations": n_iterations,
            "memory": [{k: v for k, v in m.items() if k != "code"} for m in memory]
        }, f, indent=2, default=str)
    print(f"Report: {report_path}")
    return best

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="spaceship_titanic")
    p.add_argument("--iterations", type=int, default=8)
    args = p.parse_args()
    run_search(args.task, args.iterations)
