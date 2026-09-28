"""
Builds notebooks/03_unlearn_and_audit_kaggle.ipynb from notebooks/02_unlearn_and_audit.ipynb.

Changes vs the Colab version:
  - Cell 1 (setup): no Drive mount; clones the repo (Kaggle sessions have
    internet enabled per kernel-metadata.json) instead.
  - Cell 2: TARGET_DIR points at the attached Kaggle Dataset input path
    instead of a Drive path.
  - Cell 5 (main loop): checkpoints to /kaggle/working instead of Drive.
    /kaggle/working survives "Save Version" (commit); it does NOT survive an
    un-committed session simply closing, so KEEP_FOR_DIAGNOSTICS models are
    still checkpointed the same way as before -- just to local disk here,
    since Kaggle GPU sessions run interactively for hours without Colab's
    idle-kick behavior, and results also travel out via git at the end.
  - Cell 9 (final push): tries a Kaggle Secret named GITHUB_TOKEN for the
    push; if it's not configured, prints the results paths instead of
    failing the cell.

Run once from the repo root: `python3 scripts/build_kaggle_notebook.py`
"""
import json
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_DIR, "notebooks", "02_unlearn_and_audit.ipynb")
DST = os.path.join(REPO_DIR, "notebooks", "03_unlearn_and_audit_kaggle.ipynb")


def src_lines(text):
    return text.splitlines(keepends=True)


def main():
    nb = json.load(open(SRC))
    cells = nb["cells"]

    orig_cell1 = "".join(cells[1]["source"])
    orig_cell2 = "".join(cells[2]["source"])
    orig_cell5 = "".join(cells[5]["source"])
    orig_cell9 = "".join(cells[9]["source"])

    # --- Cell 1: setup, Kaggle flavor -----------------------------------
    expected_old_cell1_head = "import os\n\n# Absolute path"
    assert orig_cell1.startswith(expected_old_cell1_head), "cell 1 source drifted, update this script"

    new_cell1 = """import os, sys
# Kaggle sessions start fresh each time (no persistent VM disk to check
# for an existing clone like Colab's), so this is a plain clone -- no
# idempotency branch needed. enable_internet is set in kernel-metadata.json.
REPO_DIR = '/kaggle/working/forgetproof'
if not os.path.isdir(REPO_DIR):
    !git clone https://github.com/vinay-reddaboina/forgetproof.git {REPO_DIR}
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
!pip install -q -r requirements.txt
print("repo ready at", REPO_DIR)
"""

    # --- Cell 2: swap TARGET_DIR to the attached dataset -----------------
    old_target_line = 'TARGET_DIR = "/content/drive/MyDrive/forgetproof/target_model"'
    assert old_target_line in orig_cell2, "cell 2 TARGET_DIR line drifted, update this script"
    new_target_line = 'TARGET_DIR = "/kaggle/input/forgetproof-target-model"  # attached Kaggle Dataset'
    new_cell2 = orig_cell2.replace(old_target_line, new_target_line)
    new_cell2 = new_cell2.replace(
        'print("loaded target model + data (base_model kept on CPU -- see comment above)")',
        'print("loaded target model + data (base_model kept on CPU -- see comment above)")\n'
        'print("device:", device, "-- should be cuda; if not, GPU wasn\'t actually attached to this session")',
    )

    # --- Cell 5: Drive checkpoint dirs -> Kaggle working dir -------------
    old_repo_dir_line = "REPO_DIR = '/content/forgetproof'"
    old_drive_dir = 'DRIVE_RESULTS_DIR = "/content/drive/MyDrive/forgetproof/results_checkpoints"'
    old_ckpt_dir = 'CKPT_DIR = "/content/drive/MyDrive/forgetproof/checkpoints"'
    assert old_repo_dir_line in orig_cell5, "cell 5 REPO_DIR line drifted, update this script"
    assert old_drive_dir in orig_cell5 and old_ckpt_dir in orig_cell5, "cell 5 checkpoint dirs drifted, update this script"
    new_cell5 = orig_cell5.replace(
        old_repo_dir_line,
        "REPO_DIR = '/kaggle/working/forgetproof'  # matches cell 1's clone target on Kaggle",
    ).replace(
        old_drive_dir,
        '# /kaggle/working persists across cells in this session and is saved whenever\n'
        '# you hit "Save Version" -- the Kaggle equivalent of Drive surviving a disconnect.\n'
        'DRIVE_RESULTS_DIR = "/kaggle/working/results_checkpoints"',
    ).replace(
        old_ckpt_dir,
        'CKPT_DIR = "/kaggle/working/checkpoints"',
    )

    # --- Cell 9: best-effort push using a Kaggle Secret ------------------
    assert "!git push" in orig_cell9, "cell 9 drifted, update this script"
    new_cell9 = '''# Push results json + diagnostics back to the repo, if a GITHUB_TOKEN Kaggle
# Secret is configured (Add-ons -> Secrets in the Kaggle notebook editor).
# If not, results still sit in /kaggle/working and are saved to Kaggle's own
# output the moment you hit "Save Version" -- download them from there.
import subprocess
have_token = False
try:
    from kaggle_secrets import UserSecretsClient
    github_token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    have_token = bool(github_token)
except Exception as e:
    print("no GITHUB_TOKEN Kaggle Secret configured:", e)

if have_token:
    push_url = f"https://{github_token}@github.com/vinay-reddaboina/forgetproof.git"
    subprocess.run(["git", "add", "results/"] + __import__("glob").glob("results/*.json") + __import__("glob").glob("results/*.png"), cwd=REPO_DIR)
    subprocess.run(["git", "commit", "-m", "Add audit results + PCA diagnostics for all 8 methods"], cwd=REPO_DIR)
    subprocess.run(["git", "push", push_url, "HEAD:main"], cwd=REPO_DIR)
else:
    print("Skipping git push -- results are in:", os.path.join(REPO_DIR, "results"))
    print("Hit 'Save Version' on this Kaggle notebook to keep them as this notebook's output,")
    print("or add a GITHUB_TOKEN Kaggle Secret (repo scope) and rerun this cell.")
'''

    replacements = {1: new_cell1, 2: new_cell2, 5: new_cell5, 9: new_cell9}
    for idx, new_src in replacements.items():
        cells[idx]["source"] = src_lines(new_src)
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None

    # clear stale outputs/execution counts on every cell for a clean notebook
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # retitle the markdown intro cell so it's obviously the Kaggle variant
    md = "".join(cells[0]["source"])
    cells[0]["source"] = src_lines(
        md.replace(
            "# ForgetProof — 02: Unlearn + audit",
            "# ForgetProof — 02: Unlearn + audit (Kaggle variant)\n\n"
            "Adapted from `02_unlearn_and_audit.ipynb` to run on Kaggle instead of Colab: "
            "loads `target_model` from an attached Kaggle Dataset instead of Google Drive, "
            "and checkpoints to `/kaggle/working` instead of Drive. See that notebook for "
            "the full method/attack writeup.",
        )
    )

    json.dump(nb, open(DST, "w"), indent=1)
    print("wrote", DST)


if __name__ == "__main__":
    main()
