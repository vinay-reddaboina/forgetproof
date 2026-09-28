"""
One-shot migration helper: pushes the locally-saved target_model to Kaggle as a
private Dataset, then publishes an adapted copy of the unlearn+audit notebook
as a Kaggle Notebook (kernel) with GPU + internet enabled, pointing at that
dataset.

Run from inside Colab (which has Drive mounted) after ~/.kaggle/kaggle.json
has been written from the KAGGLE_USERNAME / KAGGLE_KEY secrets:

    !python scripts/push_to_kaggle.py

Everything here is flat, single-purpose, non-interactive shell-outs to the
`kaggle` CLI -- kept out of the Colab cell editor entirely so there's no risk
of the notebook UI mangling indented Python while it's typed in.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, **kw):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)


def kaggle_username():
    kj = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kj):
        sys.exit(f"ERROR: {kj} not found -- write your Kaggle API creds there first.")
    with open(kj) as f:
        return json.load(f)["username"]


def push_dataset(target_model_dir, slug, username, staging_root):
    print(f"\n=== Staging + pushing dataset {username}/{slug} ===")
    stage = os.path.join(staging_root, "dataset_stage")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    shutil.copytree(target_model_dir, stage)

    meta = {
        "title": "ForgetProof target model",
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(stage, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Does it already exist? `kaggle datasets status` exits non-zero if not.
    exists = subprocess.run(
        ["kaggle", "datasets", "status", f"{username}/{slug}"],
        capture_output=True,
    ).returncode == 0

    if exists:
        sh(["kaggle", "datasets", "version", "-p", stage, "-m", "update target_model", "--dir-mode", "zip"])
    else:
        sh(["kaggle", "datasets", "create", "-p", stage, "--dir-mode", "zip"])

    print(f"Dataset ready: https://www.kaggle.com/datasets/{username}/{slug}")
    return f"{username}/{slug}"


def push_kernel(notebook_path, kernel_slug, username, dataset_ref, staging_root):
    print(f"\n=== Staging + pushing kernel {username}/{kernel_slug} ===")
    stage = os.path.join(staging_root, "kernel_stage")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)

    nb_filename = os.path.basename(notebook_path)
    shutil.copy(notebook_path, os.path.join(stage, nb_filename))

    meta = {
        "id": f"{username}/{kernel_slug}",
        "title": "ForgetProof Unlearn + Audit (Kaggle)",
        "code_file": nb_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(stage, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    sh(["kaggle", "kernels", "push", "-p", stage])
    print(f"Kernel ready: https://www.kaggle.com/code/{username}/{kernel_slug}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model-dir", default="/content/drive/MyDrive/forgetproof/target_model")
    ap.add_argument("--dataset-slug", default="forgetproof-target-model")
    ap.add_argument("--kernel-slug", default="forgetproof-unlearn-audit")
    ap.add_argument("--notebook", default=os.path.join(REPO_DIR, "notebooks", "03_unlearn_and_audit_kaggle.ipynb"))
    ap.add_argument("--staging-root", default="/content/kaggle_push_staging")
    args = ap.parse_args()

    if not os.path.isdir(args.target_model_dir):
        sys.exit(f"ERROR: target model dir not found: {args.target_model_dir}")

    os.makedirs(args.staging_root, exist_ok=True)
    username = kaggle_username()

    dataset_ref = push_dataset(args.target_model_dir, args.dataset_slug, username, args.staging_root)
    push_kernel(args.notebook, args.kernel_slug, username, dataset_ref, args.staging_root)

    print("\nDone. Open the kernel URL above, hit 'Edit', then 'Run All' -- "
          "GPU and internet are already enabled and the dataset is attached "
          "at /kaggle/input/" + args.dataset_slug)


if __name__ == "__main__":
    main()
