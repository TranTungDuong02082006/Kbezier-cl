import os
import shutil

src_dir = r"d:\AI_Projects\Kbezier-cl\Kbezier_fixed"
dst_dir = r"d:\AI_Projects\Kbezier-cl"

def copy_recursive(src, dst):
    for name in os.listdir(src):
        # Exclude metadata, git, and workspace results
        if name in [".git", ".gemini", ".pytest_cache", "Kbezier_fixed", "results", "kbezier_cl.egg-info", "__pycache__", "scratch"]:
            continue
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
        if os.path.isdir(src_path):
            os.makedirs(dst_path, exist_ok=True)
            copy_recursive(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {name}")

if __name__ == "__main__":
    print(f"Starting sync from {src_dir} to {dst_dir}...")
    copy_recursive(src_dir, dst_dir)
    print("Sync completed successfully.")
