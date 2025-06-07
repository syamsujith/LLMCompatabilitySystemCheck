import sys
import platform
import subprocess

def check_nvidia_smi():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected:")
            print(result.stdout.splitlines()[0])  # print first line
        else:
            print("❌ No NVIDIA GPU found or nvidia-smi not installed.")
    except FileNotFoundError:
        print("❌ nvidia-smi command not found; no NVIDIA GPU or drivers installed.")

def check_ram():
    try:
        if platform.system() == "Windows":
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        elif platform.system() == "Linux" or platform.system() == "Darwin":
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        else:
            ram_gb = None

        if ram_gb:
            print(f"💾 RAM: {ram_gb:.2f} GB")
        else:
            print("💾 Could not detect RAM.")
    except ImportError:
        print("⚠️ psutil not installed, RAM info unavailable. Run: pip install psutil")

def check_python_version():
    print(f"🐍 Python version: {platform.python_version()}")

def check_library(lib_name):
    try:
        __import__(lib_name)
        print(f"📦 Library '{lib_name}' is installed.")
    except ImportError:
        print(f"❌ Library '{lib_name}' is NOT installed.")

def main():
    print("=== System Capability Check ===\n")
    check_python_version()
    check_nvidia_smi()
    check_ram()
    check_library("torch")
    check_library("transformers")

if __name__ == "__main__":
    main()
