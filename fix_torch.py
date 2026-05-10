"""
Fix PyTorch DLL loading issue on Microsoft Store Python.
Must be imported BEFORE 'import torch'
"""

import os
import sys
import glob


def fix_dll_paths():
    """Add torch DLL directories to search path"""
    
    # Find torch installation
    possible_paths = []
    
    for site_dir in sys.path:
        torch_lib = os.path.join(site_dir, "torch", "lib")
        if os.path.exists(torch_lib):
            possible_paths.append(torch_lib)
        
        # Also check for CUDA DLLs
        nvidia_path = os.path.join(site_dir, "nvidia")
        if os.path.exists(nvidia_path):
            for root, dirs, files in os.walk(nvidia_path):
                if "bin" in dirs:
                    possible_paths.append(os.path.join(root, "bin"))
                if any(f.endswith(".dll") for f in files):
                    possible_paths.append(root)
    
    # Also add common CUDA paths
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\Program Files\NVIDIA Corporation\NVSMI",
    ]
    
    for p in cuda_paths:
        if os.path.exists(p):
            possible_paths.append(p)
    
    # Add all found paths
    for path in possible_paths:
        if os.path.exists(path):
            # Method 1: os.add_dll_directory (Python 3.8+)
            try:
                os.add_dll_directory(path)
            except (OSError, AttributeError):
                pass
            
            # Method 2: Add to PATH
            if path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = path + ";" + os.environ.get("PATH", "")
    
    # Method 3: Set specific env variable for torch
    for site_dir in sys.path:
        torch_dir = os.path.join(site_dir, "torch")
        if os.path.exists(torch_dir):
            os.environ["TORCH_HOME"] = torch_dir
            break
    
    print(f"[DLL Fix] Added {len(possible_paths)} DLL paths")


# Auto-run on import
fix_dll_paths()