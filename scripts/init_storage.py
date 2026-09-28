import os
import json

def initialize_workspace_storage():
    """
    Creates local storage structure for MindGate AI workspaces.
    Ensures 100% offline isolation without leaking files.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    storage_dir = os.path.join(base_dir, "storage")
    mindgate_hidden_dir = os.path.join(storage_dir, ".mindgate")
    
    # Create directories if they don't exist
    os.makedirs(storage_dir, exist_ok=True)
    os.makedirs(mindgate_hidden_dir, exist_ok=True)
    
    # Create initial memory.json if missing
    memory_file = os.path.join(mindgate_hidden_dir, "memory.json")
    if not os.path.exists(memory_file):
        default_memory = {
            "container_id": "workspace_001",
            "documents": [],
            "memory_highlights": []
        }
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(default_memory, f, indent=2)
            
    # Create .gitkeep to preserve folder structure
    gitkeep_file = os.path.join(storage_dir, ".gitkeep")
    if not os.path.exists(gitkeep_file):
        with open(gitkeep_file, "w") as f:
            f.write("")

    print(f"[Storage Initialized]: Local workspace path ready at {storage_dir}")

if __name__ == "__main__":
    initialize_workspace_storage()