import os
import shutil
from pathlib import Path
from typing import List, Union, Optional


class WorkspaceManager:
    """
    Workspace aur directory structure ko manage karne ke liye utility class.
    """

    def __init__(self, base_dir: Union[str, Path] = "workspace"):
        """
        Workspace initialization.
        :param base_dir: Root directory path for workspace.
        """
        self.base_dir = Path(base_dir).resolve()
        self.ensure_workspace_exists()

    def ensure_workspace_exists(self) -> Path:
        """
        Root workspace directory check karta hai aur zaroorat padne par create karta hai.
        """
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)
            print(f"[+] Root Workspace created at: {self.base_dir}")
        return self.base_dir

    def create_project_structure(self, project_name: str, subfolders: Optional[List[str]] = None) -> Path:
        """
        Ek naya project folder create karta hai default ya custom sub-folders ke saath.
        """
        project_path = self.base_dir / project_name
        project_path.mkdir(parents=True, exist_ok=True)

        if subfolders:
            for folder in subfolders:
                folder_path = project_path / folder
                folder_path.mkdir(parents=True, exist_ok=True)

        print(f"[+] Project structure initialized: {project_path}")
        return project_path

    def list_contents(self, subpath: str = "") -> List[str]:
        """
        Workspace ke andar ki files aur folders list karta hai.
        """
        target_path = self.base_dir / subpath
        if not target_path.exists():
            print(f"[-] Path does not exist: {target_path}")
            return []
        
        return [item.name for item in target_path.iterdir()]

    def clean_workspace(self, project_name: Optional[str] = None) -> None:
        """
        Ek specific project folder ko delete karta hai, ya pure workspace ko clean karta hai.
        """
        target_path = self.base_dir / project_name if project_name else self.base_dir

        if target_path.exists():
            shutil.rmtree(target_path)
            print(f"[!] Cleaned up: {target_path}")
            if project_name is None:
                self.ensure_workspace_exists()
        else:
            print(f"[-] Nothing to clean at: {target_path}")


# Basic usage example
if __name__ == "__main__":
    # Workspace instance creation
    ws = WorkspaceManager(base_dir="my_workspace")

    # App ya project structural layout Setup
    project_dirs = ["src", "data", "logs", "build"]
    app_path = ws.create_project_structure("my_app", subfolders=project_dirs)

    # Contents print karna
    print("Workspace Items:", ws.list_contents())
    print("Project Items:", ws.list_contents("my_app"))