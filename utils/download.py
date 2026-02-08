"""
Utility functions module
"""
import os
import subprocess
import re
from pathlib import Path
from typing import Optional, Tuple


def extract_repo_name(github_repo_url: str) -> Optional[str]:
    """
    Extract repository name from GitHub repository URL
    
    Args:
        github_repo_url: GitHub repository address, e.g.:
            - https://github.com/username/repo.git
            - https://github.com/username/repo
            - git@github.com:username/repo.git
    
    Returns:
        Repository name, returns None if unable to extract
    """
    if not github_repo_url:
        return None
    
    # Remove .git suffix
    url = github_repo_url.rstrip('/').rstrip('.git')
    
    # Match HTTPS or SSH format
    patterns = [
        r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$',  # https://github.com/user/repo or git@github.com:user/repo.git
        r'github\.com/([^/]+)/([^/]+?)(?:\.git)?$',     # https://github.com/user/repo.git
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(2)  # Return repository name
    
    return None


def is_repo_downloaded(repo_name: str, workspace_dir: str = "workspace") -> bool:
    """
    Check if repository has been downloaded to workspace directory
    
    Args:
        repo_name: Repository name
        workspace_dir: Workspace directory path
    
    Returns:
        Returns True if repository is downloaded, otherwise False
    """
    if not repo_name:
        return False
    
    repo_path = Path(workspace_dir) / repo_name
    
    # Check if directory exists and contains .git directory (indicates it's a git repository)
    if repo_path.exists() and repo_path.is_dir():
        git_dir = repo_path / ".git"
        if git_dir.exists() and git_dir.is_dir():
            return True
    
    return False


def download_github_repo(
    github_repo_url: str,
    workspace_dir: str = "workspace"
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Download GitHub repository (if not already downloaded)
    
    Args:
        github_repo_url: GitHub repository address
        workspace_dir: Workspace directory path, defaults to "workspace"
    
    Returns:
        (success, repo_path, message) tuple
        - success: Whether successful
        - repo_path: Local repository path (if successful)
        - message: Message (success or error information)
    """
    if not github_repo_url:
        return False, None, "GitHub repository URL is empty"
    
    # Extract repository name
    repo_name = extract_repo_name(github_repo_url)
    if not repo_name:
        return False, None, f"Unable to extract repository name from URL: {github_repo_url}"
    
    # Ensure workspace directory exists
    workspace_path = Path(workspace_dir)
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    repo_path = workspace_path / repo_name
    
    # Check if already downloaded
    if is_repo_downloaded(repo_name, workspace_dir):
        return True, str(repo_path), f"Repository already exists, skipping download: {repo_path}"
    
    # Download repository
    try:
        # Use git clone to download
        cmd = ["git", "clone", github_repo_url, str(repo_path)]
        print(f"[Download] Download command: {cmd}")
        result = subprocess.run(
            cmd,
            #apture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        if repo_path.exists() and (repo_path / ".git").exists():
            return True, str(repo_path), f"Successfully downloaded repository to: {repo_path}"
        else:
            return False, None, "Download completed but unable to verify repository integrity"
            
    except subprocess.TimeoutExpired:
        return False, None, "Download timeout (exceeded 5 minutes)"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        return False, None, f"Download failed: {error_msg}"
    except FileNotFoundError:
        return False, None, "git command not found, please ensure Git is installed"
    except Exception as e:
        return False, None, f"Error occurred during download: {str(e)}"

