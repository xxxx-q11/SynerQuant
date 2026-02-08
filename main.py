"""
Entry point file
"""
import argparse
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
from graph import create_agent


def run(task: str = "", github_repo_url: str = ""):
    """Run Agent"""
    agent = create_agent()
    
    initial_state = {
        "task": task,
        "github_repo_url": github_repo_url,
        "factors": [],
        "model": {},
        "strategy": {},
        "risk_report": {},
        "logs": [],
        "current_node": "start",
    }
    
    result = agent.invoke(initial_state)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantitative Trading Agent")
    parser.add_argument(
        "--task",
        type=str,
        default="Test task",
        help="Task description (optional)"
    )
    
    args = parser.parse_args()
    
    result = run(task=args.task)
    print("Execution logs:")
    for log in result.get("logs", []):
        print(f"  {log}")

