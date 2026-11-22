import subprocess

def when_better_deploy():
    print("\nStarting automated deployment...")

    try:
        subprocess.run(["git", "add", "artifacts/", "production_metrics.json"], check=True)

        result = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if(result.returncode == 0):
            print("No artifact changes detected.")
            return
        
        subprocess.run(["git", "commit", "-m", "Auto retrained production model"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)

        print("\nGithub updated successfully.")
        print("GitHub Actions pipeline has been triggered.")

    except subprocess.CalledProcessError as e:
        print(f"Deployment failed. {e}")
