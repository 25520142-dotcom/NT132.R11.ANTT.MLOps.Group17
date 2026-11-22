import subprocess
import sys
import logging
import requests

from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s -  %(levelname)s - %(message)s")

services = {
    "api"    : "http://api:8000",
    "monitor": "http://monitor:8001"
}

def reload_services():
    versions = {}

    for name, url in services.items():
        response = requests.post(f"{url}/admin/reload-production", timeout=30)
        versions[name] = response.json()["version"]
        logging.info("%s reloaded: version %s", name, versions[name])

def retraining_job():
    logging.info("Starting periodic training...")
    subprocess.run([sys.executable, "-m", "retraining.retraining_pipeline"], check=True)
    logging.info("Periodic retraining completed.")

    reload_services()

if __name__ == "__main__":
    scheduler = BlockingScheduler()

    logging.info("Periodic retraining...")
    scheduler.add_job(
        retraining_job,
        "interval",
        max_instances=1,
        minutes=5,
        next_run_time=datetime.now()
    )
    scheduler.start()