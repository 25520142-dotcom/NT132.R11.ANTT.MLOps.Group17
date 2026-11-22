# NT132.R11.ANTT.MLOps.Group17

WIP

this is only used for study purpose. 80% of the structure was from https://github.com/arpanneupane75/churn-prediction-mlops/tree/main all credit to it without it this wouldnt be possible (the other 18% is AI, 1.9% zinc and 0.1% me)

requirement: installed docker, python 3.14.8

how to use:
1. clone this
2. make virtual env
3. pip install --upgrade pip | pip install -r requirements.txt (Install requiremnt)
4. run python -m retraining.retraining_pipeline (Run pipeline. Currently theres bug that docker compose doesnt work because api is mounting on mlflow.db rn)
5. open docker
6. docker compose up --build -d (takes almost 10gb each api is around 2gb)
7. python generator.py --train data\train_data.csv --test data\test_data.csv --n 500 (This was made by AI, just to shorten the time to check the drift)
8. monitor health with localhost:9090 (prometheus), monitor drift and other stuffs with localhost:3000 (grafana)
9. in the future: connect minIO objects and PostgreSQL for loggings runs, metrics, etc 
