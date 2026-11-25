
from prefect import serve

from pipeline_prefect import all_flow


if __name__ == "__main__":
    all_flow.serve(
        name="customer-churn-daily-deployment",
        cron="0 9 * * *",
        parameters={
            "data_path": "Churn_Modelling.csv"
        },
        tags=["machine-learning", "daily"],
    )

