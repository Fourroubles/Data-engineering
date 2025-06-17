from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False
}

with DAG(
    'breast_cancer_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for breast cancer classification',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    extract_task = PythonOperator(
        task_id='extract_task',
        python_callable=lambda: exec(open('etl/extract.py').read()),
    )

    transform_task = PythonOperator(
        task_id='transform_task',
        python_callable=lambda: exec(open('etl/transform.py').read()),
    )

    train_task = PythonOperator(
        task_id='train_task',
        python_callable=lambda: exec(open('etl/train.py').read()),
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_task',
        python_callable=lambda: exec(open('etl/evaluate.py').read()),
    )

    save_local = PythonOperator(
    task_id="save_local",
    python_callable=lambda: exec(open('etl/save_local.py').read()),
    dag=dag
)

    extract_task >> transform_task >> train_task >> evaluate_task >> save_local