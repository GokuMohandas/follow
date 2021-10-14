# airflow/dags/example.py
from airflow.decorators import dag
from airflow.operators.bash_operator import BashOperator


# Default DAG args
default_args = {
    "owner": "airflow",
}


@dag(
    dag_id="example",
    description="Example DAG",
    default_args=default_args,
    tags=["example"],
)
def example():
    # Define tasks
    task_1 = BashOperator(task_id="task_1", bash_command="echo 1")
    task_2 = BashOperator(task_id="task_2", bash_command="echo 2")

    # Task relationships
    task_1 >> task_2


# Define DAG
example_dag = example()
