import argparse
import subprocess
import sys
from pathlib import Path

from prefect import task, flow

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_FILES = [
    "model_pipeline.py",
    "main.py",
    "pipeline_prefect.py",
]


# ==========================================
# 2. DEFINE MACHINE LEARNING TASKS
# ==========================================


@task(name="Install Dependencies")
def install_dependencies():
    requirements = Path("requirements.txt")

    if requirements.exists():
        print("Installing dependencies...")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements),
            ],
            check=True,
        )
    else:
        raise FileNotFoundError("requirements.txt not found!")


@task(name="Prepare Data")
def prepare_data_task(data_path):
    print("Preparing data...")
    return prepare_data(data_path)


@task(name="Train Model")
def train_model_task(X_train, y_train):
    print("Training model...")
    return train_model(X_train, y_train)


@task(name="Save Model")
def save_model_task(model):
    print("Saving model...")
    save_model(model)


@task(name="Load Model")
def load_model_task():
    print("Loading model...")
    return load_model()


@task(name="Evaluate Model")
def evaluate_model_task(model, X_test, y_test):
    print("Evaluating model...")
    return evaluate_model(model, X_test, y_test)


# ==========================================
# 3. DEFINE CODE QUALITY TASKS
# ==========================================


@task(name="Format Code")
def format_code():
    print("Formatting Python code with Black...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "black",
            *PROJECT_FILES,
        ],
        check=True,
    )


@task(name="Check Code Quality")
def check_code_quality():
    print("Checking code quality with Flake8...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "flake8",
            *PROJECT_FILES,
        ],
        check=True,
    )


@task(name="Check Code Security")
def check_code_security():
    print("Checking code security with Bandit...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "bandit",
            "-q",
            "--skip",
            "B404,B603",
            *PROJECT_FILES,
        ],
        check=True,
    )


@task(name="Run Unit Tests")
def run_unit_tests():
    print("Running unit tests with Pytest...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
        ],
        check=True,
    )


# ==========================================
# 4. DEFINE CODE FLOW
# ==========================================


@flow(name="code")
def code_flow():
    format_code()
    check_code_quality()
    check_code_security()
    run_unit_tests()


# ==========================================
# 5. DEFINE MACHINE LEARNING FLOWS
# ==========================================


@flow(name="all")
def all_flow(data_path="Churn_Modelling.csv"):
    # Step 1: Install dependencies
    install_dependencies()

    # Step 2: Execute code checks
    code_flow()

    # Step 3: Prepare data
    X_train, X_test, y_train, y_test = prepare_data_task(data_path)

    # Step 4: Train model
    model = train_model_task(X_train, y_train)

    # Step 5: Save model
    save_model_task(model)

    # Step 6: Evaluate model
    evaluate_model_task(model, X_test, y_test)


@flow(name="train")
def train_flow(data_path="Churn_Modelling.csv"):
    # Step 1: Install dependencies
    install_dependencies()

    # Step 2: Prepare data
    X_train, X_test, y_train, y_test = prepare_data_task(data_path)

    # Step 3: Train model
    model = train_model_task(X_train, y_train)

    # Step 4: Save model
    save_model_task(model)


@flow(name="evaluate")
def evaluate_flow(data_path="Churn_Modelling.csv"):
    # Step 1: Prepare test data
    _, X_test, _, y_test = prepare_data_task(data_path)

    # Step 2: Load saved model
    model = load_model_task()

    # Step 3: Evaluate model
    evaluate_model_task(model, X_test, y_test)


# ==========================================
# 6. MAIN SCRIPT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefect ML Pipeline")

    parser.add_argument(
        "--flow",
        choices=["all", "train", "evaluate", "code"],
        required=True,
        help="Choose the flow to execute",
    )

    parser.add_argument(
        "--data",
        default="Churn_Modelling.csv",
        help="Path to dataset",
    )

    args = parser.parse_args()

    if args.flow == "all":
        all_flow(args.data)

    elif args.flow == "train":
        train_flow(args.data)

    elif args.flow == "evaluate":
        evaluate_flow(args.data)

    elif args.flow == "code":
        install_dependencies()
        code_flow()
