import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(
        description=("Customer Churn Prediction " "Pipeline")
    )

    parser.add_argument(
        "--data",
        default="Churn_Modelling.csv",
        help="Path to the dataset",
    )

    args = parser.parse_args()

    # Step 1: Prepare data
    print("\n1. Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    # Step 2: Train model
    print("\n2. Training model...")
    model = train_model(X_train, y_train)

    # Step 3: Evaluate model
    print("\n3. Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # Step 4: Save model
    print("\n4. Saving model...")
    save_model(model)

    # Step 5: Load model
    print("\n5. Loading model...")
    load_model()

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
