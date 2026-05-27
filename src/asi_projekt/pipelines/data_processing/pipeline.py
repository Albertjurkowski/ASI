from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_and_log, preprocess, split_data, train_model

preprocess_node = node(
    func=preprocess,
    inputs=["raw_data", "parameters"],
    outputs="processed_data",
    name="preprocess_node",
)
split_data_node = node(
    func=split_data,
    inputs=["processed_data", "parameters"],
    outputs=["X_train", "X_val", "X_test",
             "y_train", "y_val", "y_test"],
    name="split_data_node",
)
train_model_node = node(
    func=train_model,
    inputs=["X_train", "y_train", "parameters"],
    outputs="trained_model",
    name="train_model_node",
)
evaluate_and_log_node = node(
    func=evaluate_and_log,
    inputs=["trained_model", "X_val", "y_val", "parameters"],
    outputs="metrics",
    name="evaluate_and_log_node",
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        preprocess_node,
        split_data_node,
        train_model_node,
        evaluate_and_log_node,
    ])
