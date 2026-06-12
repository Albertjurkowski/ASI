from kedro.pipeline import Pipeline

from asi_projekt.pipelines.data_processing import create_pipeline as dp_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    data_processing = dp_pipeline()

    pipelines: dict[str, Pipeline] = {
        "data_processing": data_processing,
        "__default__": data_processing,
    }

    try:
        from asi_projekt.pipelines.automl import create_pipeline as automl_pipeline
        automl = automl_pipeline()
        pipelines["automl"] = automl
        pipelines["full"] = data_processing + automl
    except ImportError:
        pipelines["full"] = data_processing

    return pipelines