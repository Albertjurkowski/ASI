from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = find_pipelines()
    return {
        **pipelines,
        "__default__": pipelines["data_processing"],
        "full": sum(pipelines.values()),
    }