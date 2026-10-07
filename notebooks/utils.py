import os


def is_serverless() -> bool:
    """Check whether the active compute is Databricks serverless.

    :return: ``True`` if running on serverless, ``False`` otherwise.
    """
    return os.environ.get("DATABRICKS_RUNTIME_VERSION", "").startswith("client.")
