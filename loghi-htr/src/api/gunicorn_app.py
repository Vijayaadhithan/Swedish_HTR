# Imports

# > Standard library
import os

# > Third-party dependencies
from gunicorn.app.base import BaseApplication

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GunicornApp(BaseApplication):
    """
    Custom Gunicorn application class.

    This class allows for the configuration and running of a Flask application
    using Gunicorn with custom settings.

    Attributes
    ----------
    options : dict, optional
        Gunicorn configuration options.
    application : Flask application instance
        The Flask application to be run with Gunicorn.
    """

    def __init__(self, app, options: dict = None):
        """
        Initialize the GunicornApp with a Flask application and options.

        Parameters
        ----------
        app : Flask application instance
            The Flask application to be run with Gunicorn.
        options : dict, optional
            Gunicorn configuration options. Default is an empty dictionary.
        """

        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        """
        Load Gunicorn configuration from the provided options.

        Side Effects
        ------------
        - Updates the internal Gunicorn configuration based on `self.options`.
        """

        for key, value in self.options.items():
            self.cfg.set(key, value)

    def load(self):
        """
        Load the Flask application for Gunicorn.

        Returns
        -------
        Flask application instance
            The Flask application to be run with Gunicorn.
        """

        return self.application


if __name__ == "__main__":
    # > Standard library
    # Import here to set TF_CPP_MIN_LOG_LEVEL before importing TensorFlow
    from flask_app import create_app
    from app_utils import get_env_variable, setup_logging, start_processes

    # Set up logging
    logging_level = get_env_variable("LOGGING_LEVEL", "INFO")
    logger = setup_logging(logging_level)

    # Get Gunicorn options from environment variables
    logger.info("Getting Gunicorn options from environment variables")
    bind = get_env_variable("GUNICORN_RUN_HOST", "127.0.0.1:8000")
    workers = int(get_env_variable("GUNICORN_WORKERS", "1"))
    threads = int(get_env_variable("GUNICORN_THREADS", "1"))
    accesslog = get_env_variable("GUNICORN_ACCESSLOG", "-")

    # Get Loghi-HTR options from environment variables
    logger.info("Getting Loghi-HTR options from environment variables")
    batch_size = int(get_env_variable("LOGHI_BATCH_SIZE", "256"))
    model_path = get_env_variable("LOGHI_MODEL_PATH")
    output_path = get_env_variable("LOGHI_OUTPUT_PATH")
    max_queue_size = int(get_env_variable("LOGHI_MAX_QUEUE_SIZE", "10000"))

    # Get GPU options from environment variables
    logger.info("Getting GPU options from environment variables")
    gpus = get_env_variable("LOGHI_GPUS", "0")

    # Start the processing and prediction processes
    logger.info("Starting processing and prediction processes")
    request_queue, preparation_process, prediction_process = start_processes(
        batch_size,
        max_queue_size,
        output_path,
        gpus,
        model_path
    )

    options = {
        'bind': bind,
        'workers': workers,
        'threads': threads,
        'accesslog': accesslog,
        'worker_class': 'sync',
    }

    logger.info(f"Starting Gunicorn with options: {options}")
    gunicorn_app = GunicornApp(
        create_app(request_queue),
        options
    )

    gunicorn_app.run()
