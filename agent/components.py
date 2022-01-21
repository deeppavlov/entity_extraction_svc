import logging
from os import getenv
from typing import List, Any, Dict

import requests
from deeppavlov.core.models.component import Component


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class ApiRequester(Component):
    """Component for forwarding parameters to APIs

    Args:
        url: url of the API.
        out: count of expected returned values or their names in a chainer.
        param_names: list of parameter names for API requests.

    Attributes:
        url: url of the API.
        out: count of expected returned values.
        param_names: list of parameter names for API requests.
    """

    def __init__(self, url: str, out: [int, list], param_names: [list, tuple] = (), *args, **kwargs):
        if url.endswith('_URL'):
            url = getenv(url)
        self.url = url
        self.param_names = param_names
        self.out_count = out if isinstance(out, int) else len(out)

    def __call__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        """

        Args:
            *args: list of parameters sent to the API endpoint. Parameter names are taken from self.param_names.
            **kwargs: named parameters to send to the API endpoint. If not empty, args are ignored

        Returns:
            result of the API request(s)
        """
        data = kwargs or dict(zip(self.param_names, args))

        logger.info(f"Sending to {self.url}: {data}")
        response_batch = requests.post(self.url, json=data).json()
        logger.info(f"Response {response_batch}")

        response = response_batch[0]
        return response


class OutputFormatter(Component):
    """Component for formatting final pipeline output
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        """

        Args:
            *args:
            **kwargs:

        Returns:
            formatted wikidata output
        """
        print(args)
        print(kwargs)
        return args
