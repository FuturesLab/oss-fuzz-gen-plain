"""A global pool of containers for OFG to pull from to enable quick rebuilding of harnesses and docker container reuse"""

import logging
from dataclasses import dataclass
from multiprocessing import Manager, Queue
from typing import Optional

from project.benchmark import Benchmark
from utils import oss_fuzz_checkout
from execution.container_tool import ProjectContainerTool

logger = logging.getLogger(__name__)


@dataclass
class ContainerPair:
    address_container: ProjectContainerTool
    coverage_container: ProjectContainerTool

class ContainerPool:
    """This class represents a pool of containers. Each item in the pool has one container instrumented with ASAN and one for coverage"""

    def __init__(self, benchmark: Benchmark, num_cores: int):
        manager = Manager()
        self.containers = manager.Queue()
        logger.info("Initializing %d container pairs for the trial", num_cores)
        for _ in range(num_cores):
            self.containers.put(self.create_container_pair(benchmark))

    def create_container_pair(self, benchmark: Benchmark) -> ContainerPair:
        address = ProjectContainerTool(benchmark=benchmark)

        oss_fuzz_checkout.ENABLE_CACHING = False
        coverage = ProjectContainerTool(benchmark=benchmark)
        oss_fuzz_checkout.ENABLE_CACHING = True

        return ContainerPair(address, coverage)

    def get_container_pair(self) -> Optional[ContainerPair]:
        return self.containers.get()

    def release_container_pair(self, pair: ContainerPair):
        self.containers.put(pair)
    