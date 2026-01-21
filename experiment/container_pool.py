"""A global pool of containers for OFG to pull from to enable quick rebuilding of harnesses and docker container reuse"""

import logging
from dataclasses import dataclass
from multiprocessing import Manager, Queue
from typing import Optional

from experiment.benchmark import Benchmark
from experiment import oss_fuzz_checkout
from tool.container_tool import ProjectContainerTool

logger = logging.getLogger(__name__)


@dataclass
class ContainerPair:
    address_container: ProjectContainerTool
    coverage_container: ProjectContainerTool

class ContainerPool:
    """This class represents a pool of containers. Each item in the pool has one container instrumented with ASAN and one for coverage"""
    SYSTEM_CORES = 24

    def __init__(self, benchmark: Benchmark, pool_size: int):
        manager = Manager()
        self.containers = manager.Queue()
        logger.info("Initializing %d container pairs for the trial", pool_size)
        cores_per_container = max(1, int(self.SYSTEM_CORES) / pool_size)
        for _ in range(pool_size):
            self.containers.put(self.create_container_pair(benchmark, cores_per_container))

    def create_container_pair(self, benchmark: Benchmark, pool_size: int) -> ContainerPair:
        address = ProjectContainerTool(benchmark=benchmark, sanitizer="address", pool_size=pool_size)
        coverage = ProjectContainerTool(benchmark=benchmark, sanitizer="coverage", pool_size=pool_size)
        return ContainerPair(address, coverage)

    def get_container_pair(self) -> Optional[ContainerPair]:
        return self.containers.get()

    def release_container_pair(self, pair: ContainerPair):
        self.containers.put(pair)