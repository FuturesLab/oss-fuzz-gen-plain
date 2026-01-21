# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A tool for LLM agents to interact within a project's docker container."""
import logging
import os
import re
import subprocess as sp
import time
from typing import Optional

from experiment.benchmark import Benchmark
from experiment.workdir import WorkDirs
from experiment import oss_fuzz_checkout
from tool.base_tool import BaseTool

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
"""
Workflow: 
Initialization:
- Copy oss-fuzz dir on host machine into specially named dir. The container will reuse this throughout its lifetime
- self.generated_project_path holds this local path

Execution Stage:
- Write fuzz driver to container (similar to OnePromptPrototyper)
- Write build script to container if necessary 
- Run the driver with infra/helper.py
"""


class ProjectContainerTool(BaseTool):
    """A tool for LLM agents to interact within a project's docker container."""
    SANITIZERS = ["coverage", "address"]

    def __init__(
        self, benchmark: Benchmark, sanitizer: str, name: str = "", project_name: str = "", pool_size: int = 4
    ) -> None:
        super().__init__(benchmark, name)
        
        self.benchmark = benchmark
        self.sanitizer = sanitizer
        self.project_name = project_name or benchmark.project
        self.pool_size = pool_size
        
        self._validate_sanitizer()
        self.image_name = self._prepare_project_image(self.project_name, self.sanitizer)
        self.rebuild_chronos_success = self._check_chronos_build_success()
        
        self.generated_oss_fuzz_name = self._get_project_name()
        self.generated_project_path = self._get_project_path()
        
        self.vmap_outdir = get_build_artifact_dir(self.generated_oss_fuzz_name, "out")
        self.vmap_workdir = get_build_artifact_dir(self.generated_oss_fuzz_name, "work")
        self.vmap_ccache = get_ccache_dir(self.generated_oss_fuzz_name)
        
        self.container_id = self._start_docker_container()
        self.build_script_path = "/src/build.sh"
        self._backup_default_build_script()
        self.project_dir = self._get_project_dir()
        
        if not self.rebuild_chronos_success:
            self._setup_container_env()
        
        self.patch_compile_script()

        self.avg_cov_runtime = -1
        self.total_cov_executions = 0

        logger.info(
            "Generated image %s -- container -- %s linked to directory %s",
            self.image_name,
            self.container_id,
            self.generated_project_path,
        )

    def _validate_sanitizer(self):
        if self.sanitizer not in self.SANITIZERS:
            raise ValueError(
                "Supplied sanitizer is invalid. Please provide 'address' or 'coverage'"
            )

    def _check_chronos_build_success(self) -> bool:
        success = "ofg-cached" in self.image_name
        if success:
            logger.info("Successfully built image %s with rebuild chronos", self.image_name)
        return success

    def _get_project_name(self) -> str:
        # Strip out -ofg-cached-* tags
        return re.sub(r"-ofg-cached-.*$", "", os.path.basename(self.image_name))

    def _get_project_path(self) -> str:
        return os.path.join(
            oss_fuzz_checkout.OSS_FUZZ_DIR, "projects", self.generated_oss_fuzz_name
        )

    def patch_compile_script(self) -> str:
        """
        Copying the source directory on every recompilation can be extremely time+resource consuming. 
        Don't re-copy source directory on every build, treat it as a one time cost instead, which is taken care of in _prepare_project_image.
        I noticed cases where the rebuild itself would take ~7 seconds, while copying would take ~40 seconds when /src was > 1G
        """
        compile_path = os.path.join("/usr", "local", "bin", "compile")
        proc =  self.execute(f"cat {compile_path}")
        compile_contents = proc.stdout
        original_cmd = 'COPY_SOURCES_CMD="cp -rL --parents $SRC $WORK /usr/include /usr/local/include $GOPATH $OSSFUZZ_RUSTPATH /rustc $OUT"'
        new_cmd = 'COPY_SOURCES_CMD="cp -rL --parents $WORK /usr/include /usr/local/include $GOPATH $OSSFUZZ_RUSTPATH /rustc $OUT"'
        new_contents = compile_contents.replace(original_cmd, new_cmd)

        self.write_to_file(new_contents, compile_path)

    def tutorial(self) -> str:
        """Constructs a tool guide tutorial for LLM agents."""
        return self._get_tutorial_file_content("container_tool.txt").replace(
            "{FUZZ_TARGET_PATH}", self.benchmark.target_path
        )

    def _setup_container_env(self):
        """Alias mkdir to mkdir -p so we can reuse build artifacts"""
        command = """cat > /etc/profile.d/mkdir.sh <<'EOF'
mkdir() { command mkdir -p "$@"; }
export -f mkdir
EOF"""
        self.execute(command)

    def _prepare_project_image(self, project_name: str, sanitizer: str) -> str:
        """Prepares the project's OSS-Fuzz docker image and returns the image name."""
        image_name = oss_fuzz_checkout.prepare_project_image_by_name_w_rebuild(project_name, sanitizer)
        if image_name:
            return image_name
        raise RuntimeError(f"Failed to build image for {project_name}")

    def _execute_command_in_container(
        self, command: list[str], log_path: Optional[str] = None
    ) -> sp.CompletedProcess:
        """Executes the |command| in subprocess and log output."""
        log_file = sp.PIPE
        if log_path is not None:
            log_file = open(log_path, "w+")
        try:
            result = sp.run(
                command,
                stdout=sp.PIPE,
                stderr=log_file,
                check=False,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            if log_path is not None:
                output = f"Logged in {log_path}"
            else:
                output = result.stdout
            logger.debug(
                "Executing command (%s) in container %s: Return code %d. STDOUT: %s, "
                "STDERR: %s",
                command,
                self.container_id,
                result.returncode,
                output,
                result.stderr,
            )
            return result
        except Exception as e:
            logger.error(
                "Executing command (%s) in container failed with Exception: %s",
                command,
                e,
            )
            return sp.CompletedProcess(command, returncode=1, stdout="", stderr="")
        finally:
            if log_path is not None:
                log_file.close()

    def _execute_command(self, command: list[str]) -> sp.CompletedProcess:
        """Executes the |command| in subprocess and log output."""
        try:
            result = sp.run(
                command,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                check=False,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )

            logger.debug(
                "Executing command (%s): Return code %d. STDOUT: %s, STDERR: %s",
                command,
                result.returncode,
                result.stdout,
                result.stderr,
            )
            return result
        except Exception as e:
            logger.error("Executing command (%s) failed with Exception: %s", command, e)
            return sp.CompletedProcess(command, returncode=1, stdout="", stderr="")

    def _backup_default_build_script(self) -> None:
        """Creates a copy of the human-written /src/build.sh for LLM to use."""
        backup_command = f"cp {self.build_script_path} /src/build.bk.sh"
        process = self.execute(backup_command)
        if process.returncode:
            logger.error(
                "Failed to create a backup of %s: %s",
                self.build_script_path,
                self.image_name,
            )

    def _get_project_dir(self) -> str:
        """Returns the project-under-test's source code directory."""
        pwd_command = "pwd"
        process = self.execute(pwd_command)
        if process.returncode:
            logger.error("Failed to get the WORKDIR: %s", self.image_name)
            return ""
        return process.stdout.strip()

    def _start_docker_container(self) -> str:
        """Runs the project's OSS-Fuzz image as a background container and returns
        the container ID."""
        command = [
            "docker",
            "run",
            "-d",
            "--privileged",
            "--shm-size=2g",
            "--platform",
            "linux/amd64",
            f"--cpus={self.pool_size}",
            "-t",
            "-e",
            "FUZZING_ENGINE=libfuzzer",
            "-e",
            "ARCHITECTURE=x86_64",
            "-e",
            f"PROJECT_NAME={self.generated_oss_fuzz_name}",
            "-e",
            f"FUZZING_LANGUAGE={self.benchmark.language}",
            "-e",
            "CCACHE_DIR=/workspace/ccache",
            "-v",
            f"{self.vmap_outdir}:/out",
            "-v",
            f"{self.vmap_workdir}:/work",
            "-v",
            f"{self.vmap_ccache}:/workspace/ccache",
            "--entrypoint=/bin/bash",
            f"{self.image_name}",
        ]
        os.makedirs(self.vmap_outdir, exist_ok=True)
        os.makedirs(self.vmap_workdir, exist_ok=True)
        result = self._execute_command(command)
        if result.returncode:
            logger.error("Failed to start container of image: %s", self.image_name)
        container_id = result.stdout.strip()
        return container_id

    def execute(
        self, command: str, log_path: Optional[str] = None
    ) -> sp.CompletedProcess:
        """Executes the |command| in the container and returns the output."""
        logger.debug("Executing command (%s) in %s: ", command, self.container_id)
        execute_command_in_container = [
            "docker",
            "exec",
            self.container_id,
            "/bin/bash",
            "-c",
            command,
        ]
        process = self._execute_command_in_container(
            execute_command_in_container, log_path
        )
        process.args = command
        return process

    def compile(
        self,
        extra_commands: str = "",
        log_path: Optional[str] = None,
    ) -> sp.CompletedProcess:
        """Compiles the fuzz target."""
        command = "compile" + extra_commands
        command = f"SANITIZER={self.sanitizer} " + command
        if not self.rebuild_chronos_success:
            mkdir_alias = "source /etc/profile.d/mkdir.sh; "
            command = mkdir_alias + command

        begin_time = time.time()
        compile_process = self.execute(command, log_path)
        end_time = time.time()
        # Hide Compilation command so that LLM won't reuse it in the inspection tool
        # and be distracted by irrelevant errors, e.g., `build/ already exits`.
        compile_process.args = "# Compiles the fuzz target."
        logger.debug(
            "Container %s: Compiled fuzz driver with sanitizer %s in %.4f seconds.",
            self.container_id,
            self.sanitizer,
            end_time - begin_time,
        )
        return compile_process

    def fuzz(
        self, run_timeout: int, log_path: str, corpus_dir: str
    ) -> None:
        command = [
            "python3",
            "infra/helper.py",
            "run_fuzzer",
            "-e",
            "ASAN_OPTIONS=detect_leaks=0",
            "--corpus-dir",
            corpus_dir,
            self.generated_oss_fuzz_name,
            self.benchmark.target_name,
            "--",
        ] + _libfuzzer_args(run_timeout)

        with open(log_path, "w") as f:
            proc = sp.Popen(
                command,
                stdin=sp.DEVNULL,
                stdout=f,
                stderr=sp.STDOUT,
                cwd=oss_fuzz_checkout.OSS_FUZZ_DIR,
            )
            try:
                proc.wait(timeout=run_timeout + 5)
            except sp.TimeoutExpired:
                logger.info("%s timed out during fuzzing.", self.generated_project_path)
                # Try continuing and parsing the logs even in case of timeout.
        if proc.returncode != 0:
            logger.debug(
                "Container %s: Fuzzing trial terminated with non-zero exit code",
                self.container_id,
            )
        else:
            logger.debug(
                "Container %s: Fuzzing trial was successful",
                self.container_id,
            )

    def update_runtime(self, runtime: float) -> None:
        self.total_cov_executions += 1
        if self.avg_cov_runtime < 0:
            self.avg_cov_runtime = runtime
        self.avg_cov_runtime += (
            runtime - self.avg_cov_runtime
        ) / self.total_cov_executions

    def get_coverage(self, corpus_dir: str, harness_name: str= "") -> sp.CompletedProcess:
        """Allocate two minutes for coverage collection"""
        corpus_size =  len(os.listdir(corpus_dir))
        if corpus_size == 0:
            logger.warning("Provided corpus path (%s) has no seeds in it!", corpus_dir)
        logger.debug("Corpus %s has size: %d", corpus_dir, len(os.listdir(corpus_dir)))
        command = [
            "python3",
            "infra/helper.py",
            "coverage",
            "--corpus-dir",
            corpus_dir,
            "--fuzz-target",
            self.benchmark.target_name,
            "--port",
            "",
            "--no-serve",
            self.generated_oss_fuzz_name,
        ]
        # add two seconds to the average timeout to accomodate for slower harnesses
        timeout_threshold = self.avg_cov_runtime + 2 if self.avg_cov_runtime > 0 else 30
        try:
            proc_start = time.perf_counter()
            proc = sp.run(
                command,
                capture_output=True,
                cwd=oss_fuzz_checkout.OSS_FUZZ_DIR,
                stdin=sp.DEVNULL,
                check=True,
                timeout=timeout_threshold,
            )
            proc_runtime = time.perf_counter() - proc_start
            self.update_runtime(proc_runtime)
            logger.debug("Container stdout:\n%s", proc.stdout)
        except sp.TimeoutExpired as e:
            logger.info(
                "Coverage timed out in %s seconds for harness %s",
                timeout_threshold,
                harness_name,
            )
        except sp.CalledProcessError as e:
            logger.info(
                "Failed to generate coverage for %s:\n%s\n%s",
                self.generated_oss_fuzz_name,
                e.stdout,
                e.stderr,
            )

    def rewrite_driver(self, content: str) -> None:
        self.write_to_file(content, self.benchmark.target_path)

    def rewrite_build_script(self, content: str) -> None:
        self.write_to_file(content, "/src/build.sh")

    def write_to_file(self, content: str, file_path: str) -> None:
        replace_file_content_command = (
            f'cat << "OFG_EOF" > {file_path}\n{content}\nOFG_EOF'
        )
        self.execute(replace_file_content_command)

    def terminate(self) -> bool:
        return True



def get_build_artifact_dir(generated_project: str, build_artifact: str) -> str:
    """
    Returns the |build_artifact| absolute directory path for |generated_project|.
    """
    return os.path.join(
        oss_fuzz_checkout.OSS_FUZZ_DIR, "build", build_artifact, generated_project
    )

def get_ccache_dir(generated_project: str) -> str:
    return os.path.join(oss_fuzz_checkout.OSS_FUZZ_DIR, "ccaches", generated_project, "ccache")

def _libfuzzer_args(run_timeout: int) -> list[str]:
    return [
        "-print_final_stats=1",
        f"-max_total_time={run_timeout}",
        # Without this flag, libFuzzer only consider short inputs in short
        # experiments, which lowers the coverage for quick performance tests.
        "-len_control=0",
        # Timeout per testcase.
        "-timeout=30",
        "-detect_leaks=0",
    ]
