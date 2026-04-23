from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="stage-b-test-1-gpu-large")

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class TestNemotronDiffusionFastDiffuser(CustomTestCase):
    """Test Nemotron-Diffusion with FastDiffuser (iterative denoising)."""

    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "4",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "FastDiffuser",
            "--dllm-algorithm-config",
            "test/registered/dllm/configs/nemotron_fastdiffuser.yaml",
            "--cuda-graph-bs",
            "1",
            "2",
            "3",
            "4",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["score"], 0.85)
        self.assertGreater(metrics["output_throughput"], 100)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (nemotron-diffusion-fastdiffuser) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, 100)


class TestNemotronDiffusionLinearSpec(CustomTestCase):
    """Test Nemotron-Diffusion with LinearSpec (speculative decoding)."""

    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "4",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "LinearSpec",
            "--dllm-algorithm-config",
            "test/registered/dllm/configs/nemotron_linearspec.yaml",
            "--cuda-graph-bs",
            "1",
            "2",
            "3",
            "4",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["score"], 0.85)
        self.assertGreater(metrics["output_throughput"], 200)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (nemotron-diffusion-linearspec) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, 200)


if __name__ == "__main__":
    unittest.main()
