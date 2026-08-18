import subprocess

import pytest

import oxspires_tools.bash_command as bash_command


class FakeProcess:
    def __init__(self, return_code):
        self.stdout = ["output line\n"]
        self.return_code = return_code
        self.wait_called = False

    def wait(self):
        self.wait_called = True
        return self.return_code


def test_run_command_waits_for_successful_streamed_command(monkeypatch):
    process = FakeProcess(0)
    monkeypatch.setattr(bash_command.subprocess, "Popen", lambda *args, **kwargs: process)

    assert bash_command.run_command("benchmark command", print_command=False) is None
    assert process.wait_called is True


def test_run_command_raises_on_failed_streamed_command(monkeypatch):
    process = FakeProcess(7)
    monkeypatch.setattr(bash_command.subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(subprocess.CalledProcessError) as exc:
        bash_command.run_command("benchmark command", print_command=False)

    assert exc.value.returncode == 7
    assert exc.value.cmd == "benchmark command"
    assert process.wait_called is True


def test_run_command_keeps_async_process_contract_when_output_not_streamed(monkeypatch):
    process = FakeProcess(9)
    monkeypatch.setattr(bash_command.subprocess, "Popen", lambda *args, **kwargs: process)

    returned = bash_command.run_command("evo command", print_command=False, print_output=False)

    assert returned is process
    assert process.wait_called is False
