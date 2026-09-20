import json
import subprocess
import sys
import textwrap
import unittest

from cytobridge_agent.utils.codex_sidecar_client import CodexSidecarClient


class CodexSidecarClientTests(unittest.TestCase):
    def test_call_ignores_stale_response_ids(self) -> None:
        script = textwrap.dedent(
            """
            import json
            import sys

            for line in sys.stdin:
                request = json.loads(line)
                print(json.dumps({"id": 999, "result": {"stale": True}}), flush=True)
                print(json.dumps({"id": request["id"], "result": {"ok": True}}), flush=True)
            """
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        client = CodexSidecarClient()
        client._proc = proc
        try:
            result = client.call("chat.codex.complete", {"hello": "world"})
        finally:
            client.close()
            if proc.stdin:
                proc.stdin.close()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

        self.assertEqual(result, {"ok": True})


if __name__ == "__main__":
    unittest.main()
