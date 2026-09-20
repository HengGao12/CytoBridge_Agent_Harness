import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from cytobridge_agent.benchmark_artifacts import (
    build_algorithm_benchmark_bundle,
    install_algorithm_benchmarks,
    load_manifest,
    sha256_file,
)


class BenchmarkArtifactTests(unittest.TestCase):
    def test_bundled_manifest_loads(self):
        manifest = load_manifest()
        self.assertEqual(manifest["name"], "cellcompass-algorithm-benchmarks")
        self.assertTrue(manifest["assets"])

    def test_build_and_install_bundle_excludes_top_level_builtin_runs_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "algorithm_benchmarks"
            dataset = root / "datasets" / "toy_dataset"
            dataset.mkdir(parents=True)
            (root / "simulation_generators").mkdir()
            (root / "builtin_runs").mkdir()
            (root / "registry.json").write_text(
                json.dumps({"schema_version": 1, "datasets": {}}),
                encoding="utf-8",
            )
            (dataset / "dataset.json").write_text(
                json.dumps({"dataset_id": "toy_dataset", "title": "Toy"}),
                encoding="utf-8",
            )
            (dataset / "leaderboard.json").write_text(
                json.dumps({"entries": []}),
                encoding="utf-8",
            )
            (dataset / "data.h5ad").write_bytes(b"fake-h5ad")
            (root / "builtin_runs" / "historical.json").write_text("{}", encoding="utf-8")

            archive = Path(tmp) / "benchmarks.tar"
            metadata = build_algorithm_benchmark_bundle(source_root=root, output_path=archive)
            self.assertEqual(metadata["file_count"], 4)

            output = Path(tmp) / "install"
            result = install_algorithm_benchmarks(
                asset_path=archive,
                output_root=output,
                verify_sha256=False,
            )
            self.assertTrue(result.ok)
            self.assertIn("toy_dataset", result.datasets)
            self.assertTrue((output / "algorithm_benchmarks" / "datasets" / "toy_dataset" / "data.h5ad").exists())
            self.assertFalse((output / "algorithm_benchmarks" / "builtin_runs").exists())

    def test_install_skips_existing_files_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "benchmarks.tar"
            payload = b"from-archive"
            with tarfile.open(archive, "w") as tar:
                info = tarfile.TarInfo("algorithm_benchmarks/datasets/d1/data.h5ad")
                info.size = len(payload)
                tar.addfile(info, fileobj=io.BytesIO(payload))

            output = Path(tmp) / "install"
            target = output / "algorithm_benchmarks" / "datasets" / "d1" / "data.h5ad"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"existing")

            result = install_algorithm_benchmarks(
                asset_path=archive,
                output_root=output,
                verify_sha256=False,
            )
            self.assertEqual(result.files_skipped, 1)
            self.assertEqual(target.read_bytes(), b"existing")

            result = install_algorithm_benchmarks(
                asset_path=archive,
                output_root=output,
                force=True,
                verify_sha256=False,
            )
            self.assertEqual(result.files_written, 1)
            self.assertEqual(target.read_bytes(), payload)

    def test_install_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "bad.tar"
            payload = b"bad"
            with tarfile.open(archive, "w") as tar:
                info = tarfile.TarInfo("algorithm_benchmarks/../evil.txt")
                info.size = len(payload)
                tar.addfile(info, fileobj=io.BytesIO(payload))

            with self.assertRaises(ValueError):
                install_algorithm_benchmarks(
                    asset_path=archive,
                    output_root=Path(tmp) / "install",
                    verify_sha256=False,
                )

    def test_install_multipart_manifest_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "benchmarks.tar"
            payload = b"multipart"
            with tarfile.open(archive, "w") as tar:
                info = tarfile.TarInfo("algorithm_benchmarks/datasets/d1/data.h5ad")
                info.size = len(payload)
                tar.addfile(info, fileobj=io.BytesIO(payload))

            content = archive.read_bytes()
            part_0 = Path(tmp) / "benchmarks.tar.part-00"
            part_1 = Path(tmp) / "benchmarks.tar.part-01"
            midpoint = len(content) // 2
            part_0.write_bytes(content[:midpoint])
            part_1.write_bytes(content[midpoint:])
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "assets": [
                            {
                                "name": archive.name,
                                "sha256": sha256_file(archive),
                                "parts": [
                                    {
                                        "name": part_0.name,
                                        "url": part_0.as_uri(),
                                        "sha256": sha256_file(part_0),
                                    },
                                    {
                                        "name": part_1.name,
                                        "url": part_1.as_uri(),
                                        "sha256": sha256_file(part_1),
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output = Path(tmp) / "install"
            result = install_algorithm_benchmarks(manifest_path=manifest, output_root=output)
            self.assertTrue(result.ok)
            self.assertEqual(
                (output / "algorithm_benchmarks" / "datasets" / "d1" / "data.h5ad").read_bytes(),
                payload,
            )


if __name__ == "__main__":
    unittest.main()
