from pathlib import Path

from cytobridge_agent.tools.path_resolver import pick_path_from_message


def test_pick_path_skips_root_slash_from_prose(tmp_path: Path) -> None:
    data = tmp_path / "train.h5ad"
    data.write_text("placeholder")

    message = (
        "Completion requires files produced for the current requested output "
        f"directory / current run. Workspace: {tmp_path}. Training AnnData: {data}"
    )

    assert pick_path_from_message(message, cwd=str(tmp_path)) == str(data.resolve())


def test_pick_path_does_not_return_root_directory(tmp_path: Path) -> None:
    assert pick_path_from_message("directory / current run", cwd=str(tmp_path)) is None
