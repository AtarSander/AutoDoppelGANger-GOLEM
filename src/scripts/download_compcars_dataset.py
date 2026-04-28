from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import gdown
import hydra
from loguru import logger
from omegaconf import DictConfig


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_command(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required command '{name}' is not available on PATH.")


def normalize_google_drive_folder_url(folder_url: str) -> str:
    parsed = urlparse(folder_url)
    query_id = parse_qs(parsed.query).get("id")
    if query_id:
        return f"https://drive.google.com/drive/folders/{query_id[0]}"
    return folder_url


def download_folder(raw_dir: Path, folder_url: str) -> list[Path]:
    ensure_dir(raw_dir)
    normalized_folder_url = normalize_google_drive_folder_url(folder_url)
    logger.info("Downloading CompCars folder from {}", normalized_folder_url)
    downloaded = gdown.download_folder(
        url=normalized_folder_url,
        output=str(raw_dir),
        quiet=False,
    )
    if not downloaded:
        raise RuntimeError(f"No files were downloaded from {normalized_folder_url}")
    normalized_paths: list[Path] = []
    for item in downloaded:
        path = item if isinstance(item, str) else item.path
        normalized_paths.append(Path(path))
    return normalized_paths


def collect_archive_parts(raw_dir: Path, prefix: str) -> list[Path]:
    parts = sorted(path for path in raw_dir.glob(f"{prefix}.*") if path.is_file())
    if not parts:
        raise FileNotFoundError(f"No archive parts matching {prefix}.* were found in {raw_dir}")
    zip_path = raw_dir / f"{prefix}.zip"
    if zip_path not in parts:
        raise FileNotFoundError(f"Expected archive starter {zip_path} was not downloaded")
    return parts


def combine_archive_parts(raw_dir: Path, prefix: str, combined_suffix: str) -> Path:
    ensure_command("zip")
    zip_path = raw_dir / f"{prefix}.zip"
    combined_path = raw_dir / f"{prefix}_{combined_suffix}"
    if combined_path.exists():
        logger.info("Using existing combined archive: {}", combined_path)
        return combined_path

    logger.info("Combining multipart archive for {}", prefix)
    subprocess.run(
        ["zip", "-F", zip_path.name, "--out", combined_path.name],
        cwd=raw_dir,
        check=True,
        text=True,
    )
    return combined_path


def extract_password_archive(
    combined_path: Path,
    output_dir: Path,
    password: str,
    disable_zipbomb_detection: bool,
) -> None:
    ensure_command("unzip")
    ensure_dir(output_dir)
    marker = output_dir / ".extracted"
    if marker.exists():
        logger.info("Using existing extraction: {}", output_dir)
        return

    logger.info("Extracting {} -> {}", combined_path, output_dir)
    env = os.environ.copy()
    if disable_zipbomb_detection:
        env["UNZIP_DISABLE_ZIPBOMB_DETECTION"] = "TRUE"
    subprocess.run(
        ["unzip", "-o", "-P", password, str(combined_path), "-d", str(output_dir)],
        check=True,
        text=True,
        env=env,
    )
    marker.write_text("ok\n", encoding="utf-8")


def write_manifest(output_dir: Path, manifest: dict[str, object]) -> Path:
    ensure_dir(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def process_archive_group(
    raw_dir: Path,
    extracted_root: Path,
    prefix: str,
    password: str,
    combined_suffix: str,
    disable_zipbomb_detection: bool,
) -> dict[str, object]:
    parts = collect_archive_parts(raw_dir, prefix)
    combined_path = combine_archive_parts(raw_dir, prefix, combined_suffix)
    group_output_dir = extracted_root / prefix
    extract_password_archive(
        combined_path=combined_path,
        output_dir=group_output_dir,
        password=password,
        disable_zipbomb_detection=disable_zipbomb_detection,
    )
    return {
        "prefix": prefix,
        "parts": [str(path) for path in parts],
        "combined_archive": str(combined_path),
        "output_dir": str(group_output_dir),
    }


@hydra.main(
    config_path="../../config",
    config_name="download_compcars_dataset",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:
    raw_dir = Path(hydra.utils.to_absolute_path(cfg.raw_dir))
    interim_dir = Path(hydra.utils.to_absolute_path(cfg.interim_dir))
    output_dir = Path(hydra.utils.to_absolute_path(cfg.output_dir))

    ensure_dir(raw_dir)
    ensure_dir(interim_dir)
    ensure_dir(output_dir)

    if cfg.download.source != "google_drive":
        raise ValueError("CompCars downloader currently supports only download.source=google_drive")

    downloaded_files = download_folder(raw_dir, cfg.download.google_drive_folder_url)
    extracted_root = interim_dir / "extracted"
    manifest: dict[str, object] = {
        "dataset": "CompCars",
        "source": cfg.download.google_drive_folder_url,
        "mirrors": {
            "google_drive": cfg.download.google_drive_folder_url,
            "dropbox": cfg.download.dropbox_folder_url,
        },
        "downloaded_files": [str(path) for path in downloaded_files],
        "archives": {},
    }

    web_report = process_archive_group(
        raw_dir=raw_dir,
        extracted_root=extracted_root,
        prefix=cfg.archives.web_prefix,
        password=cfg.archives.password,
        combined_suffix=cfg.archives.combined_suffix,
        disable_zipbomb_detection=cfg.archives.disable_zipbomb_detection,
    )
    manifest["archives"]["web"] = web_report

    if cfg.include_surveillance:
        surveillance_report = process_archive_group(
            raw_dir=raw_dir,
            extracted_root=extracted_root,
            prefix=cfg.archives.surveillance_prefix,
            password=cfg.archives.password,
            combined_suffix=cfg.archives.combined_suffix,
            disable_zipbomb_detection=cfg.archives.disable_zipbomb_detection,
        )
        manifest["archives"]["surveillance"] = surveillance_report

    manifest_path = write_manifest(output_dir, manifest)
    logger.info("CompCars dataset prepared. Manifest written to {}", manifest_path)

    if cfg.cleanup_intermediate:
        logger.info("Removing intermediate extraction directory: {}", extracted_root)
        shutil.rmtree(extracted_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
