from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

import hydra
import lmdb
from loguru import logger
from omegaconf import DictConfig
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    timeout_seconds: int,
    retries: int,
    backoff_seconds: int,
    chunk_size_bytes: int,
    user_agent: str,
) -> None:
    ensure_dir(destination.parent)
    if destination.exists():
        logger.info("Using existing archive: {}", destination)
        return

    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        partial_path = destination.with_suffix(f"{destination.suffix}.part")
        logger.info(
            "Downloading {} -> {} (attempt {}/{}, timeout={}s)",
            url,
            destination,
            attempt,
            retries,
            timeout_seconds,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                total_size = response.headers.get("Content-Length")
                total = int(total_size) if total_size is not None else None
                with partial_path.open("wb") as handle, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name,
                ) as progress:
                    while chunk := response.read(chunk_size_bytes):
                        handle.write(chunk)
                        progress.update(len(chunk))
            partial_path.replace(destination)
            return
        except (TimeoutError, URLError, HTTPError, OSError) as error:
            partial_path.unlink(missing_ok=True)
            last_error = error
            if attempt == retries:
                break

            sleep_seconds = backoff_seconds * attempt
            logger.warning(
                "Download attempt {}/{} failed for {}: {}. Retrying in {}s.",
                attempt,
                retries,
                url,
                error,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_error


def verify_zip_archive(archive_path: Path) -> dict[str, object]:
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"{archive_path} is not a valid zip archive.")

    with zipfile.ZipFile(archive_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"{archive_path} contains a corrupt member: {corrupt_member}")
        member_count = len(archive.infolist())

    archive_size = archive_path.stat().st_size
    archive_sha256 = sha256_file(archive_path)
    return {
        "archive": str(archive_path),
        "size_bytes": archive_size,
        "sha256": archive_sha256,
        "members": member_count,
    }


def ensure_valid_archive(url: str, archive_path: Path, cfg: DictConfig) -> dict[str, object]:
    download_file(
        url=url,
        destination=archive_path,
        timeout_seconds=cfg.download.timeout_seconds,
        retries=cfg.download.retries,
        backoff_seconds=cfg.download.backoff_seconds,
        chunk_size_bytes=cfg.download.chunk_size_bytes,
        user_agent=cfg.download.user_agent,
    )
    try:
        return verify_zip_archive(archive_path)
    except ValueError:
        logger.warning("Archive verification failed for {}, re-downloading once.", archive_path)
        archive_path.unlink(missing_ok=True)
        download_file(
            url=url,
            destination=archive_path,
            timeout_seconds=cfg.download.timeout_seconds,
            retries=cfg.download.retries,
            backoff_seconds=cfg.download.backoff_seconds,
            chunk_size_bytes=cfg.download.chunk_size_bytes,
            user_agent=cfg.download.user_agent,
        )
        return verify_zip_archive(archive_path)


def extract_zip_archive(archive_path: Path, destination: Path) -> Path:
    ensure_dir(destination)
    marker = destination / ".extracted"
    if marker.exists():
        logger.info("Using existing extraction: {}", destination)
        return locate_lmdb_root(destination)

    logger.info("Extracting {} -> {}", archive_path, destination)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    marker.write_text("ok\n", encoding="utf-8")
    return locate_lmdb_root(destination)


def locate_lmdb_root(extraction_dir: Path) -> Path:
    data_mdb_matches = list(extraction_dir.rglob("data.mdb"))
    if len(data_mdb_matches) != 1:
        raise FileNotFoundError(
            f"Expected one LMDB under {extraction_dir}, found {len(data_mdb_matches)}."
        )
    return data_mdb_matches[0].parent


def verify_lmdb_directory(lmdb_dir: Path) -> int:
    lock_path = lmdb_dir / "lock.mdb"
    data_path = lmdb_dir / "data.mdb"
    if not data_path.exists() or not lock_path.exists():
        raise FileNotFoundError(f"Missing LMDB files in {lmdb_dir}")

    env = lmdb.open(
        str(lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=32,
        subdir=True,
    )
    try:
        with env.begin(write=False) as txn:
            count = txn.stat()["entries"]
    finally:
        env.close()
    return int(count)


def export_lmdb_to_imagefolder(
    lmdb_dir: Path,
    class_dir: Path,
    split_name: str,
    limit: int | None = None,
) -> dict[str, object]:
    ensure_dir(class_dir)
    env = lmdb.open(
        str(lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=32,
        subdir=True,
    )
    exported = 0
    skipped = 0
    skipped_keys: list[str] = []

    try:
        with env.begin(write=False) as txn:
            total_entries = txn.stat()["entries"]
            cursor = txn.cursor()
            with tqdm(total=total_entries, desc=f"Converting {split_name}", unit="img") as progress:
                for key, value in cursor:
                    key_text = key.decode("utf-8", errors="ignore")
                    try:
                        with Image.open(io_from_bytes(value)) as image:
                            image.verify()
                        with Image.open(io_from_bytes(value)) as image:
                            rgb_image = image.convert("RGB")
                            output_path = class_dir / f"{split_name}_{key_text}.jpg"
                            rgb_image.save(output_path, format="JPEG", quality=95)
                        exported += 1
                    except (OSError, UnidentifiedImageError, ValueError):
                        skipped += 1
                        if len(skipped_keys) < 100:
                            skipped_keys.append(key_text)

                    progress.update(1)
                    if limit is not None and exported >= limit:
                        break
    finally:
        env.close()

    return {
        "split": split_name,
        "lmdb": str(lmdb_dir),
        "exported_images": exported,
        "skipped_images": skipped,
        "sample_skipped_keys": skipped_keys,
    }


def io_from_bytes(raw_bytes: bytes) -> BytesIO:
    return BytesIO(raw_bytes)


def write_manifest(output_dir: Path, manifest: dict[str, object]) -> Path:
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def archive_name_from_url(url: str, split_name: str) -> str:
    parsed = urlparse(url)
    candidate = Path(parsed.path).name
    if candidate:
        return candidate
    return f"{split_name}.zip"


def prepare_output_dir(output_dir: Path) -> Path:
    class_dir = output_dir / "car"
    if class_dir.exists():
        logger.info("Removing previous ImageFolder output: {}", class_dir)
        shutil.rmtree(class_dir)
    ensure_dir(class_dir)
    return class_dir


@hydra.main(config_path="../../config", config_name="download_lsun_dataset", version_base="1.3")
def main(cfg: DictConfig) -> int:
    raw_dir = Path(hydra.utils.to_absolute_path(cfg.raw_dir))
    interim_dir = Path(hydra.utils.to_absolute_path(cfg.interim_dir))
    output_dir = Path(hydra.utils.to_absolute_path(cfg.output_dir))

    ensure_dir(raw_dir)
    ensure_dir(interim_dir)
    ensure_dir(output_dir)

    class_dir = prepare_output_dir(output_dir)
    manifest: dict[str, object] = {
        "dataset": cfg.dataset.name,
        "source": cfg.dataset.source,
        "splits": {},
    }

    logger.info("Preparing {} ImageFolder dataset in {}", cfg.dataset.name, class_dir)

    for split_name, url in cfg.dataset.splits.items():
        archive_name = archive_name_from_url(url, split_name)
        archive_stem = Path(archive_name).stem
        archive_path = raw_dir / archive_name
        extraction_dir = interim_dir / archive_stem

        archive_report = ensure_valid_archive(url, archive_path, cfg)
        lmdb_root = extract_zip_archive(archive_path, extraction_dir)
        lmdb_entries = verify_lmdb_directory(lmdb_root)
        export_report = export_lmdb_to_imagefolder(
            lmdb_dir=lmdb_root,
            class_dir=class_dir,
            split_name=split_name,
            limit=cfg.limit,
        )

        manifest["splits"][split_name] = {
            "archive": archive_report,
            "lmdb_entries": lmdb_entries,
            "conversion": export_report,
        }

        if not cfg.keep_lmdb:
            logger.info("Removing extracted LMDB: {}", extraction_dir)
            shutil.rmtree(extraction_dir, ignore_errors=True)

    manifest_path = write_manifest(output_dir, manifest)
    total_exported = sum(
        split["conversion"]["exported_images"]  # type: ignore[index]
        for split in manifest["splits"].values()  # type: ignore[union-attr]
    )
    logger.info("Finished. Exported {} images into {}", total_exported, class_dir)
    logger.info("Manifest written to {}", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
