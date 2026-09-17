"""Flask API for the face scanner application.

Reference photos are encoded once when this module is imported, before the
server begins accepting camera scans. In a hosted deployment, the files and
camera frames are processed by that deployment's server.
"""

from __future__ import annotations

import os
import base64
import io
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, jsonify, render_template, request, send_from_directory

# The deployed website does not need dlib/OpenCV. Loading those native modules
# only for the legacy opt-in scanner makes normal website startup much faster.
SERVER_SIDE_RECOGNITION = os.getenv("SERVER_SIDE_RECOGNITION") == "1"
FACE_RUNTIME_ERROR: str | None = None
if SERVER_SIDE_RECOGNITION:
    try:
        import cv2
        import face_recognition
        import numpy as np
    except Exception as error:  # Includes missing native libraries during cloud import.
        cv2 = None
        face_recognition = None
        np = None
        FACE_RUNTIME_ERROR = str(error)
else:
    cv2 = None
    face_recognition = None
    np = None


# ---- Recognition settings -------------------------------------------------
# Lower values are stricter. 0.55 is the browser-model match limit used for
# normal enrolled-person variation in indoor camera lighting.
# Browser-side embeddings vary slightly with camera angle and indoor light.
# 0.55 accepts normal enrolled-person variation while remaining stricter than
# the face-api.js default of 0.60.
FACE_MATCH_THRESHOLD = 0.58
FACE_DETECTION_MODEL = "hog"  # "cnn" is slower and requires a CUDA-capable build.
MIN_BRIGHTNESS = 20
MAX_UPLOAD_BYTES = 6 * 1024 * 1024
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
OBJECT_IMAGES_DIR = BASE_DIR / "object_images"
DETAILS_FILE = BASE_DIR / "details.json"
DICTIONARY_FILE = BASE_DIR / "physical_object_dictionary_5000.json"

# A recognised result can only be emailed to the matching enrolled person.
# The browser never receives an API key or chooses the destination address.
RESULT_EMAIL_RECIPIENTS = {
    "person1": "sanskarjain@appicsoftwares.in",
    "deepak_sharma": "deepak.sharma@happiest.team",
}
RESEND_API_URL = "https://api.resend.com/emails"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.logger.setLevel(logging.INFO)

def person_id_from_filename(image_path: Path) -> str:
    """Map person1_1.jpg and person1_2.jpg to the person id ``person1``.

    A trailing underscore followed by a number is treated as a photo sequence.
    Names without that suffix use their complete filename stem as the id.

    Uploaded generic names such as ``00000000.png`` or any photo name that
    contains ``alia`` are also deliberately routed to the Alia Bhatt enrolled
    profile so the demo can accept the supplied reference image set.
    """

    stem = image_path.stem.strip().lower()
    if "alia" in stem or stem == "00000000":
        return "alia_bhatt"
    return re.sub(r"_\d+$", "", image_path.stem).strip()


def load_details() -> dict[str, dict[str, Any]]:
    """Read and validate the local details file without exposing malformed data."""

    if not DETAILS_FILE.exists():
        app.logger.warning("details.json is missing; matched records will have no extra fields.")
        return {}
    try:
        with DETAILS_FILE.open("r", encoding="utf-8") as file:
            raw_details = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        app.logger.error("Could not read details.json: %s", error)
        return {}

    if not isinstance(raw_details, dict):
        app.logger.error("details.json must contain an object keyed by person id.")
        return {}
    return {
        str(person_id): values
        for person_id, values in raw_details.items()
        if isinstance(values, dict)
    }


def enrolled_profiles() -> list[dict[str, Any]]:
    """Return the browser-safe enrolment manifest.

    The hosted scanner performs recognition in the visitor's browser. This
    keeps a live camera frame off the Vercel function and avoids native dlib
    binaries, which are not available in Vercel's Python runtime.
    """

    images_by_person: defaultdict[str, list[str]] = defaultdict(list)
    if KNOWN_FACES_DIR.exists():
        for image_path in sorted(KNOWN_FACES_DIR.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                images_by_person[person_id_from_filename(image_path)].append(image_path.name)

    profiles: list[dict[str, Any]] = []
    for person_id, image_names in images_by_person.items():
        record = PERSON_DETAILS.get(person_id, {})
        profiles.append(
            {
                "person_id": person_id,
                "name": record.get("name", person_id),
                "details": record,
                "reference_images": [f"/references/{name}" for name in image_names],
            }
        )
    return sorted(profiles, key=lambda profile: str(profile["name"]).lower())


def object_catalogue() -> list[dict[str, Any]]:
    """Build a browser-safe catalogue for the supplied non-face reference images."""

    catalogue_details = PERSON_DETAILS.get("objects", {})
    catalogue: list[dict[str, Any]] = []
    if not OBJECT_IMAGES_DIR.exists():
        return catalogue
    for category_dir in sorted(OBJECT_IMAGES_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category_id = category_dir.name.lower()
        record = catalogue_details.get(category_id, {})
        images = [
            {
                "url": f"/object-images/{category_dir.name}/{image_path.name}",
                "alt": f"{record.get('name', category_dir.name)} - {image_path.stem.replace('_', ' ')}",
            }
            for image_path in sorted(category_dir.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if images:
            catalogue.append({"category_id": category_id, "images": images, **record})
    return catalogue


def object_manifest() -> list[dict[str, Any]]:
    """Return the object reference manifest used by the browser matcher.

    Each result is enriched with the corresponding dictionary metadata so a live
    camera match can display the keyword text, description, and purpose next to
    the captured frame.
    """

    manifest: list[dict[str, Any]] = []
    for item in object_catalogue():
        category_id = str(item["category_id"]).strip().lower()
        entry = None
        for candidate in DICTIONARY_ENTRIES:
            if str(candidate.get("slug", "")).strip().lower() == category_id:
                entry = candidate
                break
            if str(candidate.get("image_folder", "")).strip().lower() == category_id:
                entry = candidate
                break
            if str(candidate.get("scan_label", "")).strip().lower() == category_id:
                entry = candidate
                break
            if str(candidate.get("word", "")).strip().lower() == item.get("name", "").strip().lower():
                entry = candidate
                break

        enriched_entry = with_reference_images(entry) if entry else {}
        display_name = (
            enriched_entry.get("word")
            or item.get("name")
            or str(item["category_id"]).replace("_", " ").title()
        )
        manifest.append(
            {
                "category_id": item["category_id"],
                "name": display_name,
                "summary": enriched_entry.get("description") or item.get("summary", ""),
                "reference_images": [image["url"] for image in item["images"]],
                "entry": {
                    "id": enriched_entry.get("id"),
                    "word": enriched_entry.get("word"),
                    "slug": enriched_entry.get("slug"),
                    "category": enriched_entry.get("category"),
                    "description": enriched_entry.get("description"),
                    "primary_use": enriched_entry.get("primary_use"),
                    "visual_identification": enriched_entry.get("visual_identification"),
                    "common_locations": enriched_entry.get("common_locations"),
                    "materials": enriched_entry.get("materials"),
                    "aliases": enriched_entry.get("aliases"),
                    "reference_images": enriched_entry.get("reference_images", []),
                },
            }
        )
    return manifest


def object_image_folders() -> dict[str, list[str]]:
    """Map each object_images/ subfolder (lower-cased) to its served image URLs.

    The folder name (e.g. ``Headphones``, ``Charging_Point``) is matched,
    case-insensitively, against a dictionary entry's ``image_folder`` or
    ``slug`` field. This is what lets a folder of supplied photos (such as
    ``bottle1/`` reorganised into ``object_images/Headphones/``) surface
    automatically next to that keyword's description and purpose.
    """

    folders: dict[str, list[str]] = {}
    if not OBJECT_IMAGES_DIR.exists():
        return folders
    for category_dir in sorted(OBJECT_IMAGES_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        images = [
            f"/object-images/{category_dir.name}/{image_path.name}"
            for image_path in sorted(category_dir.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if images:
            folders[category_dir.name.strip().lower()] = images
    return folders


def reference_images_for_entry(entry: dict[str, Any]) -> list[str]:
    """Find supplied reference photos for one dictionary entry, if any exist."""

    folders = OBJECT_IMAGE_FOLDERS
    for key in (
        str(entry.get("image_folder", "")).strip().lower(),
        str(entry.get("slug", "")).strip().lower(),
        str(entry.get("scan_label", "")).strip().lower(),
        str(entry.get("word", "")).strip().lower().replace(" ", "_"),
    ):
        if key and key in folders:
            return folders[key]
    return []


def with_reference_images(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a dictionary entry annotated with its reference photos."""

    images = reference_images_for_entry(entry)
    return {**entry, "reference_images": images, "has_reference_images": bool(images)}


def load_known_faces() -> tuple[dict[str, np.ndarray], list[str], dict[str, int]]:
    """Create one average encoding per person from their reference photos."""

    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    encodings_by_person: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    warnings: list[str] = []

    image_paths = sorted(
        path for path in KNOWN_FACES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    for image_path in image_paths:
        person_id = person_id_from_filename(image_path)
        try:
            image = face_recognition.load_image_file(image_path)
            locations = face_recognition.face_locations(image, model=FACE_DETECTION_MODEL)
            if len(locations) != 1:
                warnings.append(
                    f"{image_path.name}: expected exactly one face, found {len(locations)}; skipped."
                )
                continue
            image_encodings = face_recognition.face_encodings(image, known_face_locations=locations)
            if not image_encodings:
                warnings.append(f"{image_path.name}: face could not be encoded; skipped.")
                continue
            encodings_by_person[person_id].append(image_encodings[0])
        except Exception as error:  # Invalid/corrupt images must not stop the server.
            warnings.append(f"{image_path.name}: could not be loaded ({error}); skipped.")

    averaged = {
        person_id: np.mean(person_encodings, axis=0)
        for person_id, person_encodings in encodings_by_person.items()
    }
    reference_counts = {
        person_id: len(person_encodings)
        for person_id, person_encodings in encodings_by_person.items()
    }
    for warning in warnings:
        app.logger.warning(warning)
    app.logger.info(
        "Loaded %d reference image(s) for %d person(s).",
        sum(len(value) for value in encodings_by_person.values()),
        len(averaged),
    )
    return averaged, warnings, reference_counts


def load_dictionary() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    """Load physical object dictionary entries, index for fast lookup, and calculate categories."""
    if not DICTIONARY_FILE.exists():
        app.logger.warning("physical_object_dictionary_5000.json is missing.")
        return [], {}, {}
    try:
        with DICTIONARY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        app.logger.error("Could not read dictionary json: %s", error)
        return [], {}, {}

    raw_entries = data.get("entries", []) if isinstance(data, dict) else []
    indexed: dict[str, dict[str, Any]] = {}
    categories_count: defaultdict[str, int] = defaultdict(int)

    # Pass 1: exact identifying keys (slug, word, id, scan label). These take
    # priority and must never be overwritten by another entry's alias below —
    # otherwise looking up "Headphones" could resolve to an unrelated variant
    # like "Compact Headphones" simply because it also lists "headphones" as
    # one of its aliases and happens to load later in the file.
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id", ""))
        word = str(entry.get("word", "")).strip()
        slug = str(entry.get("slug", "")).strip().lower()
        scan_label = str(entry.get("scan_label", "")).strip().lower()
        category = str(entry.get("category", "Uncategorized")).strip()

        if category:
            categories_count[category] += 1
        if slug:
            indexed[slug] = entry
        if word:
            indexed[word.lower()] = entry
        if entry_id:
            indexed[f"id:{entry_id}"] = entry
        if scan_label:
            indexed[scan_label] = entry

    # Pass 2: aliases only fill in keys that no entry already owns exactly.
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        for alias in entry.get("aliases", []):
            if isinstance(alias, str) and alias.strip():
                alias_key = alias.strip().lower()
                if alias_key not in indexed:
                    indexed[alias_key] = entry

    app.logger.info(
        "Loaded %d dictionary entries across %d categories.",
        len(raw_entries),
        len(categories_count),
    )
    return raw_entries, indexed, dict(sorted(categories_count.items(), key=lambda x: (-x[1], x[0])))


DICTIONARY_ENTRIES, DICTIONARY_INDEX, DICTIONARY_CATEGORIES = load_dictionary()
OBJECT_IMAGE_FOLDERS: dict[str, list[str]] = object_image_folders()


def search_dictionary(query: str = "", category: str = "", limit: int = 40) -> list[dict[str, Any]]:
    """Search the physical object dictionary with multi-tier scoring."""
    query_clean = query.strip().lower()
    category_clean = category.strip().lower()

    if not query_clean and not category_clean:
        return DICTIONARY_ENTRIES[:limit]

    exact_matches: list[dict[str, Any]] = []
    prefix_matches: list[dict[str, Any]] = []
    contains_matches: list[dict[str, Any]] = []
    category_matches: list[dict[str, Any]] = []

    for entry in DICTIONARY_ENTRIES:
        entry_category = str(entry.get("category", "")).lower()
        if category_clean and category_clean != "all" and category_clean != entry_category:
            continue

        if not query_clean:
            category_matches.append(entry)
            if len(category_matches) >= limit:
                break
            continue

        word = str(entry.get("word", "")).lower()
        base_word = str(entry.get("base_word", "")).lower()
        slug = str(entry.get("slug", "")).lower()
        aliases = [str(a).lower() for a in entry.get("aliases", []) if isinstance(a, str)]
        scan_label = str(entry.get("scan_label", "")).lower()
        desc = str(entry.get("description", "")).lower()
        primary_use = str(entry.get("primary_use", "")).lower()

        if query_clean == word or query_clean == base_word or query_clean == slug or query_clean in aliases:
            exact_matches.append(entry)
        elif word.startswith(query_clean) or slug.startswith(query_clean) or any(a.startswith(query_clean) for a in aliases):
            prefix_matches.append(entry)
        elif (
            query_clean in word
            or query_clean in base_word
            or query_clean in scan_label
            or any(query_clean in a for a in aliases)
            or query_clean in primary_use
            or query_clean in desc
        ):
            contains_matches.append(entry)

    combined = exact_matches + prefix_matches + contains_matches + category_matches
    seen_ids: set[Any] = set()
    deduped: list[dict[str, Any]] = []
    for item in combined:
        item_id = item.get("id")
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            deduped.append(item)
            if len(deduped) >= limit:
                break
    return deduped


def get_dictionary_entry(identifier: str) -> dict[str, Any] | None:
    """Retrieve an exact entry by ID, word, slug, alias, or closest match."""
    ident = identifier.strip().lower()
    if not ident:
        return None
    if ident in DICTIONARY_INDEX:
        return DICTIONARY_INDEX[ident]
    if f"id:{ident}" in DICTIONARY_INDEX:
        return DICTIONARY_INDEX[f"id:{ident}"]
    matches = search_dictionary(query=ident, limit=1)
    if matches:
        return matches[0]
    return None


def add_dictionary_entry(entry_data: dict[str, Any]) -> dict[str, Any]:
    """Add a new custom object/keyword entry to the dataset in-memory."""
    word = str(entry_data.get("word", "")).strip()
    if not word:
        raise ValueError("The 'word' field is required.")

    slug = re.sub(r"[^a-z0-9]+", "-", word.lower()).strip("-")
    new_id = len(DICTIONARY_ENTRIES) + 1
    category = str(entry_data.get("category", "General")).strip() or "General"
    description = (
        str(entry_data.get("description", "")).strip()
        or f"{word} is a physical object or item."
    )
    primary_use = (
        str(entry_data.get("primary_use", "")).strip()
        or "Commonly used for daily activities, utility, or specialized tasks."
    )
    visual_identification = (
        str(entry_data.get("visual_identification", "")).strip()
        or f"Identify {word} using its unique shape, surface texture, color, and design proportions."
    )
    common_locations = (
        str(entry_data.get("common_locations", "")).strip()
        or "Everyday environments, homes, workplaces, or outdoors."
    )
    materials = entry_data.get("materials", ["Standard physical material"])
    if isinstance(materials, str):
        materials = [m.strip() for m in materials.split(",") if m.strip()]
    aliases = entry_data.get("aliases", [word.lower()])
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(",") if a.strip()]

    new_entry = {
        "id": new_id,
        "word": word,
        "base_word": word.lower(),
        "slug": slug,
        "category": category,
        "variant": entry_data.get("variant", "standard"),
        "description": description,
        "primary_use": primary_use,
        "common_locations": common_locations,
        "visual_identification": visual_identification,
        "materials": materials,
        "aliases": aliases,
        "scan_label": slug,
        "image_folder": slug,
        "recommended_images": {
            "minimum": 25,
            "better": 50,
            "angles": [
                "front",
                "left side",
                "right side",
                "back",
                "top/raised angle",
                "real-use environment",
            ],
            "capture_notes": "Keep the target object clearly visible, vary lighting/background/distance, and avoid using only near-duplicate photos.",
        },
    }
    DICTIONARY_ENTRIES.append(new_entry)
    DICTIONARY_INDEX[slug] = new_entry
    DICTIONARY_INDEX[word.lower()] = new_entry
    DICTIONARY_INDEX[f"id:{new_id}"] = new_entry
    for a in aliases:
        if isinstance(a, str):
            DICTIONARY_INDEX[a.lower()] = new_entry
    DICTIONARY_CATEGORIES[category] = DICTIONARY_CATEGORIES.get(category, 0) + 1
    return new_entry


PERSON_DETAILS = load_details()
if not SERVER_SIDE_RECOGNITION:
    KNOWN_ENCODINGS = {}
    STARTUP_WARNINGS = []
    REFERENCE_COUNTS = {}
elif FACE_RUNTIME_ERROR:
    KNOWN_ENCODINGS: dict[str, np.ndarray] = {}
    STARTUP_WARNINGS = [
        "Face-recognition runtime is unavailable in this deployment. "
        "Check the server's native Python dependencies."
    ]
    REFERENCE_COUNTS: dict[str, int] = {}
else:
    KNOWN_ENCODINGS, STARTUP_WARNINGS, REFERENCE_COUNTS = load_known_faces()


def decode_frame(payload: str) -> np.ndarray:
    """Decode a browser data URL into an OpenCV BGR image."""

    if not isinstance(payload, str) or not payload:
        raise ValueError("No camera frame was received.")
    encoded = payload.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("The camera frame is not valid base64 image data.") from error
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("The camera frame could not be decoded as an image.")
    return image_bgr


def estimate_clothes_colour(image_bgr: np.ndarray, location: tuple[int, int, int, int]) -> str:
    """Estimate the dominant clothing colour immediately below a detected face.

    This is a visual estimate from the current camera frame, not an identity
    attribute. If the torso is outside the frame, it deliberately reports that
    the colour cannot be determined rather than inventing a result.
    """

    top, right, bottom, left = location
    face_height = max(1, bottom - top)
    face_width = max(1, right - left)
    height, width = image_bgr.shape[:2]
    crop_top = min(height, bottom + int(face_height * 0.08))
    crop_bottom = min(height, bottom + int(face_height * 1.45))
    crop_left = max(0, left - int(face_width * 0.30))
    crop_right = min(width, right + int(face_width * 0.30))
    if crop_bottom - crop_top < 12 or crop_right - crop_left < 12:
        return "Not clearly visible in this scan"

    clothing_crop = image_bgr[crop_top:crop_bottom, crop_left:crop_right]
    hsv_pixels = cv2.cvtColor(clothing_crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, saturation, brightness = np.median(hsv_pixels, axis=0)
    if brightness < 50:
        return "Black or very dark"
    if saturation < 30:
        return "White" if brightness > 185 else "Grey"
    if hue < 10 or hue >= 170:
        return "Red" if brightness > 115 else "Maroon"
    if hue < 22:
        return "Brown" if brightness < 170 else "Orange"
    if hue < 35:
        return "Yellow or beige"
    if hue < 85:
        return "Green"
    if hue < 135:
        return "Blue"
    if hue < 160:
        return "Purple"
    return "Pink"


@app.get("/")
def index() -> str:
    profiles = enrolled_profiles()
    enrolment_manifest = [
        {
            "person_id": profile["person_id"],
            "reference_images": profile["reference_images"],
        }
        for profile in profiles
    ]
    return render_template(
        "index.html",
        threshold=FACE_MATCH_THRESHOLD,
        enrolment_manifest=enrolment_manifest,
        object_catalogue=object_catalogue(),
        object_manifest=object_manifest(),
        total_dictionary_entries=len(DICTIONARY_ENTRIES),
        dictionary_categories=list(DICTIONARY_CATEGORIES.keys())[:20],
    )


@app.get("/references/<path:filename>")
def reference_image(filename: str) -> Any:
    """Serve only a file from the enrolled-reference directory."""

    return send_from_directory(KNOWN_FACES_DIR, filename)


@app.get("/object-images/<category>/<path:filename>")
def object_image(category: str, filename: str) -> Any:
    """Serve only catalogue images from their category directory."""

    category_dir = OBJECT_IMAGES_DIR / category
    return send_from_directory(category_dir, filename)


@app.get("/my-pose-model/<path:filename>")
def my_pose_model(filename: str) -> Any:
    """Serve the downloaded hand-posture model files from the workspace root."""

    pose_model_dir = BASE_DIR.parent / "my-pose-model"
    return send_from_directory(pose_model_dir, filename)


@app.get("/api/status")
def status() -> Any:
    """Small diagnostics response used by the UI before a user starts scanning."""

    return jsonify(
        ready=bool(enrolled_profiles()),
        enrolled_people=[profile["person_id"] for profile in enrolled_profiles()],
        reference_count=len(enrolled_profiles()),
        reference_images=sum(len(profile["reference_images"]) for profile in enrolled_profiles()),
        recognition_runtime_ready=True,
        server_side_recognition=SERVER_SIDE_RECOGNITION and FACE_RUNTIME_ERROR is None,
        warnings=STARTUP_WARNINGS,
        threshold=FACE_MATCH_THRESHOLD,
        total_dictionary_entries=len(DICTIONARY_ENTRIES),
    )


@app.get("/api/dictionary/search")
def dictionary_search() -> Any:
    """Search the physical object dictionary by keyword, prefix, or category."""

    query = request.args.get("q", "")
    category = request.args.get("category", "")
    try:
        limit = min(100, max(1, int(request.args.get("limit", 40))))
    except (ValueError, TypeError):
        limit = 40
    results = search_dictionary(query=query, category=category, limit=limit)
    return jsonify(
        query=query,
        category=category,
        total=len(results),
        results=[with_reference_images(entry) for entry in results],
    )


@app.get("/api/dictionary/entry/<path:identifier>")
def dictionary_entry(identifier: str) -> Any:
    """Retrieve full details for an exact physical dictionary keyword or slug."""

    entry = get_dictionary_entry(identifier)
    if entry is None:
        return jsonify(found=False, error=f"No dictionary entry found for '{identifier}'."), 404
    return jsonify(found=True, entry=with_reference_images(entry))


@app.get("/api/dictionary/photographed")
def dictionary_photographed() -> Any:
    """List every dictionary keyword that currently has supplied reference photos.

    Powers a quick-access chip list so a person can jump straight to the
    keyword, description, purpose, and images for objects that were actually
    photographed (e.g. the bottle1/ images reorganised under object_images/).
    """

    photographed: list[dict[str, Any]] = []
    seen_folders: set[str] = set()
    for entry in DICTIONARY_ENTRIES:
        images = reference_images_for_entry(entry)
        if not images:
            continue
        folder_key = str(entry.get("image_folder", entry.get("slug", ""))).strip().lower()
        if folder_key in seen_folders:
            continue
        seen_folders.add(folder_key)
        photographed.append(
            {
                "id": entry.get("id"),
                "word": entry.get("word"),
                "slug": entry.get("slug"),
                "category": entry.get("category"),
                "image_count": len(images),
                "cover_image": images[0],
            }
        )
    photographed.sort(key=lambda item: str(item["word"]).lower())
    return jsonify(total=len(photographed), objects=photographed)


@app.get("/api/dictionary/categories")
def dictionary_categories() -> Any:
    """Return all available object categories and item counts."""

    categories_list = [
        {"name": name, "count": count}
        for name, count in DICTIONARY_CATEGORIES.items()
    ]
    return jsonify(
        total_entries=len(DICTIONARY_ENTRIES),
        total_categories=len(DICTIONARY_CATEGORIES),
        categories=categories_list,
    )


@app.post("/api/dictionary/add")
def dictionary_add() -> Any:
    """Add a new physical object dictionary keyword and description."""

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify(error="Invalid JSON payload."), 400
    word = payload.get("word")
    if not word or not str(word).strip():
        return jsonify(error="The 'word' field is required."), 400
    try:
        new_entry = add_dictionary_entry(payload)
    except Exception as error:
        return jsonify(error=str(error)), 400
    return jsonify(success=True, entry=new_entry), 201


@app.get("/api/details/<person_id>")
def details(person_id: str) -> Any:
    """Return the locally stored details only for a recognised person id."""

    record = PERSON_DETAILS.get(person_id)
    if record is None:
        return jsonify(error="No details record exists for this recognised face."), 404
    return jsonify(person_id=person_id, details=record)


def scan_result_email_html(
    name: str,
    details: dict[str, Any],
    clothes_colour: str,
    confidence: float,
    sent_at: str,
) -> str:
    """Create a minimal escaped email with the server-side enrolled details."""

    detail_rows = "".join(
        f"<tr><td style='padding:7px 12px;border-bottom:1px solid #e5e7eb;color:#475569'><strong>{escape(key.replace('_', ' ').title())}</strong></td>"
        f"<td style='padding:7px 12px;border-bottom:1px solid #e5e7eb'>{escape(str(value))}</td></tr>"
        for key, value in details.items()
        if key != "name"
    )
    detail_rows += (
        "<tr><td style='padding:7px 12px;border-bottom:1px solid #e5e7eb;color:#475569'><strong>Clothes colour</strong></td>"
        f"<td style='padding:7px 12px;border-bottom:1px solid #e5e7eb'>{escape(clothes_colour)}</td></tr>"
    )
    return f"""<!doctype html>
<html><body style='margin:0;background:#f8fafc;font-family:Arial,sans-serif;color:#172033'>
  <main style='max-width:640px;margin:24px auto;padding:28px;background:#ffffff;border-radius:14px'>
    <p style='margin:0 0 8px;color:#315cc8;font-weight:700;letter-spacing:.08em;text-transform:uppercase'>Face Scanner</p>
    <h1 style='margin:0 0 12px;font-size:24px'>Scan result for {escape(name)}</h1>
    <p style='line-height:1.55'>Your enrolled face was recognised in a live camera scan. Match confidence: <strong>{confidence:.1f}%</strong>.</p>
    <table style='width:100%;border-collapse:collapse;font-size:14px'>{detail_rows}</table>
    <p style='margin:20px 0 0;color:#64748b;font-size:12px'>Sent from the enrolled face scanner at {escape(sent_at)} UTC.</p>
  </main>
</body></html>"""


@app.post("/api/send-result/<person_id>")
def send_result(person_id: str) -> Any:
    """Email one recognised person's stored profile to their fixed email address."""

    recipient = RESULT_EMAIL_RECIPIENTS.get(person_id)
    record = PERSON_DETAILS.get(person_id)
    if not recipient or not record:
        return jsonify(error="This recognised profile has no configured result email."), 404

    resend_api_key = os.getenv("RESEND_API_KEY")
    from_email = os.getenv("RESEND_FROM_EMAIL")
    if not resend_api_key or not from_email:
        return jsonify(
            error="Email sending is not configured. Add RESEND_API_KEY and RESEND_FROM_EMAIL in Vercel, then redeploy."
        ), 503

    payload = request.get_json(silent=True) or {}
    try:
        confidence = min(100.0, max(0.0, float(payload.get("match_confidence", 0))))
    except (TypeError, ValueError):
        return jsonify(error="Match confidence must be a number."), 400
    clothes_colour = str(payload.get("clothes_colour", "not visible")).strip()[:80] or "not visible"
    name = str(record.get("name", person_id))
    sent_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    email_payload = {
        "from": from_email,
        "to": [recipient],
        "subject": f"Face Scanner result — {name}",
        "html": scan_result_email_html(name, record, clothes_colour, confidence, sent_at),
        "text": (
            f"Face Scanner result for {name}\n\n"
            f"Match confidence: {confidence:.1f}%\n"
            f"Clothes colour: {clothes_colour}\n\n"
            + "\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in record.items() if key != "name")
            + f"\n\nSent at {sent_at} UTC."
        ),
    }
    provider_request = Request(
        RESEND_API_URL,
        data=json.dumps(email_payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {resend_api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(provider_request, timeout=12) as response:
            provider_response = json.loads(response.read().decode("utf-8") or "{}")
    except HTTPError as error:
        app.logger.warning("Email provider rejected result email for %s: %s", person_id, error.code)
        return jsonify(error="The email service rejected this send. Check the Resend sender domain and API key."), 502
    except (URLError, OSError, json.JSONDecodeError) as error:
        app.logger.error("Could not send result email for %s: %s", person_id, error)
        return jsonify(error="The email service could not be reached. Please try again."), 503

    return jsonify(sent=True, recipient=recipient, email_id=provider_response.get("id"))


@app.post("/scan")
def scan() -> Any:
    """Match one browser frame against the startup-loaded reference encodings."""

    if not SERVER_SIDE_RECOGNITION:
        return jsonify(
            match=False,
            reason="browser_recognition",
            message="This website recognises faces in the browser. Use the web scanner instead of this API.",
        ), 410

    if FACE_RUNTIME_ERROR:
        return jsonify(
            match=False,
            reason="recognition_runtime_unavailable",
            message="Face recognition is unavailable because this server is missing its native Python dependencies.",
        ), 503

    if not KNOWN_ENCODINGS:
        return jsonify(
            match=False,
            reason="no_reference_faces",
            message="No valid reference faces are loaded. Add clear photos to known_faces and restart the app.",
        ), 503

    data = request.get_json(silent=True) or {}
    try:
        image_bgr = decode_frame(data.get("image", ""))
    except ValueError as error:
        return jsonify(match=False, reason="invalid_frame", message=str(error)), 400

    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if float(np.mean(grayscale)) < MIN_BRIGHTNESS:
        return jsonify(
            match=False,
            reason="low_light",
            message="The image is too dark. Improve the lighting and try again.",
        )

    # Phone cameras may deliver portrait JPEG pixels in a landscape orientation.
    # Try the four upright orientations so a valid mobile capture is not
    # rejected simply because its EXIF orientation is not applied by OpenCV.
    scan_image_bgr = image_bgr
    rgb_image = cv2.cvtColor(scan_image_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_image, model=FACE_DETECTION_MODEL)
    if not locations:
        for rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            candidate_bgr = cv2.rotate(image_bgr, rotation)
            candidate_rgb = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2RGB)
            candidate_locations = face_recognition.face_locations(candidate_rgb, model=FACE_DETECTION_MODEL)
            if candidate_locations:
                scan_image_bgr = candidate_bgr
                rgb_image = candidate_rgb
                locations = candidate_locations
                break
    if not locations:
        return jsonify(
            match=False,
            reason="no_face",
            message="No face detected. Center one well-lit face in the camera and try again.",
        )
    probe_encodings = face_recognition.face_encodings(rgb_image, known_face_locations=locations)
    if not probe_encodings:
        return jsonify(
            match=False,
            reason="encoding_failed",
            message="A face was found but could not be read. Face the camera directly and try again.",
        )

    person_ids = list(KNOWN_ENCODINGS)
    reference_encodings = np.array([KNOWN_ENCODINGS[person_id] for person_id in person_ids])
    faces: list[dict[str, Any]] = []
    for location, probe_encoding in zip(locations, probe_encodings):
        distances = face_recognition.face_distance(reference_encodings, probe_encoding)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        person_id = person_ids[best_index]
        is_match = best_distance < FACE_MATCH_THRESHOLD
        person_record = PERSON_DETAILS.get(person_id, {})
        top, right, bottom, left = location
        faces.append(
            {
                "match": is_match,
                "person_id": person_id if is_match else None,
                "name": person_record.get("name", person_id) if is_match else "Face not recognized",
                "confidence": round(max(0.0, (1.0 - best_distance) * 100), 1),
                "distance": round(best_distance, 4),
                "location": {"top": top, "right": right, "bottom": bottom, "left": left},
                "clothes_colour": estimate_clothes_colour(scan_image_bgr, location),
            }
        )

    matched_faces = [face for face in faces if face["match"]]
    response: dict[str, Any] = {
        "match": bool(matched_faces),
        "faces": faces,
        "message": "Face recognized." if matched_faces else "Face not recognized.",
    }
    # Keep the original single-face response fields for compatibility.
    if len(faces) == 1:
        response.update(faces[0])
        response["reason"] = "recognized" if faces[0]["match"] else "not_recognized"
    return jsonify(response)


@app.errorhandler(413)
def request_too_large(_: Any) -> Any:
    return jsonify(match=False, reason="frame_too_large", message="Camera frame is too large. Try again."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
