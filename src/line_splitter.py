from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw

def find_word_candidates(ink_mask, join_width=9):
    binary = ink_mask.astype(np.uint8)

    kernel = np.ones((1, join_width), dtype=np.uint8)
    joined = cv2.dilate(binary, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        joined,
        connectivity=8,
    )

    candidates = []


    for label in range(1, count):
        left, top, width, height, _ = stats[label]

        local_labels = labels[
            top:top + height,
            left:left + width,
        ]

        original_ink = ink_mask[
            top:top + height,
            left:left + width,
        ]

        member_ink = (local_labels == label) & original_ink
        ys, xs = np.nonzero(member_ink)

        if len(xs) == 0:
            continue

        xs = xs + left
        ys = ys + top

        candidates.append({
            "label": int(label),
            "box": (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ),
            "center": (
                float(xs.mean()),
                float(ys.mean()),
            ),
            "ink_pixels": len(xs),
        })

    return candidates

def flag_tall_candidates(candidates):
    if len(candidates) < 4:
        return

    preliminary_height = float(np.median([
        candidate["box"][3] - candidate["box"][1]
        for candidate in candidates
    ]))

    substantial = [
        candidate
        for candidate in candidates
        if candidate["ink_pixels"] >= 40
        and candidate["box"][3] - candidate["box"][1]
        >= 0.5 * preliminary_height
    ]

    if len(substantial) < 4:
        return

    global_height = float(np.median([
        candidate["box"][3] - candidate["box"][1]
        for candidate in substantial
    ]))

    for candidate in substantial:
        left, top, right, bottom = candidate["box"]
        center_x, center_y = candidate["center"]
        box_height = bottom - top

        neighbors = [ other for other in substantial
            if other is not candidate
            and abs(other["center"][1] - center_y) <= 2 * global_height
        ]
        def horizontal_distance(other):
            return abs(other["center"][0] - center_x)

        neighbors.sort(key=horizontal_distance)
        neighbors = neighbors[:8]

        if len(neighbors) >= 3:
            expected_height = float(np.median([
                other["box"][3] - other["box"][1]
                for other in neighbors
            ]))
        else:
            expected_height = global_height

        height_ratio = box_height / expected_height

        candidate["expected_height"] = expected_height
        candidate["height_ratio"] = height_ratio
        candidate["possibly_merged"] = height_ratio > 1.6

def estimate_line_tracks(candidates, bands, width):
    centers = np.array([
        (top + bottom) / 2
        for top, bottom in bands
    ], dtype=float)

    anchors = [ candidate for candidate in candidates
        if candidate["ink_pixels"] >= 40
        and candidate["box"][2] - candidate["box"][0] >= 12
        and not candidate.get("possibly_merged", False)
    ]

    anchors = [
        candidate
        for candidate in candidates
        if "expected_height" in candidate
        and not candidate.get("possibly_merged", False)
    ]

    slopes = np.zeros(len(bands), dtype=float)
    intercepts = centers.copy()

    if not anchors:
        return (
            slopes[:, None] * np.arange(width)[None, :]
            + intercepts[:, None]
        )

    points = np.array([
        candidate["center"] for candidate in anchors
    ], dtype=float)

    xs = points[:, 0]
    ys = points[:, 1]

    for iteration in range(5):
        predicted = (
            slopes[:, None] * xs[None, :]
            + intercepts[:, None]
        )

        distances = np.abs(predicted - ys[None, :])
        assignments = np.argmin(distances, axis=0)

        next_slopes = slopes.copy()
        next_intercepts = intercepts.copy()

        for line_index in range(len(bands)):
            selected = assignments == line_index
            line_xs = xs[selected]
            line_ys = ys[selected]

            if len(line_xs) < 3:
                continue

            pair_slopes = []

            for first in range(len(line_xs)):
                for second in range(first + 1, len(line_xs)):
                    dx = line_xs[second] - line_xs[first]

                    if abs(dx) >= width * 0.1:
                        pair_slopes.append(
                            (line_ys[second] - line_ys[first]) / dx
                        )

            if not pair_slopes:
                continue

            slope = float(np.median(pair_slopes))

            slope = float(np.clip(slope, -0.25, 0.25))
            intercept = float(np.median(line_ys - slope * line_xs))

            next_slopes[line_index] = slope
            next_intercepts[line_index] = intercept

        proposed_tracks = (
            next_slopes[:, None] * np.arange(width)[None, :]
            + next_intercepts[:, None]
        )

        if np.any(np.diff(proposed_tracks, axis=0) <= 0):
            break

        slopes = next_slopes
        intercepts = next_intercepts

    return (
        slopes[:, None] * np.arange(width)[None, :]
        + intercepts[:, None]
    )


def assign_word_owners(ink_mask, candidates, tracks):
    """Assign only unambiguous word groups; -1 means leave unchanged."""
    joined = cv2.dilate(
        ink_mask.astype(np.uint8),
        np.ones((1, 9), dtype=np.uint8),
        iterations=1,
    )
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        joined, connectivity=8,
    )
    owners = np.full(count, -1, dtype=np.int32)

    if len(tracks) < 2:
        return owners[labels]

    for candidate in candidates:
        # Height flags are imperfect, so uncertain groups keep the
        # existing curved-boundary treatment (including quite/fill).
        if candidate.get("possibly_merged", False):
            continue
        if candidate["ink_pixels"] < 40:
            continue

        label = candidate["label"]
        left, top, right, bottom = candidate["box"]
        member = (
            (labels[top:bottom, left:right] == label)
            & ink_mask[top:bottom, left:right]
        )
        ys, xs = np.nonzero(member)
        xs = xs + left
        ys = ys + top

        distances = np.median(
            np.abs(tracks[:, xs] - ys[None, :]), axis=1,
        )
        nearest = np.argsort(distances)
        # Require a clear preference, not just the nearest track.
        if distances[nearest[0]] < 0.5 * distances[nearest[1]]:
            owners[label] = nearest[0]

    return owners[labels]

def trim_line_margins(line_image):
    raise NotImplementedError

def split_lines(image_path):
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    detection_image = image.copy()
    detection_image.thumbnail((1600, 1600))

    grayscale = ImageOps.grayscale(detection_image)
    width, height = grayscale.size

    background = grayscale.filter(
        ImageFilter.GaussianBlur(radius=15)
    )

    gray_values = np.asarray(grayscale, dtype=np.int16)
    background_values = np.asarray(background, dtype=np.int16)

    ink_mask = (background_values - gray_values) > 12
    original_ink_mask = ink_mask.copy()

    edge_margin = max(1, int(width * 0.03))
    ink_mask[:, :edge_margin] = False
    ink_mask[:, -edge_margin:] = False


    candidates = find_word_candidates(ink_mask)
    flag_tall_candidates(candidates)
    print(f"Found {len(candidates)} ink groups.")
    for candidate in candidates[:10]:
        print(candidate)

    # Draw on a separate copy of the detection image.
    preview = detection_image.copy()
    draw = ImageDraw.Draw(preview)

    for number, candidate in enumerate(candidates, start=1):
        left, top, right, bottom = candidate["box"]
        center_x, center_y = candidate["center"]

        suspicious = candidate.get("possibly_merged", False)
        color = "orange" if suspicious else "red"

        draw.rectangle(
            (left, top, right - 1, bottom - 1),
            outline=color,
            width=3 if suspicious else 1,
        )
        draw.ellipse(
            (
                center_x - 3,
                center_y - 3,
                center_x + 3,
                center_y + 3,
            ),
            fill="blue",
        )


        label = str(number)

        if suspicious:
            ratio = candidate["height_ratio"]
            label += f" ({ratio:.1f}x)"

            print(
                f"Check group {number}: "
                f"height={bottom - top}, "
                f"expected={candidate['expected_height']:.1f}, "
                f"ratio={ratio:.2f}"
            )

        draw.text(
            (left, max(0, top - 12)),
            label,
            fill=color,
        )

    output_folder = image_path.parent.parent / "Output"
    output_folder.mkdir(parents=True, exist_ok=True)

    preview_path = output_folder / f"{image_path.stem}_word_candidates.png"
    preview.save(preview_path)

    print(f"Word candidate preview saved to: {preview_path}")


    ink_per_row = ink_mask.sum(axis=1)

    minimum_ink = 30
    text_rows = ink_per_row >= minimum_ink

    allowed_gap = 3
    minimum_height = 8

    bands = []
    start = None
    last_ink_row = None

    for y, contains_ink in enumerate(text_rows):
        if contains_ink:
            if start is None:
                start = y

            last_ink_row = y

        elif start is not None and y - last_ink_row > allowed_gap:
            end = last_ink_row + 1

            if end - start >= minimum_height:
                bands.append((start, end))

            start = None

    # Finish a band that reaches the bottom of the image.
    if start is not None:
        end = last_ink_row + 1

        if end - start >= minimum_height:
            bands.append((start, end))

    # Translate detection coordinates back to original-image coordinates.
    if not bands:
        return []

    tracks = estimate_line_tracks(candidates, bands, width)
    word_owners = assign_word_owners(ink_mask, candidates, tracks)

    # Make crossing ink expensive, including a small safety margin
    # around each stroke. This affects detection, not the source image.
    ink_image = Image.fromarray(
        original_ink_mask.astype(np.uint8) * 255
    )
    expanded_ink = np.asarray(
        ink_image.filter(ImageFilter.MaxFilter(3))
    ) > 0

    def find_boundary(upper_track, lower_track):
        # Desired boundary follows the gap between two sloping tracks.
        preferred_y = (upper_track + lower_track) / 2

        search_top = max(0, int(np.floor(upper_track.min())))
        search_bottom = min(
            height - 1,
            int(np.ceil(lower_track.max())),
        )

        row_positions = np.arange(search_top, search_bottom + 1)
        search_height = len(row_positions)

        costs = (
            expanded_ink[search_top:search_bottom + 1].astype(float)
            * 100.0
        )

        # Guide the boundary toward the estimated inter-line gap.
        costs += (
            0.2
            * np.abs(row_positions[:, None] - preferred_y[None, :])
        )

        # Keep the path between the two tracks at each column.
        outside = (
            (row_positions[:, None] < upper_track[None, :])
            | (row_positions[:, None] > lower_track[None, :])
        )
        costs[outside] = np.inf

        previous_cost = costs[:, 0].copy()
        parents = np.zeros((search_height, width), dtype=np.int32)
        row_indices = np.arange(search_height)

        # A boundary can stay level or move one pixel up/down
        # between neighbouring columns.
        for x in range(1, width):
            from_above = np.full(search_height, np.inf)
            from_below = np.full(search_height, np.inf)

            from_above[1:] = previous_cost[:-1] + 0.2
            from_below[:-1] = previous_cost[1:] + 0.2

            choices = np.stack([
                from_above,
                previous_cost,
                from_below,
            ])

            best_choice = np.argmin(choices, axis=0)
            parents[:, x] = row_indices + best_choice - 1

            previous_cost = (
                costs[:, x] + choices[best_choice, row_indices]
            )

        # Trace the cheapest path backwards.
        boundary = np.empty(width, dtype=np.int32)
        row = int(np.argmin(previous_cost))

        for x in range(width - 1, -1, -1):
            boundary[x] = search_top + row
            if x > 0:
                row = parents[row, x]

        return boundary

    boundaries = [np.zeros(width, dtype=np.int32)]

    for index in range(len(bands) - 1):
        boundaries.append(
            find_boundary(tracks[index], tracks[index + 1])
        )

    boundaries.append(np.full(width, height, dtype=np.int32))

    # Scale the curved boundaries back to the original resolution.
    scale_y = image.height / height
    original_x = (
        (np.arange(image.width) + 0.5) * width / image.width - 0.5
    )

    full_boundaries = [
        np.clip(
            np.rint(
                np.interp(original_x, np.arange(width), boundary)
                * scale_y
            ).astype(int),
            0,
            image.height,
        )
        for boundary in boundaries
    ]

    source_pixels = np.asarray(image)
    detection_rows = np.arange(height)[:, None]
    line_images = []
    padding = max(4, round(6 * scale_y))

    for index in range(len(bands)):
        # Find ink belonging to this region, using the unmodified mask.
        region = (
            (detection_rows >= boundaries[index][None, :])
            & (detection_rows < boundaries[index + 1][None, :])
        )

        # Use the edge-filtered mask for the bounding box so the
        # page border does not expand the crop to the entire page.
        # Confident ownership can recover strokes across a boundary.
        owned = word_owners == index
        foreign = (word_owners >= 0) & (word_owners != index)
        region = (region | owned) & ~foreign
        ys, xs = np.nonzero(ink_mask & region)

        if len(xs) == 0:
            continue

        scale_x = image.width / width

        left = max(0, int(xs.min() * scale_x) - padding)
        right = min(
            image.width,
            int(np.ceil((xs.max() + 1) * scale_x)) + padding,
        )

        if xs.min() <= edge_margin + 2:
            left = 0

        top = max(0, int(ys.min() * scale_y) - padding)
        bottom = min(
            image.height,
            int(np.ceil((ys.max() + 1) * scale_y)) + padding,
        )

        crop_pixels = source_pixels[top:bottom, left:right].copy()
        crop_rows = np.arange(top, bottom)[:, None]

        belongs_to_line = (
            (crop_rows >= full_boundaries[index][None, left:right])
            & (
                crop_rows
                < full_boundaries[index + 1][None, left:right]
            )
        )

        # Preserve original pixels inside the line's region;
        # replace neighbouring text with white.
        owned_mask = Image.fromarray(owned.astype(np.uint8) * 255)
        owned_mask = owned_mask.filter(ImageFilter.MaxFilter(3))
        owned_mask = owned_mask.resize(
            image.size, resample=Image.Resampling.NEAREST,
        )
        owned_crop = np.asarray(
            owned_mask.crop((left, top, right, bottom))
        ) > 0
        belongs_to_line |= owned_crop
        crop_pixels[~belongs_to_line] = 255
        # Remove complete foreign word groups, including their faint
        # edges. Unassigned groups and small punctuation stay unchanged.
        foreign = (word_owners >= 0) & (word_owners != index)
        foreign_mask = Image.fromarray(foreign.astype(np.uint8) * 255)
        foreign_mask = foreign_mask.filter(ImageFilter.MaxFilter(3))
        foreign_mask = foreign_mask.resize(
            image.size, resample=Image.Resampling.NEAREST,
        )
        foreign_crop = np.asarray(
            foreign_mask.crop((left, top, right, bottom))
        ) > 0
        crop_pixels[foreign_crop] = 255
        line_images.append(Image.fromarray(crop_pixels))

    return line_images


def main():
    project_folder = Path(__file__).resolve().parent.parent
    source_folder = project_folder / "Source"

    filename = input("Image filename inside Source (e.g. NLS1.png): ").strip()

    image_path = source_folder / filename

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return

    line_images = split_lines(image_path)

    if not line_images:
        print("No text lines detected.")
        return

    # Each run gets a separate folder, avoiding stale crops or overwrites.
    output_base = project_folder / "Output" / "line_crops"
    output_base.mkdir(parents=True, exist_ok=True)

    run_number = 1

    while True:
        output_folder = output_base / f"{image_path.stem}_{run_number}"

        if not output_folder.exists():
            output_folder.mkdir()
            break

        run_number += 1

    for number, line_image in enumerate(line_images, start=1):
        output_path = output_folder / f"line_{number:02d}.png"
        line_image.save(output_path)

    print(f"Detected {len(line_images)} lines.")
    print(f"Crops saved to: {output_folder}")


if __name__ == "__main__":
    main()
