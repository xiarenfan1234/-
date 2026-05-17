import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def read_gray_image(image_path):
    """
    Read an image and convert it to grayscale.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def resize_to_reference(image_gray, reference_gray):
    """
    Resize generated image to the same resolution as the original image.
    """
    ref_h, ref_w = reference_gray.shape[:2]

    resized = cv2.resize(
        image_gray,
        (ref_w, ref_h),
        interpolation=cv2.INTER_AREA
    )

    return resized


def calculate_ssim(original_gray, generated_gray):
    """
    Calculate SSIM between the original grayscale image and generated pencil image.
    """
    value = ssim(original_gray, generated_gray, data_range=255)
    return float(value)


def calculate_canny_edges(gray_img, low_threshold=100, high_threshold=200):
    """
    Extract edge map by Canny detector.
    The same thresholds must be used for all methods.
    """
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)
    return edges


def calculate_ecs(original_edges, generated_edges):
    """
    Calculate Edge Consistency Score.

    ECS = 2 * |Eo ∩ Es| / (|Eo| + |Es|)
    """
    original_binary = original_edges > 0
    generated_binary = generated_edges > 0

    intersection = np.logical_and(original_binary, generated_binary).sum()
    original_count = original_binary.sum()
    generated_count = generated_binary.sum()

    denominator = original_count + generated_count

    if denominator == 0:
        return 0.0

    ecs_value = 2.0 * intersection / denominator
    return float(ecs_value)


def calculate_gray_entropy(gray_img):
    """
    Calculate gray entropy.

    H = -sum(p_i * log2(p_i))
    """
    hist = cv2.calcHist(
        [gray_img],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten()

    probabilities = hist / hist.sum()
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def calculate_edge_density(edges):
    """
    Calculate edge density.

    Edge density = edge pixels / total pixels
    """
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]

    density = edge_pixels / total_pixels
    return float(density)


def evaluate_one_method(
    original_gray,
    original_edges,
    generated_path,
    method_name,
    canny_low=100,
    canny_high=200
):
    """
    Evaluate one generated pencil drawing result.
    """
    generated_gray = read_gray_image(generated_path)
    generated_gray = resize_to_reference(generated_gray, original_gray)

    generated_edges = calculate_canny_edges(
        generated_gray,
        low_threshold=canny_low,
        high_threshold=canny_high
    )

    ssim_value = calculate_ssim(original_gray, generated_gray)
    ecs_value = calculate_ecs(original_edges, generated_edges)
    entropy_value = calculate_gray_entropy(generated_gray)
    edge_density_value = calculate_edge_density(generated_edges)

    return {
        "Method": method_name,
        "SSIM": round(ssim_value, 3),
        "ECS": round(ecs_value, 3),
        "Gray entropy (bits)": round(entropy_value, 3),
        "Edge density": f"{edge_density_value * 100:.2f}%"
    }


def print_table(results):
    """
    Print result table without using pandas.
    """
    headers = [
        "Method",
        "SSIM",
        "ECS",
        "Gray entropy (bits)",
        "Edge density"
    ]

    col_widths = {}

    for header in headers:
        col_widths[header] = len(header)

    for row in results:
        for header in headers:
            col_widths[header] = max(
                col_widths[header],
                len(str(row[header]))
            )

    header_line = " | ".join(
        header.ljust(col_widths[header]) for header in headers
    )

    separator = "-+-".join(
        "-" * col_widths[header] for header in headers
    )

    print(header_line)
    print(separator)

    for row in results:
        line = " | ".join(
            str(row[header]).ljust(col_widths[header])
            for header in headers
        )
        print(line)


def save_to_csv(results, output_path):
    """
    Save the result table to CSV.
    """
    headers = [
        "Method",
        "SSIM",
        "ECS",
        "Gray entropy (bits)",
        "Edge density"
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def save_edge_maps(
    original_gray,
    original_edges,
    generated_images,
    output_dir,
    canny_low=100,
    canny_high=200
):
    """
    Optional: save edge maps for checking.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "original_edges.png"), original_edges)

    for method_name, image_path in generated_images.items():
        gray = read_gray_image(image_path)
        gray = resize_to_reference(gray, original_gray)

        edges = calculate_canny_edges(
            gray,
            low_threshold=canny_low,
            high_threshold=canny_high
        )

        safe_name = method_name.lower().replace(" ", "_").replace("/", "_")
        cv2.imwrite(str(output_dir / f"{safe_name}_edges.png"), edges)


def main():

    original_path = "images/original.jpg"


    generated_images = {
        "Traditional method": "images/traditional.jpg",
        "First deep learning model": "images/first_deep_learning.jpg",
        "Second optimized deep learning model": "images/second_optimized_deep_learning.jpg",
        "Hybrid 72/28": "images/hybrid_72_28.jpg",
        "Hybrid 30/70": "images/hybrid_30_70.jpg"
    }


    canny_low = 100
    canny_high = 200


    original_gray = read_gray_image(original_path)

    original_edges = calculate_canny_edges(
        original_gray,
        low_threshold=canny_low,
        high_threshold=canny_high
    )


    results = []

    for method_name, image_path in generated_images.items():
        print(f"Evaluating: {method_name}")

        result = evaluate_one_method(
            original_gray=original_gray,
            original_edges=original_edges,
            generated_path=image_path,
            method_name=method_name,
            canny_low=canny_low,
            canny_high=canny_high
        )

        results.append(result)


    print("\nQuantitative comparison of the five processing methods:\n")
    print_table(results)


    save_to_csv(results, "quantitative_results.csv")
    print("\nResults saved to quantitative_results.csv")

    save_edge_maps(
        original_gray=original_gray,
        original_edges=original_edges,
        generated_images=generated_images,
        output_dir="edge_maps",
        canny_low=canny_low,
        canny_high=canny_high
    )

    print("Edge maps saved to edge_maps folder")


if __name__ == "__main__":
    main()
