#!/usr/bin/env python3
import os
import psycopg
import pandas as pd
from dotenv import load_dotenv
import requests
from PIL import Image
import io
import glob
import boto3
from datetime import datetime
import math

load_dotenv()

# Local paths
REF_MOODBOARD_DIR = "data/ref_moodboard"
MOODBOARD_DATA_CSV = "data/ref_moodboard/moodboard_data.csv"
MOODBOARD_TILE_PATH = "data/ref_moodboard/moodboard_tile.png"

def query_moodboard_collection_item(moodboardId='cmfeyqgsm000yl1046yzyi9rw'):
    try:
        conn = psycopg.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )

        cursor = conn.cursor()

        query = """
        SELECT
            g.id,
            g."s3Key",
            COALESCE(ARRAY_TO_STRING(ARRAY_AGG(gt."B"), ','), '') AS tags
        FROM public."Gen" AS g
        JOIN public."MoodboardCollectionItem" AS mci
        ON mci."genId" = g.id
        LEFT JOIN public."_GenTags" AS gt
        ON gt."A" = g.id
        WHERE mci."moodboardId" = %s
        GROUP BY g.id, g."s3Key";
        """

        cursor.execute(query, (moodboardId,))
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

        cursor.close()
        conn.close()

        df = pd.DataFrame(results, columns=column_names)
        return df

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def get_image_from_s3(s3_key):
    url = f"https://artofficial-metagen.s3.us-east-1.amazonaws.com/{s3_key}"
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def create_moodboard_tile():
    input_dir = REF_MOODBOARD_DIR
    output_path = MOODBOARD_TILE_PATH

    # Get all PNG files from the input directory
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    # Exclude the moodboard_tile.png itself if it exists
    image_files = [f for f in image_files if not f.endswith("moodboard_tile.png")]

    if not image_files:
        print(f"No PNG files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Calculate grid dimensions for approximately 3:2 ratio
    num_images = len(image_files)

    # Find optimal grid dimensions
    # Start with square root and adjust to get close to 3:2 ratio (2 wide, 3 tall)
    # For 3:2 ratio, we want cols/rows ≈ 2/3, so rows should be about 1.5 * cols
    sqrt_images = math.sqrt(num_images)

    # Try different column counts around the square root
    best_cols = max(1, int(sqrt_images))
    best_rows = (num_images + best_cols - 1) // best_cols
    best_diff = float('inf')
    target_ratio = 2/3  # cols/rows ratio we want

    for cols in range(max(1, int(sqrt_images * 0.5)), min(num_images, int(sqrt_images * 2)) + 1):
        rows = (num_images + cols - 1) // cols  # Ceiling division
        current_ratio = cols / rows
        diff = abs(current_ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_cols = cols
            best_rows = rows

    cols = best_cols
    rows = best_rows
    print(f"Using {cols}x{rows} grid for {num_images} images")

    # Load and resize images to reduce file size
    images = []
    target_width = 400  # Reduced from original size

    for img_path in image_files:  # Use all images
        try:
            with Image.open(img_path) as img:
                # Verify image can be loaded
                img.load()

                # Calculate height to maintain aspect ratio
                aspect_ratio = img.height / img.width
                target_height = int(target_width * aspect_ratio)

                # Resize image
                resized_img = img.resize((target_width, target_height), Image.LANCZOS)
                images.append(resized_img)
        except (OSError, IOError) as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            continue

    if not images:
        print("No valid images found after filtering corrupted files")
        return None

    print(f"Successfully loaded {len(images)} valid images")

    # Calculate dimensions for the tiled image
    separator_width = 20

    # Assume all images have similar dimensions (use first image as reference)
    img_width = images[0].width
    img_height = images[0].height

    # Calculate total dimensions
    total_width = (cols * img_width) + ((cols - 1) * separator_width)
    total_height = (rows * img_height) + ((rows - 1) * separator_width)

    # Create the final tiled image with black background
    tiled_image = Image.new('RGB', (total_width, total_height), color='black')

    # Place images in grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        x = col * (img_width + separator_width)
        y = row * (img_height + separator_width)

        tiled_image.paste(img, (x, y))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the tiled image
    tiled_image.save(output_path, 'PNG', optimize=True)
    print(f"Tiled moodboard saved to {output_path}")
    print(f"Final dimensions: {total_width}x{total_height}")

    return output_path


def upload_to_s3(file_path):
    """Upload file to S3 bucket under kqwtest subfolder and write S3 path to txt file"""

    # Check for AWS credentials
    if not (os.getenv('AWS_ACCESS_KEY_ID') or os.getenv('AWS_PROFILE')):
        print("AWS credentials not found. Please add to .env file:")
        print("AWS_ACCESS_KEY_ID=your_access_key")
        print("AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("AWS_DEFAULT_REGION=us-east-1")
        print("\nOr configure AWS CLI with 'aws configure'")
        return None

    try:
        s3_client = boto3.client('s3')
        bucket_name = 'artofficial-metagen'

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"moodboard_tile_{timestamp}.png"
        s3_key = f"kqwtest/{filename}"

        # Upload file to S3
        s3_client.upload_file(file_path, bucket_name, s3_key)
        s3_path = f"https://{bucket_name}.s3.us-east-1.amazonaws.com/{s3_key}"
        print(f"Successfully uploaded to S3: {s3_path}")

        # Write S3 path to txt file in data/
        os.makedirs("data", exist_ok=True)
        txt_path = f"data/s3_path_{timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write(s3_path)
        print(f"S3 path written to: {txt_path}")

        return s3_path
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None


if __name__ == "__main__":
    df = query_moodboard_collection_item()

    file_paths = []
    for idx, row in df.iterrows():
        gen_id = row['id']
        s3Key = row['s3Key']
        tags = row['tags']
        print(f"{gen_id}: {s3Key}, tags: {tags}")

        file_path = f"{REF_MOODBOARD_DIR}/{gen_id}.png"
        image = get_image_from_s3(s3Key)
        image.save(file_path)
        file_paths.append(file_path)

    df['file_path'] = file_paths
    df.to_csv(MOODBOARD_DATA_CSV, index=False)

    # Create moodboard tile from downloaded images
    create_moodboard_tile()


