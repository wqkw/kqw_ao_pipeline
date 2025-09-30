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

    # Parse coordinates from filenames and load images
    image_data = []  # List of (row, col, image) tuples
    target_width = 400
    max_row = 0
    max_col = 0

    for img_path in image_files:
        try:
            # Extract filename from path
            filename = os.path.basename(img_path)
            # Parse coordinates from filename (format: XX_YY.png)
            coords = filename.replace('.png', '').split('_')
            if len(coords) != 2:
                print(f"Warning: Skipping file with invalid name format: {filename}")
                continue

            row = int(coords[0])
            col = int(coords[1])
            max_row = max(max_row, row)
            max_col = max(max_col, col)

            # Load and resize image
            with Image.open(img_path) as img:
                img.load()
                aspect_ratio = img.height / img.width
                target_height = int(target_width * aspect_ratio)
                resized_img = img.resize((target_width, target_height), Image.LANCZOS)
                image_data.append((row, col, resized_img))
                print(f"Loaded {filename} at position ({row}, {col})")

        except (OSError, IOError, ValueError) as e:
            print(f"Warning: Skipping corrupted or invalid image {img_path}: {e}")
            continue

    if not image_data:
        print("No valid images found after filtering corrupted files")
        return None

    # Grid dimensions based on actual coordinates
    rows = max_row + 1
    cols = max_col + 1
    print(f"Grid dimensions: {cols} cols x {rows} rows")

    # Calculate dimensions for the tiled image
    separator_width = 20

    # Use first image as reference for dimensions
    img_width = image_data[0][2].width
    img_height = image_data[0][2].height

    # Calculate total dimensions
    total_width = (cols * img_width) + ((cols - 1) * separator_width)
    total_height = (rows * img_height) + ((rows - 1) * separator_width)

    # Create the final tiled image with black background
    tiled_image = Image.new('RGB', (total_width, total_height), color='black')

    # Place images at their specified coordinates
    for row, col, img in image_data:
        x = col * (img_width + separator_width)
        y = row * (img_height + separator_width)

        tiled_image.paste(img, (x, y))
        print(f"Placed image at grid position ({row}, {col}) -> pixel position ({x}, {y})")

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
    # Create ref_moodboard directory if it doesn't exist
    os.makedirs(REF_MOODBOARD_DIR, exist_ok=True)

    df = query_moodboard_collection_item()
    df = df.dropna(how = 'any')

    # Calculate grid dimensions for approximately 3:2 ratio
    num_images = len(df)
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

    # Generate grid positions first
    grid_positions = []
    for i in range(len(df)):
        grid_row = i // cols
        grid_col = i % cols
        grid_positions.append(f"{grid_row:02d}_{grid_col:02d}")

    df['grid_position'] = grid_positions

    file_paths = []
    for position_idx, (idx, row) in enumerate(df.iterrows()):
        gen_id = row['id']
        s3Key = row['s3Key']
        tags = row['tags']
        grid_pos = row['grid_position']

        print(f"{gen_id}: {s3Key}, tags: {tags}, position: {grid_pos}")

        file_path = f"{REF_MOODBOARD_DIR}/{grid_pos}.png"
        image = get_image_from_s3(s3Key)
        image.save(file_path)
        file_paths.append(file_path)

    df['file_path'] = file_paths
    df.to_csv(MOODBOARD_DATA_CSV, index=False)

    # Create moodboard tile from downloaded images
    create_moodboard_tile()


