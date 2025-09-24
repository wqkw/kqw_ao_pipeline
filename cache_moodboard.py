#!/usr/bin/env python3
import os
import psycopg
import pandas as pd
from dotenv import load_dotenv
import requests
from PIL import Image
import io

load_dotenv()

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
        SELECT g.id, g."s3Key"
        FROM public."Gen" AS g
        JOIN public."MoodboardCollectionItem" AS mci
        ON mci."genId" = g.id
        WHERE mci."moodboardId" = %s;
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


if __name__ == "__main__":
    df = query_moodboard_collection_item()
    srs = df.set_index('id')['s3Key']

    for index, s3Key in srs.items():
        print(index)
        image = get_image_from_s3(s3Key)
        image.save(f"data/ref_moodboard/{index}.png")


