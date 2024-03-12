from google.cloud import storage
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize the GCS client
client = storage.Client()

# Define the buckets and base paths
source_infos = [
    ('lt-research-mm-datasets-us-east5', 'laion2b-en/sg40/llava-aestetic-0/'),
    ('lt-research-mm-datasets-us-east5', 'laion2b-en/sg40/llava-aestetic-4/'),
    ('lt-research-mm-datasets-us-east5', 'laion2b-en/sg40/llava-aestetic-7/')
]
output_bucket_name = 'lt-research-mm-datasets-us-east5'
output_base_path = 'laion2b-en/sg40/llava/'

# Ensure the output directory exists
output_bucket = client.get_bucket(output_bucket_name)

# Function to read a feather file from GCS
def read_feather_from_gcs(bucket_name, full_path):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(full_path)
    content = blob.download_as_bytes()
    return pd.read_feather(BytesIO(content))

# Function to write a DataFrame to a feather file in GCS
def write_feather_to_gcs(df, bucket_name, full_path):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(full_path)
    buffer = BytesIO()
    df.to_feather(buffer)
    buffer.seek(0)
    blob.upload_from_file(buffer, content_type='application/octet-stream')

def process_file(file_path):
    dfs = []
    for bucket_name, base_path in source_infos:
        full_path = base_path + file_path
        try:
            df = read_feather_from_gcs(bucket_name, full_path)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
            return None

    if len(dfs) == len(source_infos):
        merged_df = pd.concat(dfs, ignore_index=True)
        output_full_path = output_base_path + file_path
        write_feather_to_gcs(merged_df, output_bucket_name, output_full_path)
        return file_path

    return None


# Use ThreadPoolExecutor for Concurrent Processing - Step 3: Apply Concurrency
# Identify all unique feather files across the source directories
unique_files = set()
for bucket_name, base_path in source_infos:
    blobs = client.list_blobs(bucket_name, prefix=base_path)
    for blob in blobs:
        if blob.name.endswith('.feather'):
            relative_path = blob.name[len(base_path):]
            unique_files.add(relative_path)

workers = 18  # Number of worker threads, adjust based on your environment and GCS limits
with ThreadPoolExecutor(max_workers=workers) as executor:
    # Submit tasks
    futures = [executor.submit(process_file, file_path) for file_path in unique_files]

    # Progress tracking with tqdm
    for future in tqdm(as_completed(futures), total=len(unique_files), desc="Merging Files", unit="file"):
        result = future.result()  # Blocking call, waits for the file to be processed
        if result is None:
            tqdm.write("A file was skipped due to an error.")

print("Merging completed.")
