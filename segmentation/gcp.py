"""
When interacting with Google Cloud Client libraries, the library can auto-detect the
credentials to use.

1. Before running this sample, set up ADC as described in
   https://cloud.google.com/docs/authentication/external/set-up-adc
2. Make sure that the user account or service account that you are using
   has the required permissions. For this sample, you must have "storage.buckets.list".
"""
from pathlib import Path
import logging

from google.cloud import storage

GCP_PROJECT_ID = "vv-segmentation"
BUCKET_ID = "vv-segmentation"

logger = logging.getLogger(__name__)


def authenticate_implicit_with_adc():
    # This snippet demonstrates how to list buckets.
    # *NOTE*: Replace the client created below with the client required for your application.
    # Note that the credentials are not specified when constructing the client.
    # Hence, the client library will look for credentials using ADC.
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    buckets = storage_client.list_buckets()
    print("Buckets:")
    for bucket in buckets:
        print(bucket.name)
    print("Listed all storage buckets.")


def upload_blob(source_file_name: str | Path, destination_blob_name: str):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_ID)

    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(str(source_file_name))

    logger.debug(f"File {source_file_name} uploaded to {destination_blob_name}.")
