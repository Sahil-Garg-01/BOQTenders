# services/s3_utils.py

import boto3
import os
from botocore.exceptions import NoCredentialsError
import mimetypes
from config.settings import settings

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.s3.access_key_id,
    aws_secret_access_key=settings.s3.secret_access_key,
    region_name=settings.s3.region
)

def upload_to_s3(local_path_or_obj, s3_key: str) -> bool:
    try:
        if isinstance(local_path_or_obj, str):
            s3_client.upload_file(local_path_or_obj, settings.s3.bucket_name, s3_key)
        else:
            # Assume it's a file-like object (e.g., BytesIO)
            s3_client.upload_fileobj(local_path_or_obj, settings.s3.bucket_name, s3_key)
        print(f"Uploaded to S3: {s3_key}")
        return True
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
        return False

def generate_presigned_get_url(
    s3_key: str,
    expires_in: int = settings.s3.url_expires,
    as_attachment: bool = True,
    override_filename: str | None = None,
) -> str | None:
    if not s3_key:
        return None
    ct = mimetypes.guess_type(s3_key)[0] or "application/octet-stream"
    filename = override_filename or os.path.basename(s3_key)

    params = {
        "Bucket": settings.s3.bucket_name,
        "Key": s3_key,
        "ResponseContentType": ct,
        "ResponseContentDisposition": f'attachment; filename="{filename}"' if as_attachment
                                      else f'inline; filename="{filename}"',
    }
    try:
        return s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params=params,
            ExpiresIn=expires_in,
        )
    except Exception as e:
        print(f"Failed to generate presigned URL: {e}")
        return None