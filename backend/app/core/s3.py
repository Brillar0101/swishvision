"""
AWS S3 client utilities for video storage.
"""
import boto3
from botocore.exceptions import ClientError
from typing import Optional, BinaryIO
import os

from app.core.config import settings


class S3Client:
    """S3 client wrapper for video upload/download operations."""

    def __init__(self):
        self.client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        self.bucket = settings.S3_BUCKET_NAME

    def upload_file(
        self,
        file_obj: BinaryIO,
        s3_key: str,
        content_type: str = "video/mp4",
    ) -> str:
        """
        Upload a file to S3.

        Args:
            file_obj: File-like object to upload
            s3_key: S3 object key (path)
            content_type: MIME type of the file

        Returns:
            S3 URI of uploaded file
        """
        self.client.upload_fileobj(
            file_obj,
            self.bucket,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        return f"s3://{self.bucket}/{s3_key}"

    def upload_local_file(self, local_path: str, s3_key: str) -> str:
        """
        Upload a local file to S3.

        Args:
            local_path: Path to local file
            s3_key: S3 object key (path)

        Returns:
            S3 URI of uploaded file
        """
        content_type = "video/mp4" if local_path.endswith(".mp4") else "application/octet-stream"
        self.client.upload_file(
            local_path,
            self.bucket,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        return f"s3://{self.bucket}/{s3_key}"

    def download_file(self, s3_key: str, local_path: str) -> str:
        """
        Download a file from S3.

        Args:
            s3_key: S3 object key (path)
            local_path: Local path to save file

        Returns:
            Local path of downloaded file
        """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.client.download_file(self.bucket, s3_key, local_path)
        return local_path

    def generate_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600,
        method: str = "get_object",
    ) -> Optional[str]:
        """
        Generate a presigned URL for accessing an S3 object.

        Args:
            s3_key: S3 object key (path)
            expiration: URL expiration time in seconds
            method: S3 operation (get_object or put_object)

        Returns:
            Presigned URL or None if error
        """
        try:
            url = self.client.generate_presigned_url(
                method,
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=expiration,
            )
            return url
        except ClientError:
            return None

    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3.

        Args:
            s3_key: S3 object key (path)

        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def delete_folder(self, prefix: str) -> bool:
        """
        Delete all files with a given prefix (folder).

        Args:
            prefix: S3 key prefix

        Returns:
            True if deleted successfully
        """
        try:
            response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            if "Contents" in response:
                objects = [{"Key": obj["Key"]} for obj in response["Contents"]]
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
            return True
        except ClientError:
            return False

    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            s3_key: S3 object key (path)

        Returns:
            True if file exists
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False


# Global S3 client instance
s3_client = S3Client()
