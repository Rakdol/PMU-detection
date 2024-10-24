import json
from minio import Minio
from src.configurations import BucketServer

bucket_name = BucketServer.bucket_name

client = Minio(
    f"{BucketServer.bucket_server}:{BucketServer.bucket_port}",
    BucketServer.bucket_access,
    BucketServer.bucket_secret,
    secure=False,
)
