# encoding: utf-8
# Author: Li Qiliang(l00423096).

import os
import sys
import copy
import ctypes
import base64
import logging
import argparse
import multiprocessing
from urllib import request
import random
import json

from boto3.session import Session

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(
    description="Parameters of CSB-OBS",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--vendor", type=str, default="HEC", help="vendor of bucket.")
parser.add_argument(
    "--region", type=str, default="cn-north-1", help="region of bucket."
)
parser.add_argument("--app_token", type=str, default=None, help="appToken of CSB.")
parser.add_argument(
    "--bucket_name", type=str, default=None, help="Please input you bucket_name."
)
parser.add_argument(
    "--path", type=str, default="", help="Please input you path(objectKey)."
)
parser.add_argument(
    "--objects_storage_path", type=str, default=None, help="Output list of objectkeys."
)
parser.add_argument(
    "--buffer_size",
    type=int,
    default=8192,
    help="The default value of buffer_size is 8192.",
)

result, _ = parser.parse_known_args()
args = copy.deepcopy(result)

file_server_url = "http://10.155.175.112:8080/csb-file-server"
# file_server_url = 'http://127.0.0.1:8080/csb-file-server'

back_server_url = ""


def bucket_auth(vendor, region, bucket_name, app_token):
    try:
        if bucket_name is None:
            raise Exception("The --bucket_name can not be null.")

        if app_token is None:
            raise Exception("The --app_token can not be null.")

        bucket_auth_endpoint = (
            file_server_url
            + "/rest/boto3/s3/bucket-auth?vendor="
            + vendor
            + "&region="
            + region
            + "&bucketid="
            + bucket_name
            + "&apptoken="
            + app_token
        )
        req = request.Request(url=bucket_auth_endpoint)
        res = request.urlopen(req)
        result = res.read().decode(encoding="utf-8")
        result_dict = json.loads(result)
        return result_dict["success"], result_dict["msg"]
    except Exception as e:
        sys.stdout.write(str(e) + "\n")


def get_s3_client_list():
    global back_server_url
    s3_client_list = []
    try:
        query_urls_endpoint = (
            file_server_url
            + "/rest/boto3/s3/query/csb-file-server/all/ip-and-port?vendor="
            + args.vendor
            + "&region="
            + args.region
        )
        req = request.Request(url=query_urls_endpoint)
        res = request.urlopen(req)
        result = res.read().decode(encoding="utf-8")
        result_dict = json.loads(result)
        if result_dict["fileServerUrlList"] is not None:
            for csb_file_server_url in result_dict["fileServerUrlList"]:
                if (
                    csb_file_server_url is not None
                    and csb_file_server_url != ""
                    and csb_file_server_url != "null"
                ):
                    back_server_url = str(csb_file_server_url)
                csb_obs_service_endpoint = (
                    csb_file_server_url
                    + "/rest/boto3/s3/"
                    + args.vendor
                    + "/"
                    + args.region
                    + "/"
                    + args.app_token
                )
                session = Session("Hello", "CSB-OBS")
                s3_client = session.client("s3", endpoint_url=csb_obs_service_endpoint)
                s3_client_list.append(s3_client)
        return s3_client_list
    except Exception as e:
        sys.stdout.write(str(e) + "\n")


def get_next_1000_objects(
    vendor, region, bucket_name, app_token, object_key, next_marker
):
    try:
        list_objects_endpoint = (
            back_server_url
            + "/rest/boto3/s3/list/bucket/objectkeys?vendor="
            + vendor
            + "&region="
            + region
            + "&bucketid="
            + bucket_name
            + "&apptoken="
            + app_token
            + "&objectkey="
            + object_key
            + "&nextmarker="
            + next_marker
        )
        req = request.Request(url=list_objects_endpoint)
        res = request.urlopen(req)
        result = res.read().decode(encoding="utf-8")
        result_dict = json.loads(result)
        return result_dict
    except Exception as e:
        raise Exception(str(e))


def download_object(
    s3_client_list,
    objects_storage_path,
    bucket_name,
    original_path,
    objectKey,
    failed_count=1,
    selected_index=0,
):
    try:
        wait_select_index_list = [index for index in range(len(s3_client_list))]
        if failed_count != 1 and len(s3_client_list) > 1:
            wait_select_index_list.remove(selected_index)

        random.shuffle(wait_select_index_list)
        selected_index = wait_select_index_list[0]
        s3_client = s3_client_list[selected_index]

        if len(original_path) > 0:
            pos = original_path[:-1].rfind("/")
            tracated_objectKey = objectKey[len(original_path[: pos + 1]) :]
        pos = tracated_objectKey.rfind("/")
        local_objects_storage_path = os.path.join(
            objects_storage_path, tracated_objectKey[: pos + 1]
        )
        if not os.path.exists(local_objects_storage_path):
            os.makedirs(local_objects_storage_path)

        objectKey_base64 = base64.urlsafe_b64encode(objectKey.encode(encoding="utf-8"))
        objectKey_base64 = str(objectKey_base64, encoding="utf-8")
        resp = s3_client.get_object(Bucket=bucket_name, Key=objectKey_base64)

        object_name = tracated_objectKey[pos + 1 :]
        with open(os.path.join(local_objects_storage_path, object_name), "wb") as f:
            file = resp["Body"]
            while True:
                data = file.read(args.buffer_size)
                if data == b"":
                    break
                f.write(data)
                f.flush()

    except Exception as e:
        if failed_count >= 3:
            raise Exception(str(e))
        else:
            failed_count += 1
            download_object(
                s3_client_list,
                objects_storage_path,
                bucket_name,
                original_path,
                objectKey,
                failed_count=failed_count,
                selected_index=selected_index,
            )


def iterate_download_objects(
    s3_client_list,
    result_dict,
    objects_storage_path,
    bucket_name,
    original_path,
    object_key_count,
    download_type="folder",
):
    try:
        if download_type == "folder":
            for obsObjectKeyDo in result_dict["objectKeys"]:
                objectKey = obsObjectKeyDo["objectKey"]
                download_object(
                    s3_client_list,
                    objects_storage_path,
                    bucket_name,
                    original_path,
                    objectKey,
                )

                object_key_count.value += 1
                sys.stdout.write(
                    "The number of downloaded files is "
                    + str(object_key_count.value)
                    + ".\n"
                )
        else:
            download_object(
                s3_client_list,
                objects_storage_path,
                bucket_name,
                original_path,
                original_path,
            )

            object_key_count.value += 1
            sys.stdout.write(
                "The number of downloaded files is "
                + str(object_key_count.value)
                + ".\n"
            )

    except Exception as e:
        raise Exception(str(e))


def main():
    try:
        vendor = args.vendor
        region = args.region
        bucket_name = args.bucket_name
        app_token = args.app_token
        next_marker = ""
        truncated = "true"

        path = args.path
        if path is None or len(path) == 0:
            raise Exception("The --path can not be null.")

        path = path.replace("\\", "/")
        # if not path.endswith('/'):
        #  path += '/'

        original_path = path
        if path is not None:
            path = base64.urlsafe_b64encode(path.encode(encoding="utf-8"))
            path = str(path, encoding="utf-8")

        if bucket_name is None:
            raise Exception("The --bucket_name can not be null.")

        if app_token is None:
            raise Exception("The --app_token can not be null.")

        sys.stdout.write("begin downloading, please waiting...\n")

        # First, bucket auth
        is_success, msg = bucket_auth(vendor, region, bucket_name, app_token)
        if not is_success:
            raise Exception(msg)

        objects_storage_path = args.objects_storage_path
        if objects_storage_path is None:
            dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
            objects_storage_path = dirname
        # else:
        #  objects_storage_path = objects_storage_path.replace('\\', '/')

        s3_client_list = get_s3_client_list()

        manager = multiprocessing.Manager()
        object_key_count = manager.Value(ctypes.c_longdouble, 0, lock=True)

        # Iteratively download a folder.
        if original_path.endswith("/"):
            while truncated == "true":
                result_dict = get_next_1000_objects(
                    vendor, region, bucket_name, app_token, path, next_marker
                )

                if result_dict["success"] == "true":
                    iterate_download_objects(
                        s3_client_list,
                        result_dict,
                        objects_storage_path,
                        bucket_name,
                        original_path,
                        object_key_count,
                        "folder",
                    )
                else:
                    raise Exception(result_dict["msg"])

                next_marker = result_dict["nextmarker"]
                truncated = result_dict["truncated"]
        # download a file.
        else:
            result_dict = []
            iterate_download_objects(
                s3_client_list,
                result_dict,
                objects_storage_path,
                bucket_name,
                original_path,
                object_key_count,
                "file",
            )

        sys.stdout.write("end." + "\n")

    except Exception as e:
        sys.stdout.write(str(e) + "\n")


if __name__ == "__main__":
    main()
