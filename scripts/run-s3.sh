#!/bin/bash
BASE_FOLDER=/var/tmp/
HOSTNAME=$(hostname -f)
PRIMARY_IP=$(hostname -I | awk '{print $1}')
mkdir -p ${BASE_FOLDER}/minio/{s3-volume,certs/CAs}
openssl req -new -newkey rsa:2048 -sha256 -days 3650 -nodes -x509 -extensions v3_ca \
            -keyout ${BASE_FOLDER}/minio/certs/private.key -out ${BASE_FOLDER}/minio/certs/public.crt \
            -subj "/C=US/ST=Texas/L=Austin/O=IT/OU=IT/CN=${HOSTNAME}" \
            -addext "subjectAltName = DNS:${HOSTNAME}, IP:${PRIMARY_IP}"
ln -s ${BASE_FOLDER}/minio/certs/public.crt ${BASE_FOLDER}/minio/certs/CAs/
# Run the MinIO server using Podman
podman run --name minio-server --rm -d -p 9000:9000 -p 9001:9001 -v ${BASE_FOLDER}/minio/s3-volume:/data:rw,z \
           -v ${BASE_FOLDER}/minio/certs:/certs:rw,z -e MINIO_SERVER_URL="https://${HOSTNAME}:9000" \
           -e MINIO_ROOT_USER=admin -e MINIO_ROOT_PASSWORD=admin1234 quay.io/minio/minio:latest server /data --console-address ":9001" \
           --certs-dir /certs
