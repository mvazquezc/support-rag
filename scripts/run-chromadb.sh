#!/bin/bash

mkdir -p /var/tmp/chroma-data
podman run -d --rm --name chromadb -v /var/tmp/chroma-data:/data:rw,z -p 8000:8000 docker.io/chromadb/chroma:latest
