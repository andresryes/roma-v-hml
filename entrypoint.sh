#!/bin/bash
set -e

case "$1" in
  image)
    python3 app/evaluation_image.py
    ;;
  3d)
    python3 app/evaluation_3d.py
    ;;
  *)
    echo "Usage: docker run <image> [image|3d]"
    exit 1
    ;;
esac
