#!/bin/bash
gunicorn app:server --workers=1 --bind=0.0.0.0:10000
