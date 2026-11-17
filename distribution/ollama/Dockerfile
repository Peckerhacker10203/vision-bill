# syntax=docker/dockerfile:1
# Maintainers must supply an immutable upstream manifest reference. No default.
ARG OLLAMA_BASE
FROM ${OLLAMA_BASE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3 util-linux \
 && command -v flock && test -x /bin/ollama \
 && rm -rf /var/lib/apt/lists/*
COPY paddleocr-service/ocr_service/supervision.py /opt/ocr/supervision.py
COPY distribution/ollama/entrypoint.sh /opt/ocr/entrypoint.sh
COPY distribution/ollama/probe.py /opt/ocr/probe.py
RUN chmod 0755 /opt/ocr/entrypoint.sh
ENTRYPOINT ["/opt/ocr/entrypoint.sh"]
CMD ["serve"]
