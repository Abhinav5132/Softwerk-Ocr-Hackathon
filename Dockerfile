FROM rust:1.93-slim-bookworm AS builder

WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://hf.co/cli/install.sh | bash
RUN hf download lightonai/LightOnOCR-2-1B-bbox-soup --local-dir models/LightOnOCR

WORKDIR /app
COPY --from=builder /usr/src/app/target/release/softwerk-ocr-hackathon /app/softwerk-ocr-hackathon


CMD ["./softwerk-ocr-hackathon"]
