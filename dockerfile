FROM rust:1.93.1-bullseye as builder 

WORKDIR /app
COPY . .

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main(){}" > src/main.rs
RUN cargo build --release
RUN rm -r src

COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/softwerk-ocr-hackathon /usr/local/bin/softwerk-ocr-hackathon
CMD [ "softwerk-ocr-hackathon" ]