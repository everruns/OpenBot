# Everruns, as a Bot. Rust because that is what the framework is written in, and the protocol is the
# only thing a Bot has to share with the others.
FROM rust:1.96-slim-bookworm AS build

WORKDIR /build

# Dependencies before source, so editing the Bot does not rebuild the framework: a stub main
# compiles the whole dependency graph into a layer the real source then reuses.
COPY agent-everruns/Cargo.toml agent-everruns/Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
  && cargo build --release --locked \
  && rm -rf src target/release/deps/openbot_agent_everruns-* target/release/openbot-agent-everruns

COPY agent-everruns/src ./src
RUN cargo build --release --locked

FROM debian:bookworm-slim

# The model providers are reached over TLS, and the driver verifies against the system roots.
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/target/release/openbot-agent-everruns /usr/local/bin/openbot-agent-everruns
COPY shared/model-providers.json /shared/

ENV PORT=4214
EXPOSE 4214
# The binary binds 0.0.0.0, for the reason agent-pydantic-ai's Dockerfile gives.
CMD ["openbot-agent-everruns"]
