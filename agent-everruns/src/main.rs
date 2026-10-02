//! Everruns as a Bot.
//!
//! AG-UI is built into the Everruns framework itself, as `Session::ag_ui` behind the facade's
//! `ag-ui` feature, so this holds no protocol logic of its own: it maps each AG-UI thread to an
//! Everruns session and returns the stream the framework projects. The contract with the rest of
//! OpenBot is the one every Bot meets: serve AG-UI on a port, answer `/health`, and refuse anybody
//! who does not carry the server's token.
//!
//! The framework rather than the Everruns server, because a harness here is a framework in an
//! image that runs on the model OpenBot's model screen chose, with OpenBot as the only governor.
//! An Everruns server is a platform with its own agents, credentials and policy, and its public
//! AG-UI endpoint refuses the system messages every OpenBot run carries.

mod settings;

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use everruns::ag_ui::{AgUiError, AgUiOptions, InterruptGate, Message, RunAgentInput};
use everruns::{Agent, BuildError, Engine, Provider, Session};
use futures::StreamExt;
use serde_json::json;

/// The one header OpenBot's server sends when it calls a managed Bot.
const TOKEN_HEADER: &str = "x-openbot-agent-token";
const DEFAULT_PORT: u16 = 4214;
/// Used only when a run carries no system message, which OpenBot's server always sends.
const FALLBACK_INSTRUCTIONS: &str = "You are a helpful coworker.";

/// Builds the agent for a new thread from that thread's instructions.
type AgentFactory = Arc<dyn Fn(String) -> Result<Agent, BuildError> + Send + Sync>;

#[derive(Clone)]
struct App {
    engine: Engine,
    gate: InterruptGate,
    threads: Arc<Mutex<HashMap<String, Session>>>,
    agent: AgentFactory,
    token: String,
}

impl App {
    /// `gate` is the one the agents register as their `ask_user` and approval responder, so a
    /// question or an approval ends the run as an AG-UI interrupt and the next run answers it.
    fn new(token: String, gate: InterruptGate, agent: AgentFactory) -> Self {
        Self {
            engine: Engine::new(),
            gate,
            threads: Arc::default(),
            agent,
            token,
        }
    }

    /// One session per AG-UI thread. The session owns the conversation, so the framework reads only
    /// the run's last user message; the instructions are read once, from the run that opens it.
    fn session(&self, thread_id: &str, instructions: String) -> Result<Session, BuildError> {
        let mut threads = self
            .threads
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(session) = threads.get(thread_id) {
            return Ok(session.clone());
        }
        let agent = (self.agent)(instructions)?;
        let session = self.engine.create(agent);
        threads.insert(thread_id.to_string(), session.clone());
        Ok(session)
    }

    fn router(self) -> Router {
        Router::new()
            .route("/", post(run))
            .route("/health", get(health))
            .layer(middleware::from_fn_with_state(
                self.clone(),
                refuse_without_the_server_token,
            ))
            .with_state(self)
    }
}

/// The standing role and the granted-tools note OpenBot sends as system messages, as one
/// instruction block, and the input without them.
///
/// Taken out rather than left in: the framework's run reads the last message and refuses one that
/// is not a user's, and OpenBot's system messages are instructions, not turns.
fn split_instructions(mut input: RunAgentInput) -> (String, RunAgentInput) {
    let mut instructions = Vec::new();
    input.messages.retain(|message| match message {
        Message::System(m) | Message::Developer(m) => {
            if !m.content.trim().is_empty() {
                instructions.push(m.content.clone());
            }
            false
        }
        _ => true,
    });
    let instructions = if instructions.is_empty() {
        FALLBACK_INSTRUCTIONS.to_string()
    } else {
        instructions.join("\n\n")
    };
    (instructions, input)
}

async fn run(State(app): State<App>, Json(input): Json<RunAgentInput>) -> Response {
    let (instructions, input) = split_instructions(input);
    let session = match app.session(&input.thread_id, instructions) {
        Ok(session) => session,
        Err(error) => {
            tracing::error!(%error, "could not build the agent");
            return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response();
        }
    };
    match session
        .ag_ui_with(input, AgUiOptions::new().gate(app.gate.clone()))
        .await
    {
        Ok(stream) => {
            let events = stream.map(|event| {
                Ok::<_, Infallible>(SseEvent::default().json_data(&event).unwrap_or_default())
            });
            Sse::new(events)
                .keep_alive(KeepAlive::default())
                .into_response()
        }
        Err(AgUiError::InvalidInput(why)) => (StatusCode::BAD_REQUEST, why).into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
    }
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "ok": true, "harness": "everruns" }))
}

async fn refuse_without_the_server_token(
    State(app): State<App>,
    request: Request,
    next: Next,
) -> Response {
    if request.uri().path() != "/health" {
        let offered = request
            .headers()
            .get(TOKEN_HEADER)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .trim();
        if app.token.is_empty() || !constant_time_eq(offered.as_bytes(), app.token.as_bytes()) {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "unauthorised" })),
            )
                .into_response();
        }
    }
    next.run(request).await
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// A variable the compose file exports as "" when nobody set it reads as unset.
fn env_var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn required(name: &str, provider: &str) -> Result<String, String> {
    env_var(name).ok_or_else(|| format!("{name} is not set; the {provider} provider needs it"))
}

/// The Everruns driver for the provider the model screen chose.
fn provider_for(provider: &str) -> Result<Provider, String> {
    match provider {
        "openai" => {
            let mut built =
                everruns_openai::provider("openai", required("OPENAI_API_KEY", provider)?);
            if let Some(base) = env_var("OPENAI_BASE_URL") {
                built = built.base_url(base);
            }
            Ok(built)
        }
        "anthropic" => {
            let mut built =
                everruns_anthropic::provider("anthropic", required("ANTHROPIC_API_KEY", provider)?);
            if let Some(base) = env_var("ANTHROPIC_BASE_URL") {
                // The driver appends `/messages`, so the base carries the version, as the server's
                // `normalizeModelBaseUrls` makes it for the SDK.
                let base = base.trim_end_matches('/');
                let versioned = base
                    .rsplit('/')
                    .next()
                    .is_some_and(|last| last.starts_with('v') && last[1..].parse::<u32>().is_ok());
                built = built.base_url(if versioned {
                    base.to_string()
                } else {
                    format!("{base}/v1")
                });
            }
            Ok(built)
        }
        "google" => {
            let mut built =
                everruns_gemini::provider("google", required("GOOGLE_API_KEY", provider)?);
            if let Some(base) = env_var("GOOGLE_GENERATIVE_AI_BASE_URL") {
                built = built.base_url(base);
            }
            Ok(built)
        }
        other => Err(format!(
            "BOT_PROVIDER is {other:?}; this Bot runs openai, anthropic or google"
        )),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            // The framework logs every turn at info; the container log keeps this Bot's own.
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,openbot_agent_everruns=info".into()),
        )
        .init();

    let lookup = |name: &str| std::env::var(name).ok();
    let spec = std::fs::read_to_string(settings::spec_path(&lookup))?;
    let chosen = settings::bot_settings(&spec, &lookup)?;
    // Refused at startup rather than at the first run, so a missing key is in the container log
    // before anybody talks to the Bot.
    let provider = provider_for(&chosen.provider)?;
    let token = env_var("MANAGED_AGENT_TOKEN").unwrap_or_default();
    if token.is_empty() {
        tracing::warn!("MANAGED_AGENT_TOKEN is not set; every run will be refused");
    }

    let gate = InterruptGate::new();
    let model = chosen.model.clone();
    let agent_gate = gate.clone();
    let factory: AgentFactory = Arc::new(move |instructions| {
        Agent::builder()
            .name("openbot")
            .instructions(instructions)
            .provider(provider.clone())
            .model(model.as_str())
            .ask_user(agent_gate.clone())
            .approver(agent_gate.clone())
            .build()
    });
    let app = App::new(token, gate, factory);

    let port = env_var("PORT")
        .and_then(|port| port.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    // 0.0.0.0, not `::`, for the reason agent-pydantic-ai gives: the Compose healthcheck and the
    // server reach the Bot on 127.0.0.1.
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tracing::info!(
        provider = %chosen.provider,
        model = %chosen.model,
        "everruns Bot listening on :{port}"
    );
    axum::serve(listener, app.router())
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use everruns::Model;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn app(reply: &'static str) -> (App, Arc<Mutex<Vec<String>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let record = seen.clone();
        let factory: AgentFactory = Arc::new(move |instructions| {
            record.lock().unwrap().push(instructions.clone());
            Agent::builder()
                .instructions(instructions)
                .model(Model::simulated(reply))
                .build()
        });
        (
            App::new("secret".into(), InterruptGate::new(), factory),
            seen,
        )
    }

    fn run_request(token: Option<&str>, body: serde_json::Value) -> Request<Body> {
        let mut request = Request::post("/").header("content-type", "application/json");
        if let Some(token) = token {
            request = request.header(TOKEN_HEADER, token);
        }
        request.body(Body::from(body.to_string())).unwrap()
    }

    fn openbot_run(thread: &str, text: &str) -> serde_json::Value {
        json!({
            "threadId": thread,
            "runId": "run-1",
            "messages": [
                { "id": "standing-role:analyst", "role": "system", "content": "You are Ada, Analyst." },
                { "id": "m1", "role": "user", "content": text },
                { "id": "granted-tools:analyst", "role": "system", "content": "You hold no tools." }
            ],
            "tools": [],
            "context": []
        })
    }

    #[tokio::test]
    async fn health_needs_no_token() {
        let (app, _) = app("hi");
        let response = app
            .router()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn a_run_without_the_servers_token_is_refused() {
        let (app, _) = app("hi");
        for token in [None, Some("wrong")] {
            let response = app
                .clone()
                .router()
                .oneshot(run_request(token, openbot_run("t1", "Hi")))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        }
    }

    #[tokio::test]
    async fn an_unset_token_refuses_everybody() {
        let (mut app, _) = app("hi");
        app.token = String::new();
        let response = app
            .router()
            .oneshot(run_request(Some(""), openbot_run("t1", "Hi")))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn an_openbot_run_streams_the_reply_with_its_system_messages_as_instructions() {
        let (app, seen) = app("Hello from Everruns.");
        let response = app
            .router()
            .oneshot(run_request(Some("secret"), openbot_run("t1", "Hi")))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("RUN_STARTED"), "{body}");
        assert!(body.contains("Hello from Everruns."), "{body}");
        assert!(body.contains("RUN_FINISHED"), "{body}");
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            ["You are Ada, Analyst.\n\nYou hold no tools."]
        );
    }

    #[tokio::test]
    async fn a_thread_keeps_its_session() {
        let (app, seen) = app("ok");
        for text in ["one", "two"] {
            let response = app
                .clone()
                .router()
                .oneshot(run_request(Some("secret"), openbot_run("t1", text)))
                .await
                .unwrap();
            let _ = response.into_body().collect().await.unwrap();
        }
        assert_eq!(seen.lock().unwrap().len(), 1, "a second session was built");
    }

    #[test]
    fn an_unknown_provider_is_refused_by_name() {
        let error = provider_for("mistral").unwrap_err();
        assert!(error.contains("\"mistral\""), "{error}");
    }
}
