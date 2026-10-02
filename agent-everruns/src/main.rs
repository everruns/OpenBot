//! Everruns as a Bot.
//!
//! AG-UI is built into the Everruns framework itself, down to the HTTP route: `AgUiHandler`, behind
//! the facade's `ag-ui-axum` feature, authorizes the request, resolves its thread to a session and
//! streams the run as SSE. So this holds no protocol logic of its own, only the contract every Bot
//! meets with the rest of OpenBot: serve AG-UI on a port, answer `/health`, and refuse anybody who
//! does not carry the server's token.
//!
//! The framework rather than the Everruns server, because a harness here is a framework in an
//! image that runs on the model OpenBot's model screen chose, with OpenBot as the only governor.
//! An Everruns server is a platform with its own agents, credentials and policy, and its public
//! AG-UI endpoint refuses the system messages every OpenBot run carries.

mod settings;

use axum::http::HeaderName;
use axum::routing::get;
use axum::{Json, Router};
use everruns::ag_ui::{AgUiHandler, AgUiOptions, AgUiThreads, InterruptGate, StaticToken};
use everruns::{Agent, Anthropic, BuildError, Engine, Gemini, Model, OpenAI, Provider};
use serde_json::json;

/// The one header OpenBot's server sends when it calls a managed Bot.
const TOKEN_HEADER: &str = "x-openbot-agent-token";
const DEFAULT_PORT: u16 = 4214;
/// The agent's own instructions. Each run's system messages, the coworker's standing role and the
/// note on what it holds, follow them as that run's instructions, so this says only what is true of
/// every coworker.
const BASE_INSTRUCTIONS: &str = "You are a coworker in OpenBot. The instructions that follow say who you are and what you hold.";

/// The agent every thread's session is created from: `model` on `provider`, with `gate` as its
/// `ask_user` and approval responder, so a question or an approval ends the run as an AG-UI
/// interrupt and the next run answers it.
fn agent(provider: Provider, model: Model, gate: &InterruptGate) -> Result<Agent, BuildError> {
    Agent::builder()
        .name("openbot")
        .instructions(BASE_INSTRUCTIONS)
        .provider(provider)
        .model(model)
        .ask_user(gate.clone())
        .approver(gate.clone())
        .build()
}

/// The Bot's routes. `threads` maps each AG-UI thread to an Everruns session and, when the
/// container restarted under a thread, seeds a new session from the history the run carries, which
/// OpenBot's server sends in full on every run.
fn router(threads: AgUiThreads, gate: InterruptGate, token: String) -> Router {
    let handler = AgUiHandler::new(
        threads,
        // An empty token refuses everybody, which is the right way round for a Bot nobody gave one.
        StaticToken::header(HeaderName::from_static(TOKEN_HEADER), token),
    )
    .options(
        AgUiOptions::new()
            .gate(gate)
            // Trusted because the token above is the server's: the standing role and the
            // granted-tools note arrive as system messages, and become that run's instructions.
            .input_instructions(true),
    );
    Router::new()
        .route("/", handler.route())
        .route("/health", get(health))
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "ok": true, "harness": "everruns" }))
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

/// The Everruns provider for the one the model screen chose.
fn provider_for(provider: &str) -> Result<Provider, String> {
    match provider {
        "openai" => {
            let mut config = OpenAI::new(required("OPENAI_API_KEY", provider)?);
            if let Some(base) = env_var("OPENAI_BASE_URL") {
                config = config.base_url(base);
            }
            Ok(config.into())
        }
        "anthropic" => {
            let mut config = Anthropic::new(required("ANTHROPIC_API_KEY", provider)?);
            if let Some(base) = env_var("ANTHROPIC_BASE_URL") {
                config = config.base_url(versioned(&base));
            }
            Ok(config.into())
        }
        "google" => {
            let mut config = Gemini::new(required("GOOGLE_API_KEY", provider)?);
            if let Some(base) = env_var("GOOGLE_GENERATIVE_AI_BASE_URL") {
                config = config.base_url(base);
            }
            Ok(config.into())
        }
        other => Err(format!(
            "BOT_PROVIDER is {other:?}; this Bot runs openai, anthropic or google"
        )),
    }
}

/// An Anthropic base URL with its API version, as the server's `normalizeModelBaseUrls` makes it
/// for the SDK: the provider takes the versioned API root and appends `/messages`.
fn versioned(base: &str) -> String {
    let base = base.trim_end_matches('/');
    let has_version = base
        .rsplit('/')
        .next()
        .is_some_and(|last| last.starts_with('v') && last[1..].parse::<u32>().is_ok());
    if has_version {
        base.to_string()
    } else {
        format!("{base}/v1")
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
    let agent = agent(provider, Model::from(chosen.model.as_str()), &gate)?;
    // In memory: a restarted container gets a new session per thread, seeded from the history the
    // next run carries, so the conversation goes on without a volume to keep.
    let threads = AgUiThreads::new(Engine::new(), agent);

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
    axum::serve(listener, router(threads, gate, token))
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
    use axum::http::{Request, StatusCode};
    use everruns::{LlmSimConfig, ToolCall};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn threads(model: Model) -> (AgUiThreads, InterruptGate) {
        let gate = InterruptGate::new();
        let agent = Agent::builder()
            .instructions(BASE_INSTRUCTIONS)
            .model(model)
            .ask_user(gate.clone())
            .approver(gate.clone())
            .build()
            .unwrap();
        (AgUiThreads::new(Engine::new(), agent), gate)
    }

    fn app(reply: &'static str) -> (Router, AgUiThreads) {
        let (threads, gate) = threads(Model::simulated(reply));
        (router(threads.clone(), gate, "secret".into()), threads)
    }

    fn run_request(token: Option<&str>, body: serde_json::Value) -> Request<Body> {
        let mut request = Request::post("/").header("content-type", "application/json");
        if let Some(token) = token {
            request = request.header(TOKEN_HEADER, token);
        }
        request.body(Body::from(body.to_string())).unwrap()
    }

    /// A run shaped like the ones OpenBot's server sends: the standing role and the granted-tools
    /// note as system messages before the conversation, which it sends in full.
    fn openbot_run(thread: &str, run: &str, conversation: serde_json::Value) -> serde_json::Value {
        let mut messages = vec![
            json!({ "id": "standing-role:analyst", "role": "system", "content": "You are Ada, Analyst." }),
            json!({ "id": "granted-tools:analyst", "role": "system", "content": "You hold no tools." }),
        ];
        messages.extend(conversation.as_array().unwrap().iter().cloned());
        json!({
            "threadId": thread,
            "runId": run,
            "messages": messages,
            "tools": [],
            "context": []
        })
    }

    fn user(id: &str, text: &str) -> serde_json::Value {
        json!({ "id": id, "role": "user", "content": text })
    }

    async fn body(router: &Router, request: Request<Body>) -> (StatusCode, String) {
        let response = router.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn health_needs_no_token() {
        let (router, _) = app("hi");
        let response = router
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn a_run_without_the_servers_token_is_refused() {
        let (router, _) = app("hi");
        for token in [None, Some("wrong")] {
            let request = run_request(token, openbot_run("t1", "r1", json!([user("m1", "Hi")])));
            let (status, _) = body(&router, request).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
        }
    }

    #[tokio::test]
    async fn an_unset_token_refuses_everybody() {
        let (threads, gate) = threads(Model::simulated("hi"));
        let router = router(threads, gate, String::new());
        let request = run_request(Some(""), openbot_run("t1", "r1", json!([user("m1", "Hi")])));
        let (status, _) = body(&router, request).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn an_openbot_run_streams_the_reply_with_its_system_messages_as_instructions() {
        let (router, threads) = app("Hello from Everruns.");
        let request = run_request(
            Some("secret"),
            openbot_run("t1", "r1", json!([user("m1", "Hi")])),
        );
        let (status, body) = body(&router, request).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("RUN_STARTED"), "{body}");
        assert!(body.contains("Hello from Everruns."), "{body}");
        assert!(body.contains("RUN_FINISHED"), "{body}");

        let session = threads.session("t1").await.unwrap().into_session();
        let instructions = session.inspect().await.unwrap().instructions;
        assert!(instructions.contains(BASE_INSTRUCTIONS), "{instructions}");
        assert!(
            instructions.contains("You are Ada, Analyst."),
            "{instructions}"
        );
        assert!(
            instructions.contains("You hold no tools."),
            "{instructions}"
        );
    }

    #[tokio::test]
    async fn a_thread_keeps_its_session() {
        let (router, threads) = app("ok");
        let first = run_request(
            Some("secret"),
            openbot_run("t1", "r1", json!([user("m1", "one")])),
        );
        let _ = body(&router, first).await;
        let opened = threads.session("t1").await.unwrap().session().session_id();
        let second = run_request(
            Some("secret"),
            openbot_run(
                "t1",
                "r2",
                json!([
                    user("m1", "one"),
                    { "id": "a1", "role": "assistant", "content": "ok" },
                    user("m2", "two")
                ]),
            ),
        );
        let _ = body(&router, second).await;
        let again = threads.session("t1").await.unwrap();
        assert!(!again.created());
        assert_eq!(
            again.session().session_id(),
            opened,
            "a second session was built"
        );
    }

    #[tokio::test]
    async fn a_thread_new_to_this_container_is_seeded_from_the_runs_history() {
        let (router, threads) = app("ok");
        // As after a restart: the thread has history, but no session here holds it.
        let request = run_request(
            Some("secret"),
            openbot_run(
                "t1",
                "r5",
                json!([
                    user("m1", "My name is Ada."),
                    { "id": "a1", "role": "assistant", "content": "Hello, Ada." },
                    user("m2", "What is my name?")
                ]),
            ),
        );
        let (status, _) = body(&router, request).await;
        assert_eq!(status, StatusCode::OK);
        let session = threads.session("t1").await.unwrap().into_session();
        let history = format!("{:?}", session.inspect().await.unwrap().messages);
        assert!(history.contains("My name is Ada."), "{history}");
        assert!(history.contains("Hello, Ada."), "{history}");
    }

    #[tokio::test]
    async fn a_surface_tool_ends_the_run_and_its_result_continues_the_turn() {
        // The model calls the surface's `show_chart`, then answers once it has the result.
        let model = Model::simulated_with_config(
            LlmSimConfig::fixed("There is the chart.").with_tool_call_sequence(vec![
                vec![ToolCall {
                    id: "call_chart".to_string(),
                    name: "show_chart".to_string(),
                    arguments: json!({ "series": [1, 2, 3] }),
                }],
                vec![],
            ]),
        );
        let (threads, gate) = threads(model);
        let router = router(threads, gate, "secret".into());
        let chart = json!({
            "name": "show_chart",
            "description": "Draw a chart in the conversation.",
            "parameters": { "type": "object", "properties": { "series": { "type": "array" } } }
        });

        let mut first = openbot_run("t1", "r1", json!([user("m1", "Chart 1, 2, 3.")]));
        first["tools"] = json!([chart.clone()]);
        let (status, body_one) = body(&router, run_request(Some("secret"), first)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body_one.contains("TOOL_CALL_START"), "{body_one}");
        assert!(body_one.contains("show_chart"), "{body_one}");
        assert!(body_one.contains("call_chart"), "{body_one}");
        assert!(!body_one.contains("There is the chart."), "{body_one}");

        let mut second = openbot_run(
            "t1",
            "r2",
            json!([
                user("m1", "Chart 1, 2, 3."),
                {
                    "id": "a1",
                    "role": "assistant",
                    "toolCalls": [{
                        "id": "call_chart",
                        "type": "function",
                        "function": { "name": "show_chart", "arguments": "{\"series\":[1,2,3]}" }
                    }]
                },
                { "id": "t1", "role": "tool", "toolCallId": "call_chart", "content": "drawn" }
            ]),
        );
        second["tools"] = json!([chart]);
        let (status, body_two) = body(&router, run_request(Some("secret"), second)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body_two.contains("There is the chart."), "{body_two}");
    }

    #[test]
    fn an_unknown_provider_is_refused_by_name() {
        let error = provider_for("mistral").unwrap_err();
        assert!(error.contains("\"mistral\""), "{error}");
    }

    #[test]
    fn an_anthropic_base_url_gets_its_version() {
        assert_eq!(
            versioned("https://proxy.example/"),
            "https://proxy.example/v1"
        );
        assert_eq!(
            versioned("https://proxy.example/v1"),
            "https://proxy.example/v1"
        );
    }
}
