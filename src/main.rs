//! The binary.

use std::sync::Arc;

use clap::Parser;
use cookie_backend::api::{self, ApiState};
use cookie_backend::auth::DeviceStore;
use cookie_backend::cli::{Cli, Command};
use cookie_backend::config::{self, Config};
use cookie_backend::error::{Error, Result};
use cookie_backend::ollama::OllamaProvider;
use cookie_backend::VERSION;

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    init_logging(&cli.log_level);

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("cookie")
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("cookie-backend: could not start the async runtime: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    match runtime.block_on(run(cli)) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cookie-backend: {e}");
            if let Some(hint) = e.hint() {
                eprintln!("  hint: {hint}");
            }
            std::process::ExitCode::FAILURE
        }
    }
}

fn init_logging(level: &str) {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("cookie_backend={level},warn")));
    let _ = fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .try_init();
}

async fn run(cli: Cli) -> Result<()> {
    let path = cli.config.clone().unwrap_or_else(config::config_file);
    let (mut config, created) = Config::load_or_init(&path)?;
    if created {
        println!("wrote a default configuration to {}", path.display());
    }
    if let Some(port) = cli.port {
        config.server.port = port;
    }

    match cli.command {
        Some(Command::Init) => {
            println!("configuration: {}", path.display());
            println!("data:          {}", config.data_dir.display());
            Ok(())
        }
        Some(Command::Show) => {
            println!("# {}", path.display());
            println!("{}", config.to_toml()?);
            Ok(())
        }
        Some(Command::Pair) => pair(&config),
        Some(Command::Devices) => devices(&config),
        Some(Command::Revoke { name }) => revoke(&config, &name),
        Some(Command::Doctor) => doctor(&config).await,
        None => serve(config).await,
    }
}

fn pair(config: &Config) -> Result<()> {
    // The code lives in the serving process, so this prints one for the
    // *running* server to accept. Holding codes on disk would mean a pairing
    // window somebody forgot about surviving a restart.
    let mut store = DeviceStore::open(config.devices_file());
    let code = store.begin_pairing();
    println!("Pairing code (ten minutes, single use):\n");
    println!("    {code}\n");
    println!("On the machine running the frontend:\n");
    println!(
        "    curl -s http://<this-machine>:{}{}/v1/pair \\",
        config.server.port,
        config.server.base_path.trim_end_matches('/')
    );
    println!("      -H 'content-type: application/json' \\");
    println!("      -d '{{\"code\":\"{code}\",\"device_name\":\"laptop\"}}'\n");
    println!("Store the token it returns in COOKIE_BACKEND_TOKEN on that machine.");
    println!("\nNote: codes are held in memory by the process that issued them, so");
    println!("run this against the running server's /v1/pair endpoint.");
    Ok(())
}

fn devices(config: &Config) -> Result<()> {
    let store = DeviceStore::open(config.devices_file());
    let devices = store.devices();
    if devices.is_empty() {
        println!("Nothing paired yet. The API is open until something pairs.");
        return Ok(());
    }
    for device in devices {
        let seen = device
            .last_seen_at
            .map(|t| t.to_string())
            .unwrap_or_else(|| "never".into());
        println!(
            "  {:<24} paired {}  last seen {seen}",
            device.name, device.created_at
        );
    }
    Ok(())
}

fn revoke(config: &Config, name: &str) -> Result<()> {
    let mut store = DeviceStore::open(config.devices_file());
    if store.revoke(name)? {
        println!("revoked {name}");
        Ok(())
    } else {
        Err(Error::Other(format!("no device named {name:?}")))
    }
}

/// The same question the frontend asks: is everything working?
async fn doctor(config: &Config) -> Result<()> {
    println!("cookie-backend {VERSION}\n");
    println!("  config      {}", config::config_file().display());
    println!("  data        {}", config.data_dir.display());
    println!(
        "  listening   {}:{}{}",
        config.server.bind, config.server.port, config.server.base_path
    );

    let paired = DeviceStore::open(config.devices_file()).devices();
    if paired.is_empty() {
        println!("  paired      nothing yet — the API is open until you pair");
    } else {
        let names: Vec<String> = paired.iter().map(|d| d.name.clone()).collect();
        println!(
            "  paired      {} device(s): {}",
            paired.len(),
            names.join(", ")
        );
    }

    let provider = OllamaProvider::new(config)?;
    if !provider.available().await {
        println!("  ollama      NOT REACHABLE at {}", provider.endpoint());
        println!("              start it with: ollama serve");
        return Err(Error::OllamaUnreachable {
            endpoint: provider.endpoint().to_string(),
        });
    }
    println!("  ollama      up at {}", provider.endpoint());

    let installed = provider.installed_models().await?;
    let mut problems = 0;
    for (name, role) in &config.models {
        if installed.contains(&role.model) {
            println!(
                "  {name:<11} {} installed, keep_alive {}",
                role.model, role.keep_alive
            );
        } else {
            problems += 1;
            println!(
                "  {name:<11} {} MISSING — run: ollama pull {}",
                role.model, role.model
            );
        }
    }

    let loaded = provider.loaded_models().await.unwrap_or_default();
    let resident: f32 = loaded.iter().map(|m| m.size_gb()).sum();
    println!(
        "  resident    {resident:.1} GB of {:.1} GB budget",
        config.limits.model_memory_gb
    );
    if resident > config.limits.model_memory_gb {
        problems += 1;
        println!("              over budget: expect swapping and long pauses");
    }

    println!();
    if problems == 0 {
        println!("  Everything is ready.");
        Ok(())
    } else {
        Err(Error::Other(format!(
            "{problems} thing(s) need attention, see above"
        )))
    }
}

async fn serve(config: Config) -> Result<()> {
    let config = Arc::new(config);
    let state = ApiState::new(config.clone())?;

    println!(
        "cookie-backend {VERSION} on http://{}:{}{}",
        config.server.bind, config.server.port, config.server.base_path
    );
    if state.devices.lock().await.is_empty() {
        println!("  nothing paired yet — the API is open until the first device pairs");
    }
    if !state.provider.available().await {
        println!(
            "  warning: Ollama is not answering at {}. Start it with: ollama serve",
            state.provider.endpoint()
        );
    }

    api::serve(state, async {
        let _ = tokio::signal::ctrl_c().await;
        println!("\nstopping");
    })
    .await
}
