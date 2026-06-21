mod download;
mod gateway;
mod models;
mod orchestrator;
mod process;
mod tui;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use models::ShardSelection;
use orchestrator::{Role, ServeConfig};

#[derive(Debug, Parser)]
#[command(name = "compute-sharding")]
#[command(about = "Compute sharded inference CLI", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Download the validated Compute two-stage GGUF shards from Hugging Face.
    Download {
        #[arg(value_enum, default_value_t = ShardSelection::All)]
        shard: ShardSelection,
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },

    /// Run the local peer orchestrator and optionally launch stage sidecars.
    Serve {
        #[arg(long, value_enum, default_value_t = Role::Auto)]
        role: Role,
        #[arg(long, default_value = "0.0.0.0:8787")]
        bind: String,
        #[arg(long)]
        public_addr: Option<String>,
        #[arg(long = "peer")]
        peers: Vec<String>,
        #[arg(long)]
        stage_bind: Option<String>,
        #[arg(long)]
        stage_connect_addr: Option<String>,
        #[arg(long)]
        public_stage_addr: Option<String>,
        #[arg(long, default_value = "127.0.0.1:9300")]
        gateway_bind: String,
        #[arg(long)]
        sidecar_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long)]
        no_spawn: bool,
    },

    /// Probe peer orchestrators and print measured latency.
    Probe {
        #[arg(long = "peer")]
        peers: Vec<String>,
    },

    /// Send one prompt to a running stage gateway.
    Chat {
        #[arg(long, default_value = "127.0.0.1:9300")]
        gateway: String,
        #[arg(long, default_value_t = 96)]
        max_tokens: u32,
        prompt: Vec<String>,
    },

    /// Launch the Compute-branded TUI with globe, peers, and test chat tabs.
    Tui {
        #[arg(long, default_value = "127.0.0.1:9300")]
        gateway: String,
        #[arg(long, default_value = "http://127.0.0.1:8787")]
        orchestrator: String,
    },

    /// Print the validated shard catalog and sidecar search paths.
    Info,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Download { shard, model_dir } => {
            let root = model_dir.unwrap_or_else(models::default_model_dir);
            download::download_selection(shard, &root)
        }
        Commands::Serve {
            role,
            bind,
            public_addr,
            peers,
            stage_bind,
            stage_connect_addr,
            public_stage_addr,
            gateway_bind,
            sidecar_dir,
            model_dir,
            no_spawn,
        } => orchestrator::run(ServeConfig {
            role,
            bind,
            public_addr,
            peers,
            stage_bind,
            stage_connect_addr,
            public_stage_addr,
            gateway_bind,
            sidecar_dir,
            model_dir: model_dir.unwrap_or_else(models::default_model_dir),
            no_spawn,
        }),
        Commands::Probe { peers } => {
            if peers.is_empty() {
                anyhow::bail!("provide at least one --peer URL");
            }
            for peer in peers {
                let result =
                    orchestrator::probe_peer(&peer).with_context(|| format!("probing {peer}"))?;
                println!(
                    "{:<42} {:>6} ms  role={:<5} stage={}",
                    result.url,
                    result.latency_ms,
                    result.advert.role,
                    result.advert.stage_addr.as_deref().unwrap_or("-")
                );
            }
            Ok(())
        }
        Commands::Chat {
            gateway,
            max_tokens,
            prompt,
        } => {
            let prompt = if prompt.is_empty() {
                "Explain Compute sharding in one sentence.".to_string()
            } else {
                prompt.join(" ")
            };
            let completion = gateway::complete_prompt(&gateway, &prompt, max_tokens)
                .with_context(|| format!("chat through gateway {gateway}"))?;
            println!("{}", completion.text.trim());
            println!();
            println!(
                "tokens={} ttft={}ms total={}ms",
                completion.completion_tokens,
                completion.timings.ttft_ms,
                completion.timings.total_ms
            );
            Ok(())
        }
        Commands::Tui {
            gateway,
            orchestrator,
        } => tui::run_tui(gateway, orchestrator),
        Commands::Info => {
            models::print_catalog();
            process::print_sidecar_search_paths();
            Ok(())
        }
    }
}
