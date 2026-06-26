use super::args::Cli;
use super::{
    cli_stop_daemon, copy_model, create_model, embed_text, handle_cache_action,
    handle_daemon_action, list_all_models, list_models, load_model, ping_daemon, pull_model,
    remove_model, run_chat, run_model_with_prompt, run_server, search_models, set_default,
    show_model_details, show_repo_info, show_running_models, show_status, tokenize_text,
    unload_model, Commands,
};

pub(crate) async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::List { verbose, json } => {
            list_all_models(verbose, json).await?;
        }
        Commands::Rm { name, force } => {
            remove_model(&name, force).await?;
        }
        Commands::Ps { socket, json } => {
            show_running_models(&socket, json).await?;
        }
        Commands::Show {
            name,
            modelfile,
            json,
        } => {
            show_model_details(&name, modelfile, json).await?;
        }
        Commands::Serve {
            model,
            mmproj,
            socket,
            http_port,
            http_addr,
            api_key,
            require_api_key,
            max_tokens_limit,
            max_request_body_mb,
            max_concurrent_requests,
            max_requests_per_second,
            gpu_layers,
            context_size,
            context_pool_size,
            threads,
            verbose,
            tls_cert,
            tls_key,
            flash_attn,
            cache_type_k,
            cache_type_v,
            no_mmap,
            mlock,
            batch_size,
            rope_freq_base,
            rope_freq_scale,
            split_mode,
            defrag_thold,
            hydration,
        } => {
            if tls_cert.is_some() || tls_key.is_some() {
                eprintln!("Note: TLS is not yet supported natively. Use a reverse proxy (nginx, caddy) for HTTPS.");
            }
            run_server(
                model,
                mmproj,
                socket,
                http_port,
                http_addr,
                api_key,
                require_api_key,
                max_tokens_limit,
                max_request_body_mb,
                max_concurrent_requests,
                max_requests_per_second,
                gpu_layers,
                context_size,
                context_pool_size,
                threads,
                verbose,
                flash_attn,
                cache_type_k,
                cache_type_v,
                no_mmap,
                mlock,
                batch_size,
                rope_freq_base,
                rope_freq_scale,
                split_mode,
                defrag_thold,
                hydration,
            )
            .await?;
        }
        Commands::Chat { socket, timeout } => {
            run_chat(&socket, timeout)?;
        }
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            socket,
            image,
            http_port,
            gpu_layers,
            context_size,
            stats,
            flash_attn,
            cache_type_k,
            cache_type_v,
            no_mmap,
            mlock,
            batch_size,
        } => {
            run_model_with_prompt(
                &model,
                prompt.as_deref(),
                max_tokens,
                temperature,
                &socket,
                image.as_ref(),
                http_port,
                gpu_layers,
                context_size,
                stats,
                flash_attn,
                cache_type_k,
                cache_type_v,
                no_mmap,
                mlock,
                batch_size,
            )
            .await?;
        }
        Commands::Models { socket, verbose } => {
            list_models(&socket, verbose)?;
        }
        Commands::Load {
            spec,
            gpu_layers,
            context_size,
            mmproj,
            flash_attn,
            cache_type_k,
            cache_type_v,
            no_mmap,
            mlock,
            socket,
        } => {
            load_model(
                &socket,
                &spec,
                gpu_layers,
                context_size,
                mmproj,
                flash_attn,
                cache_type_k,
                cache_type_v,
                no_mmap,
                mlock,
            )?;
        }
        Commands::Unload { alias, socket } => {
            unload_model(&socket, &alias)?;
        }
        Commands::Default { alias, socket } => {
            set_default(&socket, &alias)?;
        }
        Commands::Status { socket, json } => {
            show_status(&socket, json)?;
        }
        Commands::Ping { socket } => {
            ping_daemon(&socket)?;
        }
        Commands::Stop { socket, force } => {
            cli_stop_daemon(&socket, force)?;
        }
        Commands::Tokenize {
            text,
            model,
            socket,
        } => {
            tokenize_text(&socket, &text, model.as_deref())?;
        }
        Commands::Embed {
            text,
            model,
            socket,
            json,
        } => {
            embed_text(&socket, &text, model.as_deref(), json)?;
        }
        Commands::Pull { spec, quiet } => {
            pull_model(&spec, !quiet).await?;
        }
        Commands::Cache { action } => {
            handle_cache_action(action).await?;
        }
        Commands::Search {
            query,
            limit,
            all,
            files,
        } => {
            search_models(&query, limit, !all, files).await?;
        }
        Commands::Info { repo } => {
            show_repo_info(&repo).await?;
        }
        Commands::Create {
            name,
            file,
            download,
            quiet,
        } => {
            create_model(&name, file, download, !quiet).await?;
        }
        Commands::Cp {
            source,
            destination,
        } => {
            copy_model(&source, &destination).await?;
        }
        Commands::Daemon { action } => {
            handle_daemon_action(action)?;
        }
    }

    Ok(())
}
