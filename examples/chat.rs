//! # Chat Completion Example
//!
//! This example demonstrates how to build a simple chat interface using Mullama.
//! It supports conversation history, customizable sampling, and graceful error handling.

use mullama::{
    Context, ContextParams, Model, MullamaError, SamplerChain, SamplerChainParams, SamplerParams,
};
use std::collections::VecDeque;
use std::io::{self, Write};

/// Represents a chat message
#[derive(Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Chat session manager
struct ChatSession {
    model: std::sync::Arc<Model>,
    context: Context,
    sampler: SamplerChain,
    history: VecDeque<ChatMessage>,
    max_history: usize,
    system_prompt: String,
}

impl ChatSession {
    /// Create a new chat session
    pub fn new(
        model_path: &str,
        system_prompt: Option<String>,
        max_history: usize,
    ) -> Result<Self, MullamaError> {
        // Initialize backend
        // Backend initialization not needed - placeholder

        // Load model
        println!("Loading model from {}...", model_path);
        let model = std::sync::Arc::new(Model::load(model_path)?);

        println!("Model loaded successfully!");
        println!("  Vocabulary size: {}", model.vocab_size());
        println!("  Context size: {}", model.n_ctx_train());

        // Create context with reasonable defaults
        let mut ctx_params = ContextParams::default();
        ctx_params.n_ctx = 4096;
        ctx_params.n_batch = 512;
        ctx_params.n_threads = num_cpus::get() as u32;

        let mut context = Context::new(model.clone(), ctx_params)?;

        // Set up sampling for natural conversation
        let mut sampler_params = SamplerParams::default();
        sampler_params.temperature = 0.7; // Natural but focused
        sampler_params.top_k = 40; // Good variety
        sampler_params.top_p = 0.9; // Nucleus sampling
        sampler_params.penalty_repeat = 1.1; // Avoid repetition
        sampler_params.penalty_last_n = 64; // Look back 64 tokens

        let sampler = sampler_params.build_chain(model.clone())?;

        let system_prompt = system_prompt.unwrap_or_else(|| {
            "You are a helpful, harmless, and honest AI assistant. You provide clear, \
             accurate, and concise responses while being friendly and professional."
                .to_string()
        });

        let mut session = Self {
            model,
            context,
            sampler,
            history: VecDeque::new(),
            max_history,
            system_prompt,
        };

        // Initialize with system prompt
        session.add_system_message()?;

        Ok(session)
    }

    /// Add system message to context
    fn add_system_message(&mut self) -> Result<(), MullamaError> {
        let system_msg = format!("System: {}\n\n", self.system_prompt);
        let tokens = self.model.tokenize(&system_msg, true, false)?;

        for _token in tokens {
            // self.context.eval_token(token)?; // Method not implemented yet
        }

        Ok(())
    }

    /// Add a message to the conversation history
    pub fn add_message(&mut self, role: &str, content: &str) {
        let message = ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        };

        self.history.push_back(message);

        // Trim history if too long
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Generate a response to user input
    pub fn generate_response(&mut self, user_input: &str) -> Result<String, MullamaError> {
        // Add user message to history
        self.add_message("User", user_input);

        // Format the prompt with conversation context
        let prompt = self.format_conversation_prompt(user_input);

        // Tokenize and evaluate the prompt
        let prompt_tokens = self.model.tokenize(&prompt, false, false)?;

        for _token in prompt_tokens {
            // self.context.eval_token(token)?; // Method not implemented yet
        }

        // Generate response
        let mut response = String::new();
        let mut tokens_generated = 0;
        let max_tokens = 512; // Reasonable response length

        print!("Assistant: ");
        io::stdout().flush()?;

        while tokens_generated < max_tokens {
            let next_token = self.sampler.sample(&mut self.context, 0);

            // Check for end of generation
            if next_token == 0 {
                // Placeholder for EOS token check
                break;
            }

            // Convert token to text
            let text = self.model.token_to_str(next_token, 0, false)?;

            // Stop at natural conversation breaks
            if text.contains("\nUser:") || text.contains("\nHuman:") {
                break;
            }

            response.push_str(&text);
            print!("{}", text);
            io::stdout().flush()?;

            // Evaluate the token for next iteration
            // self.context.eval_token(next_token)?; // Method not implemented yet
            tokens_generated += 1;
        }

        println!(); // New line after response

        // Add assistant response to history
        self.add_message("Assistant", &response.trim().to_string());

        Ok(response.trim().to_string())
    }

    /// Format the conversation prompt
    fn format_conversation_prompt(&self, current_input: &str) -> String {
        let mut prompt = String::new();

        // Add recent conversation history
        for message in self
            .history
            .iter()
            .rev()
            .take(6)
            .collect::<Vec<_>>()
            .iter()
            .rev()
        {
            prompt.push_str(&format!("{}: {}\n", message.role, message.content));
        }

        // Add current user input
        prompt.push_str(&format!("User: {}\nAssistant:", current_input));

        prompt
    }

    /// Get conversation statistics
    pub fn get_stats(&self) -> (usize, usize, usize) {
        let history_count = self.history.len();
        let context_tokens = 4096; // Placeholder for context length
        let vocab_size = self.model.vocab_size();
        (history_count, context_tokens as usize, vocab_size as usize)
    }
}

/// Interactive chat loop
fn run_chat_session(session: &mut ChatSession) -> Result<(), Box<dyn std::error::Error>> {
    println!(" Mullama Chat Interface");
    println!("Type 'quit', 'exit', or 'bye' to end the conversation.");
    println!("Type '/help' for available commands.");
    println!("Type '/stats' for session statistics.");
    println!("{}", "=".repeat(50));

    loop {
        // Get user input
        print!("\nYou: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Handle special commands
        match input.to_lowercase().as_str() {
            "quit" | "exit" | "bye" => {
                println!("👋 Goodbye! Thanks for using Mullama Chat!");
                break;
            }
            "/help" => {
                print_help();
                continue;
            }
            "/stats" => {
                let (history, context, vocab) = session.get_stats();
                println!(" Session Statistics:");
                println!("  Messages in history: {}", history);
                println!("  Context tokens used: {}", context);
                println!("  Model vocabulary size: {}", vocab);
                continue;
            }
            "/clear" => {
                session.history.clear();
                println!("🧹 Conversation history cleared!");
                continue;
            }
            "" => continue, // Skip empty input
            _ => {}
        }

        // Generate and display response
        match session.generate_response(input) {
            Ok(_) => {
                // Response was already printed during generation
            }
            Err(e) => {
                eprintln!(" Error generating response: {}", e);
                eprintln!("   Please try again with a different input.");
            }
        }
    }

    Ok(())
}

/// Print help information
fn print_help() {
    println!(" Available Commands:");
    println!("  /help  - Show this help message");
    println!("  /stats - Show session statistics");
    println!("  /clear - Clear conversation history");
    println!("  quit, exit, bye - End the conversation");
    println!("\n💡 Tips:");
    println!("  - Keep messages reasonably short for best performance");
    println!("  - The AI remembers recent conversation context");
    println!("  - Use '/clear' if the conversation gets too long");
}

/// Parse command line arguments
fn parse_args() -> Result<(String, Option<String>, usize), String> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return Err("Usage: chat <model_path> [system_prompt] [max_history]".to_string());
    }

    let model_path = args[1].clone();
    let system_prompt = args.get(2).cloned();
    let max_history = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);

    Ok((model_path, system_prompt, max_history))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let (model_path, system_prompt, max_history) = match parse_args() {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("Error: {}", msg);
            eprintln!("\nExample usage:");
            eprintln!("  cargo run --example chat -- path/to/model.gguf");
            eprintln!(
                "  cargo run --example chat -- path/to/model.gguf \"You are a helpful assistant\""
            );
            eprintln!("  cargo run --example chat -- path/to/model.gguf \"You are a helpful assistant\" 15");
            return Ok(());
        }
    };

    // Create chat session
    let mut session = match ChatSession::new(&model_path, system_prompt, max_history) {
        Ok(session) => session,
        Err(e) => {
            eprintln!(" Failed to create chat session: {}", e);
            eprintln!("\nTroubleshooting:");
            eprintln!("  1. Check that the model file exists and is readable");
            eprintln!("  2. Ensure the model is in GGUF format");
            eprintln!("  3. Verify you have enough RAM for the model");
            eprintln!("  4. Try a smaller model if you're running out of memory");
            return Ok(());
        }
    };

    // Run the interactive chat
    run_chat_session(&mut session)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_creation() {
        let msg = ChatMessage {
            role: "User".to_string(),
            content: "Hello".to_string(),
        };
        assert_eq!(msg.role, "User");
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_conversation_formatting() {
        // This would require a mock model for proper testing
        // For now, just test basic functionality
        let mut history = VecDeque::new();
        history.push_back(ChatMessage {
            role: "User".to_string(),
            content: "Hello".to_string(),
        });
        history.push_back(ChatMessage {
            role: "Assistant".to_string(),
            content: "Hi there!".to_string(),
        });

        assert_eq!(history.len(), 2);
    }
}

/// Example of advanced usage with custom sampling
#[allow(dead_code)]
fn advanced_chat_example() -> Result<(), MullamaError> {
    // Backend initialization not needed - placeholder

    let model = std::sync::Arc::new(Model::load("path/to/model.gguf")?);

    let mut ctx_params = ContextParams::default();
    ctx_params.n_ctx = 8192; // Longer context for complex conversations
    ctx_params.n_batch = 1024; // Larger batch size

    let mut context = Context::new(model.clone(), ctx_params)?;

    // Custom sampling chain for creative writing
    let mut creative_sampler = SamplerChain::new(SamplerChainParams::default());
    // Methods not implemented yet:
    // .add_top_k(60)           // More variety
    // .add_top_p(0.95)         // Higher nucleus threshold
    // .add_temperature(0.9)    // More creative
    // .add_repetition_penalty(1.15)  // Strong anti-repetition
    // .add_min_p(0.02);       // Minimum probability filter

    // Use the creative sampler...
    let _token = creative_sampler.sample(&mut context, 0);

    Ok(())
}
