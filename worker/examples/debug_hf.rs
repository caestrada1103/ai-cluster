use hf_hub::{api::tokio::Api, Repo, RepoType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("HF_ENDPOINT: {:?}", std::env::var("HF_ENDPOINT"));
    println!("HF_TOKEN: {:?}", std::env::var("HF_TOKEN").ok().map(|s| s.chars().take(4).collect::<String>() + "..."));

    let api = hf_hub::api::tokio::ApiBuilder::new()
        .with_endpoint("https://huggingface.co".to_string())
        .build()?;
    let repo = api.repo(Repo::new("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(), RepoType::Model));
    
    println!("Fetching config.json...");
    let path = repo.get("config.json").await;
    match path {
        Ok(p) => println!("Success: {:?}", p),
        Err(e) => println!("Error: {:?}", e),
    }
    Ok(())
}
