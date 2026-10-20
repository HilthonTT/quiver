//! Minimal embedded usage.
//!
//! ```sh
//! cargo run --example quickstart
//! ```

use serde_json::json;
use vectordb_db::{CollectionConfig, Database, Filter, Metric, Point, SearchOptions};

fn main() -> vectordb_db::Result<()> {
    let dir = std::env::temp_dir().join("vector-db-quickstart");
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(&dir)?;

    let books = db.create_collection("books", CollectionConfig::new(3).metric(Metric::Cosine))?;
    books.upsert([
        Point::new("dune", vec![0.9, 0.1, 0.0])
            .with_metadata(json!({"genre": "sci-fi", "year": 1965})),
        Point::new("foundation", vec![0.8, 0.2, 0.1])
            .with_metadata(json!({"genre": "sci-fi", "year": 1951})),
        Point::new("hobbit", vec![0.1, 0.9, 0.2])
            .with_metadata(json!({"genre": "fantasy", "year": 1937})),
        Point::new("emma", vec![0.0, 0.2, 0.9])
            .with_metadata(json!({"genre": "romance", "year": 1815})),
    ])?;

    println!("Nearest to [1, 0, 0]:");
    for hit in books.search(&[1.0, 0.0, 0.0], 3)? {
        println!("  {:<12} distance={:.4}", hit.id, hit.distance);
    }

    let filter = Filter::parse(&json!({"year": {"$lt": 1960}}))?;
    println!("\nNearest published before 1960:");
    for hit in books.search_with(
        &[1.0, 0.0, 0.0],
        3,
        &SearchOptions::default().filter(filter),
    )? {
        println!("  {:<12} {}", hit.id, hit.metadata.unwrap_or_default());
    }

    db.flush()?;
    println!("\nPersisted to {}", dir.display());
    Ok(())
}
