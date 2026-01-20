//! Example demonstrating safe file processing with worm_std.

use worm_std::fs::{self, File};
use worm_std::io::{self, BufRead, BufReader, Write};
use worm_std::path::Path;

fn main() -> io::Result<()> {
    println!("Worm Rust File Processing Example");
    println!("==================================");
    println!();
    
    // Create a temporary file
    let temp_path = Path::new("worm_example_temp.txt");
    
    // Write some data
    {
        let mut file = File::create(temp_path)?;
        writeln!(file, "Line 1: Hello from Worm Rust!")?;
        writeln!(file, "Line 2: This is secure file I/O.")?;
        writeln!(file, "Line 3: No network access possible!")?;
        println!("Created file: {}", temp_path.display());
    }
    
    // Read it back
    {
        let file = File::open(temp_path)?;
        let reader = BufReader::new(file);
        
        println!("File contents:");
        for (i, line) in reader.lines().enumerate() {
            println!("  {}: {}", i + 1, line?);
        }
    }
    
    // Get file metadata
    let metadata = fs::metadata(temp_path)?;
    println!("File size: {} bytes", metadata.len());
    
    // Clean up
    fs::remove_file(temp_path)?;
    println!("Cleaned up temporary file.");
    
    println!();
    println!("File processing completed successfully!");
    println!("Note: All operations were local - no network access used.");
    
    Ok(())
}
