use std::{
   path::PathBuf,
   process::{Command, Stdio},
};

use console::style;

use crate::{Result, error::Error};

fn find_smgrep_root() -> Result<PathBuf> {
   let exe = std::env::current_exe()?;
   let exe_dir = exe.parent().unwrap_or(exe.as_path());

   let candidates = [
      exe_dir.to_path_buf(),
      exe_dir.join(".."),
      exe_dir.join("../.."),
      exe_dir.join("../share/smgrep"),
   ];

   for candidate in &candidates {
      let marketplace = candidate.join(".claude-plugin/marketplace.json");
      if marketplace.exists() {
         return Ok(candidate.canonicalize()?);
      }
   }
   Ok(exe_dir.to_path_buf())
}

fn run_claude_command(args: &[&str]) -> Result<()> {
   let status = Command::new("claude")
      .args(args)
      .stdin(Stdio::inherit())
      .stdout(Stdio::inherit())
      .stderr(Stdio::inherit())
      .status()
      .map_err(Error::ClaudeSpawn)?;

   if !status.success() {
      return Err(Error::ClaudeCommand(status.code().unwrap_or(-1)));
   }

   Ok(())
}

pub async fn execute() -> Result<()> {
   println!(
      "{}",
      style("Installing smgrep plugin for Claude Code...")
         .cyan()
         .bold()
   );

   let smgrep_root = find_smgrep_root()?;
   let root_path = smgrep_root.to_string_lossy();

   println!("smgrep root: {}", style(&root_path).dim());

   println!("{}", style("Adding local marketplace...").dim());
   match run_claude_command(&["plugin", "marketplace", "add", &root_path]) {
      Ok(()) => println!("{}", style("✓ Added smgrep marketplace").green()),
      Err(e) => {
         eprintln!("{}", style(format!("✗ Failed to add marketplace: {e}")).red());
         print_troubleshooting();
         return Err(e);
      },
   }

   println!("{}", style("Installing plugin...").dim());
   match run_claude_command(&["plugin", "install", "smgrep"]) {
      Ok(()) => println!("{}", style("✓ Installed smgrep plugin").green()),
      Err(e) => {
         eprintln!("{}", style(format!("✗ Failed to install plugin: {e}")).red());
         print_troubleshooting();
         return Err(e);
      },
   }

   println!();
   println!("{}", style("Next steps:").bold());
   println!("  1. Restart Claude Code if it's running");
   println!("  2. The plugin will automatically index your project when you open it");
   println!("  3. Claude will use smgrep for semantic code search automatically");
   println!("  4. You can also use `smgrep` commands directly in your terminal");

   Ok(())
}

fn print_troubleshooting() {
   eprintln!();
   eprintln!("{}", style("Troubleshooting:").yellow().bold());
   eprintln!("  • Ensure you have Claude Code installed");
   eprintln!("  • Try running: claude --version");
   eprintln!("  • Check that smgrep was installed correctly with plugin files");
}
