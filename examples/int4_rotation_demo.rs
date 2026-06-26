//! INT4 + Hadamard rotation runtime kernel: compression and error measurement.
//!
//! Measures the relative matvec error of plain INT4 group quantization vs INT4
//! with a Hadamard rotation folded in (QuaRot-style), against the exact f32
//! result, across a sweep of *outlier density* — the variable that decides
//! whether rotation helps.
//!
//! Honest finding this demo makes explicit: weight-only rotation is NOT a free
//! win for group INT4. Group quantization already *isolates* sparse outliers
//! into a few bad groups, leaving the rest precise — so when outliers are
//! sparse (fewer than ~1 per group), plain INT4 wins and rotation, by spreading
//! the outliers across every group, actually hurts. Rotation pays off only in
//! the *concentrated* regime: when (nearly) every group already contains an
//! outlier, there is nothing left to isolate, and the incoherent spread lowers
//! each group's dynamic range. The sweep below varies outliers-PER-GROUP and
//! shows exactly that crossover.
//!
//! Usage:
//!   cargo run --release --example int4_rotation_demo [rows] [cols]

use mullama::int4_rotation::{matvec_f32, Int4RotatedMatrix, GROUP_SIZE};

fn rel_err(approx: &[f32], exact: &[f32]) -> f32 {
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (a, e) in approx.iter().zip(exact) {
        num += (a - e) * (a - e);
        den += e * e;
    }
    (num / den.max(1e-12)).sqrt()
}

/// Build weights with `per_group` outliers in each 32-wide group of each row,
/// over a small base. `per_group` controls outlier *density* — the variable
/// that decides whether rotation helps.
fn make_weights(rows: usize, cols: usize, per_group: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; rows * cols];
    let groups = cols / GROUP_SIZE;
    for r in 0..rows {
        for c in 0..cols {
            w[r * cols + c] = (((r * 31 + c * 17) % 23) as f32 - 11.0) * 0.01;
        }
        for g in 0..groups {
            for k in 0..per_group.min(GROUP_SIZE) {
                let c = g * GROUP_SIZE + (r * 7 + k * 11 + 1) % GROUP_SIZE;
                let sign = if (g + k) % 2 == 0 { 1.0 } else { -1.0 };
                w[r * cols + c] = sign * (5.0 + (k % 3) as f32);
            }
        }
    }
    w
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    // Default cols a multiple of GROUP_SIZE so "per group" is exact.
    let cols: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);

    let groups_per_row = cols / GROUP_SIZE.max(1);
    let x: Vec<f32> = (0..cols).map(|i| ((i * 3 % 13) as f32 - 6.0) * 0.1).collect();

    let f32_bytes = rows * cols * std::mem::size_of::<f32>();
    let probe = Int4RotatedMatrix::from_weights(&make_weights(rows, cols, 1), rows, cols, false);
    println!("matrix: {rows} x {cols}  ({groups_per_row} groups/row, group size {GROUP_SIZE})");
    println!(
        "storage: f32 {} B -> INT4 {} B  ({:.1}x, {:.2} bytes/weight)\n",
        f32_bytes,
        probe.storage_bytes(),
        f32_bytes as f32 / probe.storage_bytes() as f32,
        probe.storage_bytes() as f32 / (rows * cols) as f32
    );

    println!("outliers/group | plain INT4 | +rotation | winner");
    println!("---------------+------------+-----------+--------");
    for &pg in &[0usize, 1, 2, 4, 8, 16] {
        let w = make_weights(rows, cols, pg);
        let mut exact = vec![0.0; rows];
        matvec_f32(&w, rows, cols, &x, &mut exact);
        let plain = Int4RotatedMatrix::from_weights(&w, rows, cols, false);
        let rot = Int4RotatedMatrix::from_weights(&w, rows, cols, true);
        let mut yp = vec![0.0; rows];
        let mut yr = vec![0.0; rows];
        plain.matvec(&x, &mut yp);
        rot.matvec(&x, &mut yr);
        let ep = rel_err(&yp, &exact);
        let er = rel_err(&yr, &exact);
        let winner = if er < ep { "rotation" } else { "plain" };
        println!(
            "{:>14} | {:>9.2}% | {:>8.2}% | {}",
            pg,
            ep * 100.0,
            er * 100.0,
            winner
        );
    }
    println!(
        "\nAs outliers/group rises, plain INT4 can no longer isolate them and \
         rotation's incoherent spread wins — the QuaRot regime."
    );
}
