#!/usr/bin/env python3
import argparse
import math
import yaml
import sys
from tabulate import tabulate


def debug_log(debug, message):
    """Print debug message if debug mode is enabled."""
    if debug:
        print(f"[DEBUG] {message}")


def load_yaml(file_path: str, debug=False, is_config=False):
    """Load a YAML file and return as Python dict."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        if is_config:
            print(f"[INFO] Loaded configuration file: {file_path}")
        else:
            debug_log(debug, f"Loaded YAML: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        sys.exit(1)


def print_gpu_selection_mode(is_auto):
    """Print GPU selection mode."""
    mode = "auto-selected" if is_auto else "user-defined"
    print(f"[INFO] GPU selection mode: {mode}")


def derive_tps_req(workload_params, debug=False):
    """Derive TPS requirement from workload parameters if not explicitly given."""
    tps_req = workload_params.get("TPS_REQ")
    if tps_req:
        # Compare to calculated value if possible
        if all(k in workload_params and workload_params[k] is not None
               for k in ["avg_input_tokens", "avg_output_tokens", "requests_per_sec"]):
            avg_in = workload_params["avg_input_tokens"]
            avg_out = workload_params["avg_output_tokens"]
            rps = workload_params["requests_per_sec"]
            implied_tps = (avg_in + avg_out) * rps
            debug_log(debug, f"Using provided TPS_REQ: {tps_req} "
                             f"(Calculated from tokens & RPS: {implied_tps:.2f})")
        else:
            debug_log(debug, f"Using provided TPS_REQ: {tps_req}")
        return float(tps_req)

    # Must have requests_per_sec for derivation
    if "requests_per_sec" not in workload_params or workload_params["requests_per_sec"] is None:
        print("[ERROR] 'requests_per_sec' missing in workload_params in params.yaml. Please provide it or set TPS_REQ.")
        sys.exit(1)

    avg_in = workload_params["avg_input_tokens"]
    avg_out = workload_params["avg_output_tokens"]
    rps = workload_params["requests_per_sec"]
    debug_log(debug, f"Using provided requests_per_sec: {rps}")
    derived = (avg_in + avg_out) * rps
    debug_log(debug, f"Derived TPS_REQ: ({avg_in} + {avg_out}) * {rps} = {derived}")
    return derived


def compute_model_memory(model_params, workload_params, debug=False):
    """Estimate the model memory requirements."""
    params = model_params["PARAMS"]
    precision_bits = model_params["PREC_BITS"]
    layers = model_params["LAYERS"]
    d_model = model_params["D_MODEL"]
    seq_len_kv = workload_params["avg_output_tokens"]

    ##adding inference overhead: 0.2 activation, .1 for temp buffers
    weight_mem = params * (precision_bits / 8) / (1024**3) * 1.3
    kv_mem = 2 * layers * seq_len_kv * d_model * (precision_bits / 8) / (1024**3)
    total_mem = weight_mem + kv_mem
    debug_log(debug, f"Memory Calculation: weights={weight_mem:.2f} GB, KV_cache={kv_mem:.2f} GB, total={total_mem:.2f} GB")
    return weight_mem, kv_mem, total_mem

def scale_mlperf_tps(mlperf_tps_raw, mlperf_seq_len_assumed, actual_seq_len, method, debug=False):
    """
    Scale MLPerf TPS results to match actual sequence length.

    Adds a safeguard for extremely long sequence lengths (ratio < 0.2)
    to avoid unrealistic drops in per-GPU TPS.
    """
    ratio = mlperf_seq_len_assumed / actual_seq_len
    chosen_method = method
    factor = 1.0

    if method == "adaptive":
        if 0.75 <= ratio <= 1.25:
            chosen_method, factor = "none", 1.0
        elif 0.2 < ratio <= 5:
            chosen_method, factor = "sqrt", math.sqrt(ratio)
        else:
            chosen_method = "blend"
            sqrt_factor = math.sqrt(ratio)
            linear_factor = ratio
            factor = 0.5 * sqrt_factor + 0.5 * linear_factor
            if ratio < 0.1 or ratio > 10:
                factor = 0.6 * sqrt_factor + 0.4 * linear_factor
    elif method == "none":
        factor = 1.0
        chosen_method = "none"
    elif method == "linear":
        factor = ratio
        chosen_method = "linear"
    elif method == "sqrt":
        factor = math.sqrt(ratio)
        chosen_method = "sqrt"
    else:
        raise ValueError(f"Unknown MLPerf scaling method: {method}")

    # === Safeguard for extreme long-sequence workloads ===
    # If sequence length >> MLPerf profile (ratio < 0.2), don't let factor drop below this threshold
    min_factor_long_seq = 0.20  # minimum 20% of MLPerf TPS
    if ratio < 0.2 and factor < min_factor_long_seq:
        debug_log(debug, f"Long seq_len → clamping scaling factor from {factor:.4f} to {min_factor_long_seq:.2f}")
        factor = min_factor_long_seq

    scaled_tps = mlperf_tps_raw * factor
    return scaled_tps, chosen_method


def compute_throughput_requirements(model_params, gpu, workload_params, tps_req,
                                    efficiency_factor, mlperf_tps_db, mlperf_scale_method, debug=False):
    """Determine GPU throughput limits, source, and bottleneck type."""
    gpu_name = gpu["name"]
    actual_seq_len = workload_params["avg_input_tokens"] + workload_params["avg_output_tokens"]

    flops_per_token = 2 * model_params["D_MODEL"] * model_params["D_MODEL"]
    tps_compute = (gpu["TFLOPS_FP16"] * 1e12 / flops_per_token) * efficiency_factor
    bytes_per_token = model_params["D_MODEL"] * (model_params["PREC_BITS"] / 8)
    tps_bw = (gpu["BW"] * 1e9 / bytes_per_token) * efficiency_factor

    tps_source, chosen_scale_method = "calculated", "none"

    if gpu_name in mlperf_tps_db:
        tps_source = "mlperf"
        debug_log(debug, f"{gpu_name}: TPS Source: {tps_source} (MLPerf reference + scaling)")

        mlperf_tps_raw = mlperf_tps_db[gpu_name]
        mlperf_seq_len_assumed = gpu.get("MLPERF_SEQ_LEN_ASSUMED", 1024)

        adj_mlperf_tps, chosen_scale_method = scale_mlperf_tps(
            mlperf_tps_raw, mlperf_seq_len_assumed, actual_seq_len, mlperf_scale_method, debug
        )
        per_gpu_tps = adj_mlperf_tps

        # Bottleneck determination using closeness to theoretical limits
        diff_compute = abs(per_gpu_tps - tps_compute)
        diff_bw = abs(per_gpu_tps - tps_bw)
        if diff_compute < diff_bw * 0.9:
            hw_bottleneck = "Compute"
        elif diff_bw < diff_compute * 0.9:
            hw_bottleneck = "Bandwidth"
        else:
            hw_bottleneck = "Mixed"

        debug_log(debug,
                  f"{gpu_name}: MLPerf scaled TPS={per_gpu_tps:.2f} vs. "
                  f"Compute-limit={tps_compute:.2f}, BW-limit={tps_bw:.2f} → Bottleneck={hw_bottleneck}")

    else:
        tps_source = "calculated"
        debug_log(debug, f"{gpu_name}: TPS Source: {tps_source} (min of theoretical Compute/BW)")

        per_gpu_tps = min(tps_compute, tps_bw)
        hw_bottleneck = "Compute" if tps_compute < tps_bw else "Bandwidth"
        
        debug_log(debug,
                  f"{gpu_name}: Compute-limit={tps_compute:.2f}, BW-limit={tps_bw:.2f} → "
                  f"Using {per_gpu_tps:.2f} TPS → Bottleneck={hw_bottleneck}")

    # GPU count requirement
    calc_gpus_float = tps_req / per_gpu_tps
    gpus_needed_thr = math.ceil(calc_gpus_float)
    
    debug_log(debug,
              f"{gpu_name}: GPUs needed (raw)={calc_gpus_float:.2f}, ceil={gpus_needed_thr} "
              f"(TPS_REQ={tps_req:.2f}, per_gpu_tps={per_gpu_tps:.2f})")

    return gpus_needed_thr, hw_bottleneck, per_gpu_tps, tps_compute, tps_bw, tps_source, chosen_scale_method


def evaluate_gpu(model_params, workload_params, efficiency_factor, tps_req,
                 gpu_name, specs, mlperf_tps_db, mlperf_scale_method, debug=False, prompt_phase_factor=0.01):
    """Evaluate a specific GPU given the model and workload parameters."""
    gpu = specs.copy()
    gpu["name"] = gpu_name

    weight_mem, kv_mem, total_mem_needed = compute_model_memory(model_params, workload_params, debug)
    gpus_mem = math.ceil(total_mem_needed / gpu["VRAM"])

    gpus_thr, bottleneck, per_gpu_tps, tps_compute, tps_bw, tps_source, scale_method = compute_throughput_requirements(
        model_params, gpu, workload_params, tps_req, efficiency_factor, mlperf_tps_db, mlperf_scale_method, debug
    )

    avg_in_tokens = workload_params["avg_input_tokens"]
    prompt_tps = per_gpu_tps * prompt_phase_factor
    ttft_ms = (avg_in_tokens / prompt_tps) * 1000
    if debug:
        debug_log(debug, f"{gpu_name}: Estimated TTFT = {ttft_ms:.2f} ms using prompt phase factor={prompt_phase_factor}")
        if workload_params.get("CONCUR"):
            user_rps = workload_params.get("requests_per_sec", "N/A")
            est_rps_from_ttft = workload_params["CONCUR"] / (ttft_ms / 1000)
            debug_log(debug, (f"{gpu_name}: Estimated RPS from TTFT = {est_rps_from_ttft:.2f} req/s "
                             f"(user-provided RPS={user_rps}, CONCUR={workload_params['CONCUR']})"))

    # Constraint determination
    if gpus_mem > gpus_thr:
        constraint_type = "Memory Bound"
        logical_gpus = gpus_mem
        print(f"[RESULT] {gpu_name}: Memory bound - Needs {total_mem_needed:.2f} GB but VRAM is {gpu['VRAM']} GB")
    elif gpus_thr > gpus_mem:
        constraint_type = "Throughput Bound"
        logical_gpus = gpus_thr
        print(f"[RESULT] {gpu_name}: Throughput bound ({bottleneck}) - Requires {tps_req:.2f} TPS vs available {per_gpu_tps:.2f} TPS [{tps_source}, scale={scale_method}]")
    else:
        constraint_type = "Tie"
        logical_gpus = gpus_mem
        print(f"[RESULT] {gpu_name}: Tie between Memory and Throughput requirements.")

    # MIG slicing
    if gpu.get("MIG_SLICES", 0) > 0:
        slices_needed = logical_gpus
        physical_gpus = math.ceil(slices_needed / gpu["MIG_SLICES"])
        fractional_cost = slices_needed * (gpu["COST"] / gpu["MIG_SLICES"])
    else:
        slices_needed = None
        physical_gpus = logical_gpus
        fractional_cost = None

    total_cost = physical_gpus * gpu["COST"]
    cloud_hr = gpu.get("CLOUD_PRICE_HR")
    cloud_total_hr = physical_gpus * cloud_hr if cloud_hr else None
    cloud_month = cloud_total_hr * 24 * 30 if cloud_total_hr else None

    mig_info = {}
    if slices_needed is not None:
        mig_info = {
            "MIG_SLICES": gpu["MIG_SLICES"],
            "Slices_Needed": slices_needed,
            "Physical_GPUs": physical_gpus,
            "Fractional_Cost": fractional_cost
        }

    return {
        "name": gpu_name,
        "constraint": constraint_type,
        "logical_gpus": logical_gpus,
        "physical_gpus": physical_gpus,
        "bottleneck": bottleneck if constraint_type == "Throughput Bound" else "N/A",
        "per_gpu_tps": per_gpu_tps,
        "tps_compute": tps_compute,
        "tps_bw": tps_bw,
        "weight_mem": weight_mem,
        "kv_mem": kv_mem,
        "total_mem_needed": total_mem_needed,
        "total_cost": total_cost,
        "mig_info": mig_info,
        "COST": gpu["COST"],
        "cloud_price_hr": cloud_hr,
        "total_cloud_hr": cloud_total_hr,
        "total_cloud_month": cloud_month
    }


def main():
    parser = argparse.ArgumentParser(description="LLM GPU Sizing Tool")
    parser.add_argument("-c", "--config", default="params.yaml")
    parser.add_argument("-g", "--gpu-db", default="gpu_db.yaml")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--debug-file")
    parser.add_argument("--mlperf-scale", choices=["sqrt", "linear", "none", "adaptive"], default="adaptive")
    args = parser.parse_args()

    if args.debug_file:
        sys.stdout = open(args.debug_file, "w")

    config = load_yaml(args.config, args.debug, is_config=True)
    gpu_db_full = load_yaml(args.gpu_db, args.debug)
    model_params = config["model_params"]
    workload_params = config["workload_params"]
    efficiency_factor = config["efficiency_factor"]["value"]
    gpus_per_server = workload_params.get("GPUS_PER_SERVER", 8)
    gpu_db = gpu_db_full["NVIDIA_GPU_DB"]
    mlperf_tps_db = gpu_db_full["MLPERF_REF_TPS"]

    prompt_phase_factor = config.get("inference_params", {}).get("prompt_phase_factor", 0.01)
    tps_req = derive_tps_req(workload_params, args.debug)

    if args.debug:
        debug_log(True,
                  f"Workload summary: avg_input_tokens={workload_params['avg_input_tokens']}, "
                  f"avg_output_tokens={workload_params['avg_output_tokens']}, "
                  f"requests_per_sec={workload_params.get('requests_per_sec', 'N/A')}, "
                  f"TPS_REQ={tps_req:.2f}")

    if not config.get("gpu_params") or not config["gpu_params"].get("GPU_TYPE"):
        print_gpu_selection_mode(True)
        evaluations = [
            evaluate_gpu(model_params, workload_params, efficiency_factor, tps_req,
                         name, specs, mlperf_tps_db, args.mlperf_scale, args.debug, prompt_phase_factor)
            for name, specs in gpu_db.items()
        ]
        evaluations.sort(key=lambda x: x["total_cost"])
        selected_gpu = evaluations[0]
    else:
        print_gpu_selection_mode(False)
        gpu_specs = gpu_db[config["gpu_params"]["GPU_TYPE"]]
        selected_gpu = evaluate_gpu(model_params, workload_params, efficiency_factor, tps_req,
                                    config["gpu_params"]["GPU_TYPE"], gpu_specs,
                                    mlperf_tps_db, args.mlperf_scale, args.debug, prompt_phase_factor)
        evaluations = [selected_gpu]

    servers_needed = math.ceil(selected_gpu["physical_gpus"] / gpus_per_server)

    print("\n===== GPU Sizing Summary =====\n")
    print(tabulate([
        ["GPU Selection", "auto-selected" if not config.get("gpu_params") or not config["gpu_params"].get("GPU_TYPE") else "user-defined"],
        ["Primary Constraint", selected_gpu["constraint"]],
        ["Selected GPU", selected_gpu["name"]],
        ["Physical GPUs Needed", selected_gpu["physical_gpus"]],
        ["Servers Needed", servers_needed],
        ["Bottleneck (if throughput)", selected_gpu["bottleneck"]],
        ["Total Cost", f"${selected_gpu['total_cost']:,.2f}"],
        ["Cloud Hourly Cost", f"${selected_gpu['total_cloud_hr']:,.2f}" if selected_gpu['total_cloud_hr'] else "N/A"],
        ["Cloud Monthly Cost", f"${selected_gpu['total_cloud_month']:,.2f}" if selected_gpu['total_cloud_month'] else "N/A"]
    ], tablefmt="grid"))

    if selected_gpu["mig_info"]:
        print("\n--- Fractional GPU (MIG) Breakdown ---\n")
        print(tabulate([
            ["Slices per GPU", selected_gpu["mig_info"]["MIG_SLICES"]],
            ["Slices Needed (Logical GPUs)", selected_gpu["mig_info"]["Slices_Needed"]],
            ["Physical GPUs (MIG)", selected_gpu["mig_info"]["Physical_GPUs"]],
            ["Fractional Cost", f"${selected_gpu['mig_info']['Fractional_Cost']:,.2f}"]
        ], tablefmt="grid"))

    print("\n=== Recommendation ===")
    rec_line = f"Use {selected_gpu['physical_gpus']}× {selected_gpu['name']}"
    if selected_gpu["mig_info"]:
        rec_line += f" (MIG-sliced into {selected_gpu['mig_info']['Slices_Needed']} units)"
    rec_line += f" to meet {selected_gpu['constraint'].lower()} requirement"
    if selected_gpu['bottleneck'] != "N/A":
        rec_line += f" (bottleneck: {selected_gpu['bottleneck']})"
    rec_line += "."
    print(rec_line)
    print(f"This requires {servers_needed} physical server(s) with {gpus_per_server} GPUs/server capacity.")
    print(f"Estimated on-prem cost: ${selected_gpu['total_cost']:,.2f}")
    if selected_gpu['total_cloud_hr']:
        print(f"Estimated cloud cost: ${selected_gpu['total_cloud_hr']:.2f}/hr (${selected_gpu['total_cloud_month']:.2f}/month)")
    print("NOTE: This tool is intended to support preliminary planning and estimation. Validate all results before implementation.")

if __name__ == "__main__":
    main()