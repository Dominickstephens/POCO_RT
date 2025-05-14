# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# -*- coding: utf-8 -*-
import os
# Set environment variable for headless rendering if needed *before* importing OpenGL-dependent libraries
# os.environ['PYOPENGL_PLATFORM'] = 'egl' # Uncomment if necessary, e.g., on servers without display

# --- Standard Imports ---
import sys
import cv2
import time
import joblib # Used in original script logic
import torch
import argparse
import numpy as np
from os.path import join, isfile, isdir, basename, dirname
import tempfile # Used in original script logic

# --- Add torch_tensorrt import ---
# This is crucial for loading TorchScript files containing TensorRT engines
try:
    import torch_tensorrt
    print("DEMO: torch_tensorrt imported successfully.")
except ImportError:
    print("DEMO ERROR: Failed to import torch_tensorrt. "
          "This is required to load TensorRT models. "
          "Please install it (e.g., 'pip install torch-tensorrt').")
    sys.exit(1)
# --- End torch_tensorrt import ---


# --- pocolib Imports (Ensure these paths are correct) ---
try:
    from pocolib.core.tester import POCOTester
    from pocolib.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis
    from pocolib.utils.demo_utils import (
        download_youtube_clip,
        video_to_images,
        images_to_video,
        convert_crop_cam_to_orig_img,
    )
    from pocolib.utils.vibe_image_utils import get_single_image_crop_demo
    from pocolib.utils.image_utils import calculate_bbox_info, calculate_focal_length
    from pocolib.utils.vibe_renderer import Renderer
    # Make sure multi_person_tracker is findable in your environment
    from multi_person_tracker import MPT # type: ignore
except ImportError as e:
    print(f"ERROR: Failed to import pocolib components. Make sure pocolib is installed and in PYTHONPATH.")
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Imports for Socket Communication ---
import socket
import pickle
import zlib
import struct
import traceback

# --- Imports for Metrics, GUI, CSV, Deque, and Threading ---
import csv
import tkinter as tk
from tkinter import ttk
import threading
from collections import deque # For metric buffering
import queue # For passing data to metric thread
# ----------------------------------------------------------

MIN_NUM_FRAMES = 0

# Define connection details for remote_relay.py
RELAY_HOST = 'localhost'
RELAY_PORT = 9999 # Port for demo -> relay communication

# Define expected pose body dimension and betas for SMPL data sending
POSE_BODY_DIM = 69 # For basic 'smpl' model used in relay/viewer
NUM_BETAS = 10 # Standard number of shape parameters for SMPL data and metrics

MAGIC = b'SMPL'    # 4-byte magic for socket protocol

# --- Define Input/Output Keys for TensorRT Model (MUST MATCH WRAPPER) ---
# Order matters!
TRT_INPUT_KEYS = [
    'img', 'bbox_info', 'focal_length', 'scale', 'center', 'orig_shape'
]
TRT_OUTPUT_KEYS = [
    'pred_pose', 'pred_shape', 'pred_cam', 'smpl_vertices'
]
# --- End TensorRT Key Definitions ---

# --- Global variables for metrics and GUI (from metrics script) ---
current_display_metrics = {"Processing FPS": 0.0, "End-to-End Latency (ms)": 0.0}
METRIC_BUFFER_DURATION_SECONDS = 5
MAX_BUFFER_LEN = int(60 * (METRIC_BUFFER_DURATION_SECONDS + 2))

fps_buffer = deque(maxlen=MAX_BUFFER_LEN)
latency_buffer = deque(maxlen=MAX_BUFFER_LEN)
pose_change_buffer = deque(maxlen=MAX_BUFFER_LEN)
translation_change_buffer = deque(maxlen=MAX_BUFFER_LEN)
joint_pos_change_buffer = deque(maxlen=MAX_BUFFER_LEN)
shape_var_buffer = deque(maxlen=MAX_BUFFER_LEN)
detection_rate_buffer = deque(maxlen=MAX_BUFFER_LEN)

main_thread_prev_model_outputs = {
    "pred_rotmat_body": None, "pred_cam_t": None, "pred_vertices": None,
    "valid_for_comparison": False
}

CSV_FILENAME = "pocolib_demo_metrics_avg.csv" # Adapted CSV filename
CSV_FIELDNAMES = [
    "Timestamp", "Condition", "Avg Processing FPS", "Avg End-to-End Latency (ms)",
    "Avg Pose Change (Euclidean Dist)", "Avg Translation Change (mm)",
    "Avg Joint Position Change (mm)", "Avg Shape Param Variance", "Avg Detection/Tracking Rate (%)"
]

gui_root = None
condition_var = None
app_running = True # General application control flag, will be managed by main loop
metric_data_queue = queue.Queue(maxsize=10)
metric_thread_instance = None
gui_thread_instance = None
# --- End Metrics Globals ---


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def send_data(sock, data):
    """Serialize and send data with size prefix (original from pocolib demo)."""
    if sock is None:
        return False
    try:
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        length  = len(payload)
        crc32   = zlib.crc32(payload) & 0xFFFFFFFF
        header  = MAGIC + struct.pack('>I I', length, crc32)
        sock.sendall(header + payload)
        return True
    except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as e:
        print(f"Socket error during send: {e}")
        return False
    except pickle.PicklingError as e:
        print(f"Pickle error during send: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during send: {e}")
        traceback.print_exc()
        return False

# --- Metric Calculation Thread Function (from metrics script) ---
def metrics_calculation_worker():
    """
    Worker thread that runs in the background.
    It gets processed model outputs from a queue, calculates detailed metrics,
    and populates the global metric buffers.
    """
    global app_running, metric_data_queue
    global pose_change_buffer, translation_change_buffer, joint_pos_change_buffer
    global shape_var_buffer, detection_rate_buffer

    print("METRICS_THREAD: Metric calculation thread started.")
    while app_running:
        try:
            data_packet = metric_data_queue.get(timeout=0.1)
            if data_packet is None: # Sentinel value to terminate
                break

            current_ts = data_packet["timestamp"]
            current_model_outputs = data_packet["current_outputs"]
            prev_model_outputs_for_comp = data_packet["prev_outputs_for_comparison"]
            instant_detection_rate = data_packet["detection_rate"]

            instant_shape_var, instant_pose_change, instant_trans_change, instant_joint_change = np.nan, np.nan, np.nan, np.nan

            if current_model_outputs:
                pred_rotmat_curr_np = current_model_outputs["pred_rotmat_body"]
                pred_cam_t_curr_np = current_model_outputs["pred_cam_t"]
                pred_vertices_curr_np = current_model_outputs["pred_vertices"]
                pred_betas_curr_np = current_model_outputs["pred_betas"]

                if pred_betas_curr_np is not None:
                    instant_shape_var = np.var(pred_betas_curr_np)

                if prev_model_outputs_for_comp and prev_model_outputs_for_comp["valid_for_comparison"]:
                    pred_rotmat_prev_np = prev_model_outputs_for_comp["pred_rotmat_body"]
                    pred_cam_t_prev_np = prev_model_outputs_for_comp["pred_cam_t"]
                    pred_vertices_prev_np = prev_model_outputs_for_comp["pred_vertices"]

                    if pred_rotmat_prev_np is not None and pred_rotmat_curr_np is not None:
                        # Ensure shapes are compatible for subtraction (e.g. (23,3,3))
                        if pred_rotmat_curr_np.shape == pred_rotmat_prev_np.shape:
                            diff_rotmat = pred_rotmat_curr_np - pred_rotmat_prev_np
                            individual_fro_norms = np.linalg.norm(diff_rotmat, ord='fro', axis=(1,2))
                            instant_pose_change = np.sum(individual_fro_norms)
                        else:
                            print(f"METRICS_THREAD: Mismatched rotmat shapes. Curr: {pred_rotmat_curr_np.shape}, Prev: {pred_rotmat_prev_np.shape}")


                    if pred_cam_t_prev_np is not None and pred_cam_t_curr_np is not None:
                        # Assuming inputs are in meters, convert to mm for change
                        instant_trans_change = np.linalg.norm((pred_cam_t_curr_np * 1000) - (pred_cam_t_prev_np * 1000))

                    if pred_vertices_prev_np is not None and pred_vertices_curr_np is not None:
                        # Assuming inputs are in meters, convert to mm for change
                        vertex_diff_mm = np.linalg.norm((pred_vertices_curr_np*1000) - (pred_vertices_prev_np*1000), axis=1)
                        instant_joint_change = np.mean(vertex_diff_mm)
            
            # Append metrics (even if NaN, for consistent timestamping)
            pose_change_buffer.append((current_ts, instant_pose_change))
            translation_change_buffer.append((current_ts, instant_trans_change))
            joint_pos_change_buffer.append((current_ts, instant_joint_change))
            shape_var_buffer.append((current_ts, instant_shape_var))
            detection_rate_buffer.append((current_ts, instant_detection_rate))

            metric_data_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"METRICS_THREAD: Error in metric_calculation_worker: {e}")
            traceback.print_exc()
    print("METRICS_THREAD: Metric calculation thread finished.")

# --- GUI Functions (from metrics script) ---
def setup_gui():
    global gui_root, condition_var, app_running
    gui_root = tk.Tk()
    gui_root.title("POCO Metrics Recorder")
    gui_root.geometry("350x200")

    def on_gui_close():
        global app_running
        print("GUI: Closed by user.")
        # app_running = False # Main loop controls app_running, GUI just closes itself
        if gui_root:
            gui_root.destroy()

    gui_root.protocol("WM_DELETE_WINDOW", on_gui_close)
    tk.Label(gui_root, text="Condition:").pack(pady=(10,0))
    conditions = ["Optimal", "Low Light", "Partial Occlusion", "Fast Motion", "Static+Moving Occlusion", "Custom"]
    condition_var = tk.StringVar(gui_root)
    condition_var.set(conditions[0]) # Default value
    ttk.OptionMenu(gui_root, condition_var, conditions[0], *conditions).pack(pady=5, padx=10, fill='x')
    tk.Button(gui_root, text="Record Avg Metrics (5s)", command=record_metrics_action, height=2).pack(pady=10, padx=10, fill='x')
    status_label_var = tk.StringVar()
    status_label_var.set("Press 'Record' or 'r' key in OpenCV window.")
    tk.Label(gui_root, textvariable=status_label_var, wraplength=330).pack(pady=5)
    gui_root.status_label_var = status_label_var

    try:
        print("GUI: Starting Tkinter mainloop...")
        gui_root.mainloop()
    except Exception as e:
        print(f"GUI: Error in GUI mainloop: {e}")
    finally:
        print("GUI: Tkinter mainloop exited.")
        # If GUI closes, it doesn't necessarily stop the whole app,
        # main app_running flag controls that.

def record_metrics_action():
    global CSV_FILENAME, CSV_FIELDNAMES, condition_var, gui_root, METRIC_BUFFER_DURATION_SECONDS
    global fps_buffer, latency_buffer, pose_change_buffer, translation_change_buffer
    global joint_pos_change_buffer, shape_var_buffer, detection_rate_buffer

    if not app_running and gui_root and hasattr(gui_root, 'status_label_var'): # Check if app is meant to be running
        gui_root.status_label_var.set("Application is not running or shutting down.")
        print("METRICS_REC: Record action called but app not running.")
        return

    recording_timestamp = time.time()
    current_condition = "N/A"
    if condition_var: # condition_var might be None if GUI hasn't fully initialized or closed prematurely
        current_condition = condition_var.get()
    
    averaged_metrics_log = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recording_timestamp)),
        "Condition": current_condition
    }

    def get_average_from_metric_buffer(buffer_deque, nan_treatment_policy='omit'):
        buffer_snapshot = list(buffer_deque)
        values_in_window = [val for ts, val in buffer_snapshot if ts >= recording_timestamp - METRIC_BUFFER_DURATION_SECONDS]
        
        if not values_in_window: return np.nan

        if nan_treatment_policy == 'omit':
            valid_numeric_values = [v for v in values_in_window if not (isinstance(v, float) and np.isnan(v))]
            if not valid_numeric_values: return np.nan
            return np.mean(valid_numeric_values)
        elif nan_treatment_policy == 'include_zeros': # For detection rate
            return np.mean(values_in_window)
        return np.nan

    averaged_metrics_log["Avg Processing FPS"] = get_average_from_metric_buffer(fps_buffer)
    averaged_metrics_log["Avg End-to-End Latency (ms)"] = get_average_from_metric_buffer(latency_buffer)
    averaged_metrics_log["Avg Pose Change (Euclidean Dist)"] = get_average_from_metric_buffer(pose_change_buffer)
    averaged_metrics_log["Avg Translation Change (mm)"] = get_average_from_metric_buffer(translation_change_buffer)
    averaged_metrics_log["Avg Joint Position Change (mm)"] = get_average_from_metric_buffer(joint_pos_change_buffer)
    averaged_metrics_log["Avg Shape Param Variance"] = get_average_from_metric_buffer(shape_var_buffer)
    averaged_metrics_log["Avg Detection/Tracking Rate (%)"] = get_average_from_metric_buffer(detection_rate_buffer, nan_treatment_policy='include_zeros')
    
    csv_file_exists = os.path.isfile(CSV_FILENAME)
    try:
        with open(CSV_FILENAME, 'a', newline='') as csv_output_file:
            csv_writer = csv.DictWriter(csv_output_file, fieldnames=CSV_FIELDNAMES)
            if not csv_file_exists:
                csv_writer.writeheader()
            
            formatted_row_to_write = {}
            for csv_col_header, avg_val in averaged_metrics_log.items():
                if isinstance(avg_val, float):
                    formatted_row_to_write[csv_col_header] = f"{avg_val:.4f}" if not np.isnan(avg_val) else "NaN"
                else:
                    formatted_row_to_write[csv_col_header] = avg_val
            csv_writer.writerow(formatted_row_to_write)
            
        success_message = f"Avg Metrics for '{averaged_metrics_log['Condition']}' saved."
        print(f"METRICS_REC: {success_message}")
        if gui_root and hasattr(gui_root, 'status_label_var'):
            gui_root.status_label_var.set(success_message)
    except IOError as e_io:
        error_message_csv = f"METRICS_REC: CSV I/O Error: {e_io}"
        print(error_message_csv); traceback.print_exc()
        if gui_root and hasattr(gui_root, 'status_label_var'):
            gui_root.status_label_var.set(error_message_csv)
    except Exception as e_gen:
        error_message_csv = f"METRICS_REC: Unexpected CSV Error: {e_gen}"
        print(error_message_csv); traceback.print_exc()
        if gui_root and hasattr(gui_root, 'status_label_var'):
            gui_root.status_label_var.set(error_message_csv)


def main(args):
    global app_running, metric_thread_instance, gui_thread_instance # For managing threads
    global main_thread_prev_model_outputs # For metrics
    global fps_buffer, latency_buffer # Populated by main thread

    # Determine device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"DEMO: Using device: {device}")

    # --- Initialize and Start Metrics and GUI Threads ---
    app_running = True # Set app_running to True at the start of main

    # Initialize main_thread_prev_model_outputs (important for first comparison)
    main_thread_prev_model_outputs = {
        "pred_rotmat_body": None, "pred_cam_t": None, "pred_vertices": None,
        "valid_for_comparison": False
    }
    
    if args.mode == 'webcam': # Metrics and GUI are primarily for webcam mode
        print("DEMO: Starting metrics calculation thread...")
        metric_thread_instance = threading.Thread(target=metrics_calculation_worker, daemon=True)
        metric_thread_instance.start()
        
        print("DEMO: Starting GUI thread...")
        gui_thread_instance = threading.Thread(target=setup_gui, daemon=True)
        gui_thread_instance.start()

        if args.display: # Only start GUI if display is enabled
            print("DEMO: Starting GUI thread...")
            # gui_thread_instance = threading.Thread(target=setup_gui, daemon=True)
            # gui_thread_instance.start()
        else:
            print("DEMO: Display is off, GUI for metrics recording will not be started. CSV saving will not be available via GUI.")
    # --- End Metrics and GUI Thread Initialization ---

    # Initialize socket client (only in webcam mode for this example)
    sock = None
    if args.mode == 'webcam':
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0) # Timeout for connection attempt
            sock.connect((RELAY_HOST, RELAY_PORT))
            sock.settimeout(None) # Reset timeout for sending data
            print(f"DEMO: Connected to relay server at {RELAY_HOST}:{RELAY_PORT}")
        except ConnectionRefusedError:
            print(f"DEMO WARNING: Connection refused. Is remote_relay.py running and listening on port {RELAY_PORT}?")
            sock = None
        except socket.timeout:
            print(f"DEMO WARNING: Connection timed out. Is remote_relay.py running and listening on port {RELAY_PORT}?")
            sock = None
        except Exception as e:
            print(f"DEMO ERROR: Error connecting to relay server: {e}")
            traceback.print_exc()
            sock = None

    # --- Model Loading ---
    model = None
    tester = None
    model_cfg = None
    using_trt_model = False

    if args.trt_model and os.path.exists(args.trt_model):
        print(f"DEMO: Attempting to load TensorRT model from: {args.trt_model}")
        try:
            model = torch.jit.load(args.trt_model).to(device).eval()
            print("DEMO: TensorRT model loaded successfully.")
            using_trt_model = True
            print("DEMO: Initializing POCOTester for configuration...")
            try:
                tester = POCOTester(args)
                model_cfg = tester.model_cfg
                print("DEMO: POCOTester initialized for config.")
            except Exception as e:
                print(f"DEMO ERROR: Failed to initialize POCOTester for config in TRT mode: {e}")
                traceback.print_exc()
                if sock: sock.close()
                app_running = False; # Signal threads to stop
                sys.exit(1)
        except Exception as e:
            print(f"DEMO ERROR: Failed to load TensorRT model: {e}")
            print("DEMO INFO: Ensure 'import torch_tensorrt' is present and successful.")
            traceback.print_exc()
            if sock: sock.close()
            app_running = False; # Signal threads to stop
            sys.exit(1)
    else:
        if args.trt_model:
            print(f"DEMO WARNING: TensorRT model path provided but not found: {args.trt_model}. Falling back to standard model.")
        print("DEMO: Initializing POCOTester to load standard PyTorch model...")
        try:
            tester = POCOTester(args)
            model = tester.model
            model_cfg = tester.model_cfg
            model.eval()
            print("DEMO: POCO Tester Initialized with standard model.")
        except Exception as e:
            print(f"DEMO ERROR: Failed to initialize POCOTester with standard model: {e}")
            traceback.print_exc()
            if sock: sock.close()
            app_running = False; # Signal threads to stop
            sys.exit(1)

    if model is None or model_cfg is None:
        print("DEMO ERROR: Model or configuration failed to load.")
        if sock: sock.close()
        app_running = False; # Signal threads to stop
        sys.exit(1)

    demo_mode = args.mode
    stream_mode = args.stream
    frame_count = 0
    start_time_proc = time.time()
    cap = None

    if demo_mode == 'video':
        # ... (original video logic - metrics not integrated here for brevity but could be) ...
        print("DEMO: Running in VIDEO mode. Metrics integration focused on webcam mode.")
        video_file = args.vid_file
        if not isfile(video_file): sys.exit(f'Input video \"{video_file}\" does not exist!')
        # ... rest of video setup
        # For video mode, you would need to adapt the main loop similarly to webcam mode
        # if metrics are desired.
        app_running = False # Assuming video mode runs once and finishes

    elif demo_mode == 'folder':
        # ... (original folder logic - metrics not integrated here) ...
        print("DEMO: Running in FOLDER mode. Metrics integration focused on webcam mode.")
        app_running = False

    elif demo_mode == 'directory':
        # ... (original directory logic - metrics not integrated here) ...
        print("DEMO: Running in DIRECTORY mode. Metrics integration focused on webcam mode.")
        app_running = False

    elif demo_mode == 'webcam':
        if sock is None and not args.display:
            print("DEMO WARNING: Socket not connected. Cannot send data. Running display only.")
        elif sock is None and args.display:
            print("DEMO WARNING: Socket not connected. Cannot send data. Displaying locally.")

        print(f'DEMO: Webcam Demo options: \n {args}')
        print("DEMO: Initializing Multi-Person Tracker...")
        try:
            mot = MPT(
                device=device, batch_size=args.tracker_batch_size, display=args.display, # tracker_batch_size is forced to 1 for webcam later
                detector_type=args.detector, output_format='dict',
                yolo_img_size=args.yolo_img_size,
            )
            print("DEMO: Tracker Initialized.")
        except Exception as e:
            print(f"DEMO ERROR: Failed to initialize Multi-Person Tracker: {e}")
            if sock: sock.close()
            app_running = False; return

        print("DEMO: Opening video source...")
        if (stream_mode):
            rtmp_url = "rtmp://35.246.39.155:1935/live/webcam"
            print(f"DEMO: Attempting to connect to RTMP stream: {rtmp_url}")
            cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
        else:
            webcam_idx = 0
            print(f"DEMO: Attempting to open webcam (index {webcam_idx})...")
            cap = cv2.VideoCapture(webcam_idx)

        if not cap or not cap.isOpened():
            print("DEMO ERROR: Cannot open video source (webcam/stream)")
            if sock: sock.close()
            app_running = False; # Ensure threads know to stop
            # No return here, finally block will handle cleanup
        else:
            print("DEMO: Video source opened successfully.")
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"DEMO: Video source resolution: {frame_width}x{frame_height}, FPS: {video_fps if video_fps > 0 else 'N/A'}")

        print("DEMO: Starting webcam stream. Press 'q' in OpenCV window (if displayed) or Ctrl+C in terminal to exit.")
        if args.display: print("DEMO: Press 'r' in OpenCV window to record 5-sec average metrics to CSV.")
        if sock: print("DEMO: Sending SMPL data to relay server...")

        frame_count = 0
        start_time_proc = time.time() # Overall processing start time for FPS calculation
        last_log_time = time.time()
        vertical_offset = 0.8
        depth_z = 0.0

        try:
            if not cap or not cap.isOpened():
                print("DEMO: Cannot start loop, video source not opened.")
                app_running = False # Signal threads
            else:
                while app_running: # Main processing loop, controlled by app_running
                    frame_start_loop = time.time() # For current frame's latency and FPS
                    current_frame_timestamp_for_buffer = time.time() # Timestamp for metrics buffering
                    
                    # --- Initialize for metrics packet ---
                    instant_detection_rate_main_thread = 0.0
                    current_model_outputs_for_metric_thread = None
                    # ---

                    try:
                        ret, frame = cap.read()
                        if not ret:
                            print("DEMO INFO: End of stream or failed to grab frame.")
                            app_running = False; break 
                        if frame is None:
                            print("DEMO WARNING: Grabbed frame is None. Skipping frame.")
                            time.sleep(0.01); continue # Avoid busy loop on None frame
                    except Exception as e:
                        print(f"DEMO ERROR: Exception while reading frame: {e}")
                        app_running = False; break
                    
                    frame_count += 1
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        orig_h, orig_w = rgb_frame.shape[:2]
                    except Exception as e:
                        print(f"DEMO ERROR: Failed to convert frame {frame_count} to RGB: {e}")
                        main_thread_prev_model_outputs["valid_for_comparison"] = False # Mark prev data invalid
                        continue

                    dets_prepared_list = []
                    dets = np.array([])
                    try:
                        dets_raw = mot.detect_frame(rgb_frame)
                        if dets_raw is not None and dets_raw.shape[0] > 0:
                            for d_idx, d_val in enumerate(dets_raw): # Use enumerate for safer access
                                if len(d_val) >= 4:
                                    x1, y1, x2, y2 = d_val[:4]
                                    w_det, h_det = x2 - x1, y2 - y1
                                    if w_det > 0 and h_det > 0:
                                        c_x, c_y = x1 + w_det / 2, y1 + h_det / 2
                                        size = max(w_det, h_det) * 1.2
                                        bbox_prepared = np.array([c_x, c_y, size, size])
                                        # Include original detection index if needed for tracking later
                                        # For now, just the bbox for POCO input
                                        dets_prepared_list.append(bbox_prepared)
                        if dets_prepared_list:
                            dets = np.array(dets_prepared_list)
                        else:
                            dets = np.array([])
                    except Exception as e:
                        print(f"DEMO ERROR: Exception during object detection on frame {frame_count}: {e}")
                        main_thread_prev_model_outputs["valid_for_comparison"] = False
                        continue
                    
                    # --- Handle No Detections ---
                    if dets.shape[0] == 0:
                        instant_detection_rate_main_thread = 0.0 # No detection
                        main_thread_prev_model_outputs["valid_for_comparison"] = False # Prev data invalid for next frame
                        if sock: # Send T-Pose
                            # ... (original T-pose sending logic)
                            poses_body_send = np.zeros((1, POSE_BODY_DIM), dtype=np.float32)
                            poses_root_send = np.zeros((1, 3), dtype=np.float32)
                            betas_send = np.zeros((1, NUM_BETAS), dtype=np.float32)
                            trans_send = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                            data_to_send = {
                                "poses_body": poses_body_send, "poses_root": poses_root_send,
                                "betas": betas_send, "trans": trans_send,
                            }
                            if not send_data(sock, data_to_send):
                                print("DEMO ERROR: Failed to send T-pose data (no detections). Exiting loop.")
                                app_running = False; break
                        # Update display if enabled (original logic)
                        if args.display:
                            display_frame_no_det = frame.copy()
                            fps_loop_no_det = 1.0 / (time.time() - frame_start_loop) if (time.time() - frame_start_loop) > 0 else 0
                            cv2.putText(display_frame_no_det, f"FPS: {fps_loop_no_det:.2f} (No Detections)", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imshow("Webcam Demo - POCO", display_frame_no_det)
                            key_press = cv2.waitKey(1) & 0xFF
                            if key_press == ord('q'): app_running = False; break
                            if key_press == ord('r') and gui_thread_instance: record_metrics_action()
                        # Fall through to metrics packet sending and FPS/latency buffering for this frame
                    else: # Detections found
                        instant_detection_rate_main_thread = 100.0 # Detection successful

                        inp_images, bbox_info_list, focal_lengths_list, scales_list, centers_list, orig_shapes_list = [], [], [], [], [], []
                        try:
                            # Process only the first detection for single-person metrics and socket sending in this example
                            # The original script processes all detections for the batch. We'll adapt.
                            # For metrics, we'll focus on the first person.
                            # For model input, we prepare for all detected persons.
                            
                            num_persons_to_process = dets.shape[0] # Process all detected persons for model input

                            for i in range(num_persons_to_process):
                                det_person_i = dets[i] # Current person's detection
                                if len(det_person_i) == 4:
                                    norm_img, _, _ = get_single_image_crop_demo(
                                        rgb_frame, det_person_i, kp_2d=None, scale=1.0,
                                        crop_size=model_cfg.DATASET.IMG_RES
                                    )
                                    center = [det_person_i[0], det_person_i[1]]
                                    scale_val = det_person_i[2] / 200.0
                                    inp_images.append(norm_img.float())
                                    orig_shape = [orig_h, orig_w]
                                    centers_list.append(center)
                                    orig_shapes_list.append(orig_shape)
                                    scales_list.append(scale_val)
                                    bbox_info = calculate_bbox_info(center, scale_val, orig_shape)
                                    bbox_info_list.append(bbox_info)
                                    focal_length_val = calculate_focal_length(orig_h, orig_w) # This is scalar, repeated per person
                                    focal_lengths_list.append(focal_length_val)
                            
                            if not inp_images:
                                print(f"DEMO WARNING: No valid detections to process after batch prep on frame {frame_count}.")
                                main_thread_prev_model_outputs["valid_for_comparison"] = False
                                continue

                            batch = {
                                'img': torch.stack(inp_images).to(device),
                                'bbox_info': torch.FloatTensor(bbox_info_list).to(device),
                                'focal_length': torch.FloatTensor(focal_lengths_list).to(device), # Should be (N, 2) or (N, 1) depending on use
                                'scale': torch.FloatTensor(scales_list).to(device),
                                'center': torch.FloatTensor(centers_list).to(device),
                                'orig_shape': torch.FloatTensor(orig_shapes_list).to(device),
                            }
                        except Exception as e:
                            print(f"DEMO ERROR: Exception during batch preparation on frame {frame_count}: {e}")
                            main_thread_prev_model_outputs["valid_for_comparison"] = False
                            continue
                        
                        output = None
                        try:
                            model.eval()
                            with torch.no_grad():
                                if using_trt_model:
                                    try:
                                        num_dets_in_batch = batch['img'].shape[0]
                                        trt_input_tuple = tuple(batch[key][:num_dets_in_batch] for key in TRT_INPUT_KEYS)
                                    except KeyError as e:
                                        print(f"DEMO ERROR: Missing key '{e}' for TRT input on frame {frame_count}.")
                                        main_thread_prev_model_outputs["valid_for_comparison"] = False; continue
                                    except Exception as e_prep:
                                        print(f"DEMO ERROR: Failed preparing TRT input tuple on frame {frame_count}: {e_prep}")
                                        main_thread_prev_model_outputs["valid_for_comparison"] = False; continue
                                    output_tuple = model(*trt_input_tuple)
                                    if len(output_tuple) == len(TRT_OUTPUT_KEYS):
                                        output = {key: tensor for key, tensor in zip(TRT_OUTPUT_KEYS, output_tuple)}
                                    else:
                                        print(f"DEMO ERROR: TRT model output tuple length mismatch on frame {frame_count}.")
                                        output = {}
                                else:
                                    output = model(batch)

                            if output is None or not output:
                                print(f"DEMO WARNING: Model inference returned None or empty on frame {frame_count}.")
                                main_thread_prev_model_outputs["valid_for_comparison"] = False
                                continue
                        except Exception as e:
                            print(f"DEMO ERROR: Exception during model inference on frame {frame_count}: {e}")
                            traceback.print_exc()
                            main_thread_prev_model_outputs["valid_for_comparison"] = False
                            continue

                        # --- DATA EXTRACTION FOR SOCKET (Original Logic - First Person) ---
                        # This part remains for sending data via socket as per original script.
                        # We will also extract data for metrics from the *first person's* output.
                        pred_pose_raw_socket = output.get('pred_pose') # This is (B, 24, 3,3) or (B, 72)
                        pred_shape_socket = output.get('pred_shape')   # (B, 10)
                        pred_cam_socket = output.get('pred_cam')       # (B, 3)
                        
                        # Default T-Pose for socket if extraction fails
                        poses_body_send_socket = np.zeros((1, POSE_BODY_DIM), dtype=np.float32)
                        poses_root_send_socket = np.zeros((1, 3), dtype=np.float32)
                        betas_send_socket = np.zeros((1, NUM_BETAS), dtype=np.float32)
                        trans_send_socket = np.array([[0.0, 0.8, 0.0]], dtype=np.float32)

                        if pred_pose_raw_socket is not None and pred_shape_socket is not None and pred_cam_socket is not None:
                            if pred_pose_raw_socket.shape[0] > 0 and pred_shape_socket.shape[0] > 0 and pred_cam_socket.shape[0] > 0:
                                # Use first person for socket sending
                                pred_pose_person0_socket_tensor = pred_pose_raw_socket[0] # (24,3,3) or (72,)
                                pred_shape_person0_socket_tensor = pred_shape_socket[0] # (10,)
                                pred_cam_person0_socket_tensor = pred_cam_socket[0]   # (3,)

                                # Convert pose to axis-angle (72-dim) if it's not already
                                if pred_pose_person0_socket_tensor.ndim == 3 and pred_pose_person0_socket_tensor.shape == (24,3,3): # RotMat
                                    pred_pose_aa_person0_socket_flat = rotation_matrix_to_angle_axis(pred_pose_person0_socket_tensor.reshape(-1,3,3)) # (24,3)
                                    pred_pose_aa_person0_socket = pred_pose_aa_person0_socket_flat.reshape(72) # (72,)
                                elif pred_pose_person0_socket_tensor.ndim == 1 and pred_pose_person0_socket_tensor.shape == (72,): # Already (72,) axis-angle
                                    pred_pose_aa_person0_socket = pred_pose_person0_socket_tensor
                                elif pred_pose_person0_socket_tensor.ndim == 2 and pred_pose_person0_socket_tensor.shape == (24,3): # (24,3) axis-angle
                                     pred_pose_aa_person0_socket = pred_pose_person0_socket_tensor.reshape(72)
                                else:
                                    print(f"DEMO WARNING: Unrecognized pose format for socket data on frame {frame_count}. Sending T-Pose.")
                                    pred_pose_aa_person0_socket = None # Fallback

                                if pred_pose_aa_person0_socket is not None:
                                    try:
                                        # Apply 180-degree X-axis rotation fix (original logic)
                                        global_orient_aa_socket = pred_pose_aa_person0_socket[:3]
                                        body_pose_aa_socket_np = pred_pose_aa_person0_socket[3:].cpu().numpy() # For POSE_BODY_DIM

                                        global_orient_rotmat_socket = batch_rodrigues(global_orient_aa_socket.unsqueeze(0))
                                        rot_180_x_socket = torch.tensor([[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]],
                                                                dtype=global_orient_rotmat_socket.dtype, device=global_orient_rotmat_socket.device)
                                        rotated_global_orient_rotmat_socket = torch.bmm(rot_180_x_socket, global_orient_rotmat_socket)
                                        rotated_global_orient_aa_socket_np = rotation_matrix_to_angle_axis(rotated_global_orient_rotmat_socket).squeeze(0).cpu().numpy()

                                        shape_person0_socket_np = pred_shape_person0_socket_tensor.cpu().numpy()
                                        cam_person0_socket_np = pred_cam_person0_socket_tensor.cpu().numpy()

                                        if rotated_global_orient_aa_socket_np.shape == (3,) and \
                                           body_pose_aa_socket_np.shape == (POSE_BODY_DIM,) and \
                                           shape_person0_socket_np.shape[0] >= NUM_BETAS:
                                            
                                            tx_socket = cam_person0_socket_np[1]
                                            ty_socket = cam_person0_socket_np[2]
                                            trans_extracted_socket = np.array([tx_socket, ty_socket + vertical_offset, depth_z], dtype=np.float32)

                                            poses_body_send_socket = body_pose_aa_socket_np.reshape(1, POSE_BODY_DIM)
                                            poses_root_send_socket = rotated_global_orient_aa_socket_np.reshape(1, 3)
                                            betas_send_socket = shape_person0_socket_np[:NUM_BETAS].reshape(1, NUM_BETAS)
                                            trans_send_socket = trans_extracted_socket.reshape(1, 3)
                                        else:
                                            print(f"DEMO WARNING: Extracted socket pose/shape dimensions incorrect frame {frame_count}.")
                                    except Exception as e_sock_ext:
                                        print(f"DEMO ERROR: Exception during socket data extraction/fix frame {frame_count}: {e_sock_ext}.")
                        
                        data_to_send_socket = {
                            "poses_body": poses_body_send_socket, "poses_root": poses_root_send_socket,
                            "betas": betas_send_socket, "trans": trans_send_socket,
                        }
                        if sock:
                            if not send_data(sock, data_to_send_socket):
                                print("DEMO ERROR: Failed to send data to relay server. Exiting loop.")
                                app_running = False; break
                        
                        # --- DATA EXTRACTION FOR METRICS (First Person) ---
                        # We need: pred_rotmat_body (23,3,3), pred_betas (10,), pred_vertices (6890,3), pred_cam_t (3,)
                        # All as NumPy arrays.
                        
                        # 1. pred_betas (already have pred_shape_person0_socket_tensor)
                        current_pred_betas_np_metric = pred_shape_socket[0, :NUM_BETAS].cpu().numpy() # (10,)

                        # 2. pred_vertices (from output['smpl_vertices'])
                        smpl_vertices_output = output.get('smpl_vertices') # (B, 6890, 3)
                        if smpl_vertices_output is not None and smpl_vertices_output.shape[0] > 0:
                            current_pred_vertices_np_metric = smpl_vertices_output[0].cpu().numpy() # (6890, 3)
                        else:
                            current_pred_vertices_np_metric = None # Or a placeholder

                        # 3. pred_rotmat_body (from output['pred_pose'] for the first person)
                        # output['pred_pose'] is (B, 24, 3,3) or (B, 72)
                        pred_pose_metric_tensor = output.get('pred_pose')[0] # (24,3,3) or (72,) for person 0
                        
                        if pred_pose_metric_tensor.ndim == 3 and pred_pose_metric_tensor.shape == (24,3,3): # RotMat
                            body_pose_rotmats_metric_torch = pred_pose_metric_tensor[1:] # (23,3,3)
                        elif pred_pose_metric_tensor.ndim == 1 and pred_pose_metric_tensor.shape[0] == 72: # AxisAngle (72,)
                            body_pose_aa_metric_torch = pred_pose_metric_tensor[3:].reshape(23,3)
                            body_pose_rotmats_metric_torch = batch_rodrigues(body_pose_aa_metric_torch.unsqueeze(0)).squeeze(0) # (23,3,3)
                        elif pred_pose_metric_tensor.ndim == 2 and pred_pose_metric_tensor.shape == (24,3): # AxisAngle (24,3)
                            body_pose_aa_metric_torch = pred_pose_metric_tensor[1:] # (23,3)
                            body_pose_rotmats_metric_torch = batch_rodrigues(body_pose_aa_metric_torch.unsqueeze(0)).squeeze(0) # (23,3,3)
                        else:
                            body_pose_rotmats_metric_torch = None
                            print(f"DEMO WARNING: Unrecognized pred_pose format for metrics on frame {frame_count}.")

                        current_pred_rotmat_body_np_metric = body_pose_rotmats_metric_torch.cpu().numpy() if body_pose_rotmats_metric_torch is not None else None
                        
                        # 4. pred_cam_t (world translation for the first person)
                        # Use convert_crop_cam_to_orig_img as planned
                        pred_cam_metric_person0_tensor = output.get('pred_cam')[0:1] # (1,3) s,tx,ty
                        # Ensure 'dets' corresponds to the persons in 'output'. Assuming dets[0] is the first person.
                        det_metric_person0 = dets[0:1] # (1,4) bbox for the first person
                        
                        if pred_cam_metric_person0_tensor is not None and det_metric_person0 is not None and \
                           pred_cam_metric_person0_tensor.shape[0] > 0 and det_metric_person0.shape[0] > 0:
                            orig_cam_for_metric = convert_crop_cam_to_orig_img(
                                cam=pred_cam_metric_person0_tensor.cpu().numpy(), # expects numpy
                                bbox=det_metric_person0, # expects numpy
                                img_width=orig_w, img_height=orig_h
                            )
                            current_pred_cam_t_np_metric = orig_cam_for_metric[0] # (3,)
                        else:
                            current_pred_cam_t_np_metric = None
                            print(f"DEMO WARNING: Could not get camera translation for metrics on frame {frame_count}.")

                        # --- Package for metrics thread ---
                        if all(v is not None for v in [current_pred_rotmat_body_np_metric, current_pred_betas_np_metric, 
                                                       current_pred_vertices_np_metric, current_pred_cam_t_np_metric]):
                            current_model_outputs_for_metric_thread = {
                                "pred_rotmat_body": current_pred_rotmat_body_np_metric,
                                "pred_betas": current_pred_betas_np_metric,
                                "pred_vertices": current_pred_vertices_np_metric,
                                "pred_cam_t": current_pred_cam_t_np_metric
                            }
                        else: # If any component is None, don't send partial data
                            current_model_outputs_for_metric_thread = None
                            main_thread_prev_model_outputs["valid_for_comparison"] = False
                            print(f"DEMO WARNING: One or more components for metrics data was None on frame {frame_count}.")


                        # --- Optional Display (Original Logic) ---
                        if args.display:
                            try:
                                render_frame_display = frame.copy()
                                smpl_vertices_render = output.get('smpl_vertices') # (B, 6890, 3)
                                pred_cam_render = output.get('pred_cam')           # (B, 3)

                                if smpl_vertices_render is not None and pred_cam_render is not None and \
                                   smpl_vertices_render.shape[0] > 0 and pred_cam_render.shape[0] > 0:
                                    
                                    num_to_render = min(dets.shape[0], smpl_vertices_render.shape[0], pred_cam_render.shape[0])
                                    if num_to_render > 0:
                                        valid_pred_cam_render = pred_cam_render[:num_to_render].cpu().numpy()
                                        valid_dets_render = dets[:num_to_render]

                                        orig_cam_render = convert_crop_cam_to_orig_img(
                                            cam=valid_pred_cam_render,
                                            bbox=valid_dets_render,
                                            img_width=orig_w, img_height=orig_h
                                        )
                                        # Ensure renderer is initialized (it should be if args.display is true and model_cfg is loaded)
                                        # The original script initializes renderer inside the loop, which is inefficient.
                                        # We'll assume it's initialized once if display is on.
                                        # For this integration, let's re-initialize per frame to match original structure closely,
                                        # but ideally, it should be outside.
                                        # For now, let's assume renderer is available if args.display
                                        local_renderer = Renderer(resolution=(orig_w, orig_h), orig_img=True, wireframe=args.wireframe)
                                        render_frame_display_rgb = cv2.cvtColor(render_frame_display, cv2.COLOR_BGR2RGB)

                                        for i in range(num_to_render):
                                            verts_render = smpl_vertices_render[i].cpu().numpy()
                                            if np.isnan(verts_render).any() or np.isinf(verts_render).any():
                                                print(f"DEMO WARNING: NaN/Inf in vertices for person {i} on frame {frame_count}, skipping render.")
                                                continue
                                            render_frame_display_rgb = local_renderer.render(
                                                render_frame_display_rgb, verts_render, cam=orig_cam_render[i], color=[0.7, 0.7, 0.7]
                                            )
                                        render_frame_display = cv2.cvtColor(render_frame_display_rgb, cv2.COLOR_RGB2BGR)

                                fps_loop_display = 1.0 / (time.time() - frame_start_loop) if (time.time() - frame_start_loop) > 0 else 0
                                cv2.putText(render_frame_display, f"FPS: {fps_loop_display:.2f}", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(render_frame_display, f"Latency: {(time.time() - frame_start_loop)*1000:.1f}ms", (10, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Display instantaneous latency
                                cv2.imshow("Webcam Demo - POCO", render_frame_display)

                                key_press = cv2.waitKey(1) & 0xFF
                                if key_press == ord('q'): app_running = False; break
                                if key_press == ord('r') and gui_thread_instance: record_metrics_action() # Trigger CSV save
                            except Exception as e_disp:
                                print(f"DEMO ERROR: Exception during display update on frame {frame_count}: {e_disp}")
                                traceback.print_exc()
                        else: # No display
                            time.sleep(0.001) # Prevent busy-waiting

                    # --- Send data to metrics thread ---
                    data_packet_for_metric_thread = {
                        "timestamp": current_frame_timestamp_for_buffer,
                        "current_outputs": current_model_outputs_for_metric_thread, # This is None if current frame processing failed or no dets
                        "prev_outputs_for_comparison": main_thread_prev_model_outputs.copy(),
                        "detection_rate": instant_detection_rate_main_thread
                    }
                    try:
                        metric_data_queue.put_nowait(data_packet_for_metric_thread)
                    except queue.Full:
                        print("DEMO WARNING: Metric data queue full. Dropping metrics for this frame.")

                    # --- Update prev_model_outputs for next iteration ---
                    if current_model_outputs_for_metric_thread:
                        main_thread_prev_model_outputs["pred_rotmat_body"] = current_model_outputs_for_metric_thread["pred_rotmat_body"]
                        main_thread_prev_model_outputs["pred_cam_t"] = current_model_outputs_for_metric_thread["pred_cam_t"]
                        main_thread_prev_model_outputs["pred_vertices"] = current_model_outputs_for_metric_thread["pred_vertices"]
                        # Betas are not used for frame-to-frame comparison in the metrics worker, but could be stored if needed
                        main_thread_prev_model_outputs["valid_for_comparison"] = True
                    else: # Current frame failed or no detections
                        main_thread_prev_model_outputs["valid_for_comparison"] = False
                    
                    # --- Main thread's FPS & Latency buffering ---
                    loop_end_time_main_thread = time.time()
                    processing_time_main_thread = loop_end_time_main_thread - frame_start_loop # Latency for this frame
                    instant_fps_main_thread = 1.0 / processing_time_main_thread if processing_time_main_thread > 1e-6 else 0.0
                    
                    fps_buffer.append((current_frame_timestamp_for_buffer, instant_fps_main_thread))
                    latency_buffer.append((current_frame_timestamp_for_buffer, processing_time_main_thread * 1000))

                    # Log performance periodically
                    current_time_log = time.time()
                    if current_time_log - last_log_time >= 10.0:
                        # Calculate overall average FPS so far
                        avg_fps_so_far = frame_count / (current_time_log - start_time_proc) if (current_time_log - start_time_proc) > 0 else 0
                        print(f"DEMO DEBUG: Frame {frame_count}, Current Avg FPS (Overall): {avg_fps_so_far:.2f}")
                        last_log_time = current_time_log
                    
                    # Check if OpenCV window was closed by user (if display is on)
                    if args.display:
                        try:
                            if cv2.getWindowProperty("Webcam Demo - POCO", cv2.WND_PROP_VISIBLE) < 1:
                                print("DEMO: OpenCV window closed by user.")
                                app_running = False # Signal to stop
                        except cv2.error: # Window might have been destroyed
                            print("DEMO: OpenCV window seems to have been destroyed.")
                            app_running = False # Signal to stop


        except KeyboardInterrupt:
            print("\nDEMO: Ctrl+C detected. Exiting gracefully...")
            app_running = False # Signal threads
        except Exception as e:
            print(f"\nDEMO ERROR: An unexpected error occurred in the main loop: {e}")
            traceback.print_exc()
            app_running = False # Signal threads
        finally:
            print("\nDEMO: Cleaning up resources...")
            app_running = False # Ensure all threads know to stop

            end_time_proc = time.time()

            if cap is not None and cap.isOpened():
                print("DEMO: Releasing video capture device.")
                cap.release()
            if args.display:
                print("DEMO: Closing OpenCV windows.")
                cv2.destroyAllWindows()
            if sock:
                print("DEMO: Closing socket connection.")
                sock.close()

            # --- Signal and wait for Metrics and GUI Threads ---
            if metric_thread_instance and metric_thread_instance.is_alive():
                print("DEMO: Signaling metric calculation thread to stop...")
                try: metric_data_queue.put_nowait(None) # Sentinel
                except queue.Full: print("DEMO WARNING: Metric queue full during shutdown signal.")
                metric_thread_instance.join(timeout=2.0)
                if metric_thread_instance.is_alive(): print("DEMO WARNING: Metric thread did not join gracefully.")
            
            if gui_thread_instance and gui_thread_instance.is_alive():
                print("DEMO: Closing GUI thread...")
                if gui_root:
                    try: gui_root.destroy()
                    except Exception: pass # Ignore errors if already destroyed
                gui_thread_instance.join(timeout=2.0)
                if gui_thread_instance.is_alive(): print("DEMO WARNING: GUI thread did not close gracefully.")
            # --- End Thread Cleanup ---

            total_time = end_time_proc - start_time_proc
            if frame_count > 0 and total_time > 0:
                avg_fps = frame_count / total_time
                print(f"\n=============================================")
                print(f"DEMO FINAL STATS:")
                print(f"  Processed {frame_count} frames.")
                print(f"  Total processing time: {total_time:.2f} seconds.")
                print(f"  Average FPS: {avg_fps:.2f}")
                print(f"=============================================")
            elif frame_count > 0:
                print(f"\nDEMO FINAL STATS: Processed {frame_count} frames, but total time was too short for reliable FPS.")
            else:
                print("\nDEMO FINAL STATS: No frames were processed.")
            print("DEMO: Webcam demo finished.")
    else:
        print(f"DEMO ERROR: Invalid demo mode selected: {demo_mode}")
        app_running = False # Ensure no threads linger if mode is invalid from start

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/demo_poco_cliff.yaml', help='config file')
    parser.add_argument('--ckpt', type=str, default='data/poco_cliff.pt', help='Standard PyTorch checkpoint path')
    parser.add_argument('--trt_model', type=str, default=None, help='Path to the optimized TensorRT TorchScript model (.pt)')
    parser.add_argument('--mode', default='webcam', choices=['video', 'folder', 'directory', 'webcam'], help='Demo type')
    parser.add_argument('--vid_file', type=str, default=None, help='input video path or youtube link')
    parser.add_argument('--image_folder', type=str, default=None, help='input image folder')
    parser.add_argument('--output_folder', type=str, default='out', help='output folder')
    parser.add_argument('--stream', action='store_true', help='Use RTMP stream input instead of webcam')
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'], help='object detector')
    parser.add_argument('--yolo_img_size', type=int, default=256, help='yolo input size')
    parser.add_argument('--tracker_batch_size', type=int, default=1, help='tracker batch size')
    parser.add_argument('--batch_size', type=int, default=32, help='POCO batch size (used in non-webcam modes)')
    parser.add_argument('--display', action='store_true', help='display intermediate results (OpenCV window)')
    parser.add_argument('--smooth', action='store_true', help='smooth results (if implemented in POCO Tester)')
    parser.add_argument('--min_cutoff', type=float, default=0.004, help='one euro min cutoff')
    parser.add_argument('--beta', type=float, default=1.5, help='one euro beta')
    parser.add_argument('--no_kinematic_uncert', action='store_false', help='Do not use SMPL Kinematic for uncert')
    parser.add_argument('--wireframe', action='store_true', help='render wireframes in demo display')
    parser.add_argument('--exp', type=str, default='', help='experiment description')
    parser.add_argument('--inf_model', type=str, default='best', help='select model from checkpoint (for standard loading)')
    parser.add_argument('--skip_frame', type=int, default=1, help='skip frames')
    parser.add_argument('--dir_chunk_size', type=int, default=1000, help='dir chunk size')
    parser.add_argument('--dir_chunk', type=int, default=0, help='dir chunk index')
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'], help='tracking method')
    parser.add_argument('--staf_dir', type=str, default='/path/to/pose-track-framework', help='STAF dir')
    parser.add_argument('--no_render', action='store_true', help='disable rendering video output (for video/folder modes)')
    parser.add_argument('--render_crop', action='store_true', help='Render cropped image (in demo display)')
    parser.add_argument('--no_uncert_color', action='store_true', help='No uncertainty color (in demo display)')
    parser.add_argument('--sideview', action='store_true', help='render side viewpoint (in demo display)')
    parser.add_argument('--draw_keypoints', action='store_true', help='draw 2d keypoints (in demo display)')
    parser.add_argument('--save_obj', action='store_true', help='save obj files (for video/folder modes)')
    args = parser.parse_args()

    if args.mode == 'webcam':
        if args.tracker_batch_size != 1:
            print(f"DEMO WARNING: Forcing tracker_batch_size to 1 for webcam mode (was {args.tracker_batch_size}).")
            args.tracker_batch_size = 1
    main(args)

