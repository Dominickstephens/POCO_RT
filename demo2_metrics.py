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
os.environ['PYOPENGL_PLATFORM'] = 'egl' # Often needed for headless rendering or specific setups

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

# Define expected pose body dimension and betas
POSE_BODY_DIM = 69 # For basic 'smpl' model used in relay/viewer
NUM_BETAS = 10 # Standard number of shape parameters for SMPL data and metrics

MAGIC = b'SMPL'    # 4-byte magic

# --- Global variables for metrics and GUI ---
current_display_metrics = {"Processing FPS": 0.0, "End-to-End Latency (ms)": 0.0}
METRIC_BUFFER_DURATION_SECONDS = 5
MAX_BUFFER_LEN = int(60 * (METRIC_BUFFER_DURATION_SECONDS + 2)) # Buffer for ~7s at 60fps

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

CSV_FILENAME = "pocolib_demo_no_trt_metrics_avg.csv" # Adapted CSV filename
CSV_FIELDNAMES = [
    "Timestamp", "Condition", "Avg Processing FPS", "Avg End-to-End Latency (ms)",
    "Avg Pose Change (Euclidean Dist)", "Avg Translation Change (mm)",
    "Avg Joint Position Change (mm)", "Avg Shape Param Variance", "Avg Detection/Tracking Rate (%)"
]

gui_root = None
condition_var = None
app_running = True # General application control flag
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
    """Serialize and send data with size prefix."""
    if sock is None:
        # print("Error: Socket is not connected.") # Reduce noise if called frequently
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

# --- Metric Calculation Thread Function ---
def metrics_calculation_worker():
    global app_running, metric_data_queue
    global pose_change_buffer, translation_change_buffer, joint_pos_change_buffer
    global shape_var_buffer, detection_rate_buffer

    print("METRICS_THREAD: Metric calculation thread started.")
    while app_running:
        try:
            data_packet = metric_data_queue.get(timeout=0.1)
            if data_packet is None: # Sentinel
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
                        if pred_rotmat_curr_np.shape == pred_rotmat_prev_np.shape:
                            diff_rotmat = pred_rotmat_curr_np - pred_rotmat_prev_np
                            individual_fro_norms = np.linalg.norm(diff_rotmat, ord='fro', axis=(1,2))
                            instant_pose_change = np.sum(individual_fro_norms)
                        else:
                            print(f"METRICS_THREAD: Mismatched rotmat shapes. Curr: {pred_rotmat_curr_np.shape}, Prev: {pred_rotmat_prev_np.shape}")

                    if pred_cam_t_prev_np is not None and pred_cam_t_curr_np is not None:
                        instant_trans_change = np.linalg.norm((pred_cam_t_curr_np * 1000) - (pred_cam_t_prev_np * 1000))

                    if pred_vertices_prev_np is not None and pred_vertices_curr_np is not None:
                        vertex_diff_mm = np.linalg.norm((pred_vertices_curr_np*1000) - (pred_vertices_prev_np*1000), axis=1)
                        instant_joint_change = np.mean(vertex_diff_mm)
            
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

# --- GUI Functions ---
def setup_gui():
    global gui_root, condition_var
    gui_root = tk.Tk()
    gui_root.title("POCO Metrics Recorder (No TRT)")
    gui_root.geometry("350x200")

    def on_gui_close():
        print("GUI: Closed by user.")
        if gui_root:
            gui_root.destroy()

    gui_root.protocol("WM_DELETE_WINDOW", on_gui_close)
    tk.Label(gui_root, text="Condition:").pack(pady=(10,0))
    conditions = ["Optimal", "Low Light", "Partial Occlusion", "Fast Motion", "Static+Moving Occlusion", "Custom"]
    condition_var = tk.StringVar(gui_root)
    condition_var.set(conditions[0])
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

def record_metrics_action():
    global CSV_FILENAME, CSV_FIELDNAMES, condition_var, gui_root, METRIC_BUFFER_DURATION_SECONDS
    global fps_buffer, latency_buffer, pose_change_buffer, translation_change_buffer
    global joint_pos_change_buffer, shape_var_buffer, detection_rate_buffer

    if not app_running and gui_root and hasattr(gui_root, 'status_label_var'):
        gui_root.status_label_var.set("Application is not running or shutting down.")
        print("METRICS_REC: Record action called but app not running.")
        return

    recording_timestamp = time.time()
    current_condition = "N/A"
    if condition_var:
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
        elif nan_treatment_policy == 'include_zeros':
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
    global app_running, metric_thread_instance, gui_thread_instance
    global main_thread_prev_model_outputs, fps_buffer, latency_buffer

    app_running = True
    main_thread_prev_model_outputs = {
        "pred_rotmat_body": None, "pred_cam_t": None, "pred_vertices": None,
        "valid_for_comparison": False
    }

    if args.mode == 'webcam':
        print("DEMO: Starting metrics calculation thread...")
        metric_thread_instance = threading.Thread(target=metrics_calculation_worker, daemon=True)
        metric_thread_instance.start()
        print("DEMO: Starting GUI thread...")
        gui_thread_instance = threading.Thread(target=setup_gui, daemon=True)
        gui_thread_instance.start()
    
    sock = None
    if args.mode == 'webcam':
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((RELAY_HOST, RELAY_PORT))
            sock.settimeout(None)
            print(f"DEMO: Connected to relay server at {RELAY_HOST}:{RELAY_PORT}")
        except ConnectionRefusedError:
            print(f"DEMO WARNING: Connection refused. Is remote_relay.py running on port {RELAY_PORT}?")
            sock = None
        except socket.timeout:
            print(f"DEMO WARNING: Connection timed out to relay server on port {RELAY_PORT}.")
            sock = None
        except Exception as e:
            print(f"DEMO ERROR: Error connecting to relay server: {e}")
            traceback.print_exc()
            sock = None

    demo_mode = args.mode
    stream_mode = args.stream

    print("DEMO: Initializing POCO Tester...")
    try:
        tester = POCOTester(args)
        print("DEMO: POCO Tester Initialized.")
    except Exception as e:
        print(f"DEMO ERROR: initializing POCO Tester: {e}")
        traceback.print_exc()
        if sock: sock.close()
        app_running = False; sys.exit(1)

    # --- Variables for FPS calculation (original script) ---
    frame_count_orig = 0 # Use a different name to avoid conflict with metrics' frame_count if any
    start_time_proc_orig = time.time() # Overall processing start time
    # ---

    if demo_mode == 'video':
        print("DEMO: Running in VIDEO mode. Metrics integration focused on webcam mode.")
        # ... (original video logic) ...
        app_running = False # Assuming video mode runs once

    elif demo_mode == 'folder':
        print("DEMO: Running in FOLDER mode. Metrics integration focused on webcam mode.")
        # ... (original folder logic) ...
        app_running = False

    elif demo_mode == 'directory':
        print("DEMO: Running in DIRECTORY mode. Metrics integration focused on webcam mode.")
        # ... (original directory logic) ...
        app_running = False

    elif demo_mode == 'webcam':
        # Socket check is now done before tester initialization for webcam mode
        # if sock is None: # This check might be redundant if exit happens above
        #     print("DEMO ERROR: Cannot proceed in webcam mode without relay server if socket connection failed.")
        #     app_running = False; return

        print(f'DEMO: Webcam Demo options: \n {args}')
        print("DEMO: Using device:", tester.device)

        print("DEMO: Initializing Multi-Person Tracker...")
        try:
            mot = MPT(
                device=tester.device, batch_size=args.tracker_batch_size, display=args.display, # tracker_batch_size is 1 for webcam
                detector_type=args.detector, output_format='dict',
                yolo_img_size=args.yolo_img_size,
            )
            print("DEMO: Tracker Initialized.")
        except Exception as e:
            print(f"DEMO ERROR: Failed to initialize Multi-Person Tracker: {e}")
            if sock: sock.close()
            app_running = False; return

        print("DEMO: Opening video source...")
        cap = None
        if (stream_mode):
            rtmp_url = "rtmp://35.246.39.155:1935/live/webcam" # Example
            print(f"DEMO: Attempting to connect to RTMP stream: {rtmp_url}")
            cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
        else:
            webcam_idx = 0
            print(f"DEMO: Attempting to open webcam (index {webcam_idx})...")
            cap = cv2.VideoCapture(webcam_idx)

        if not cap or not cap.isOpened():
            print("DEMO ERROR: Cannot open video source (webcam/stream)")
            if sock: sock.close()
            app_running = False; # exit() # Let finally block handle cleanup
        else:
            print("DEMO: Video source opened successfully.")
            # ... (get video properties as in original)

        print("DEMO: Starting webcam stream. Press 'q' in OpenCV window (if displayed) or Ctrl+C to exit.")
        if args.display: print("DEMO: Press 'r' in OpenCV window to record 5-sec average metrics to CSV.")
        if sock: print("DEMO: Sending SMPL data to relay server...")

        frame_count_orig = 0 # Reset for this loop
        start_time_proc_orig = time.time() # Reset for this loop
        last_log_time = time.time() # For periodic debug log
        vertical_offset = 0.8 # For socket trans y
        depth_z = 0.0 # For socket trans z

        try: # Main try for webcam loop + cleanup
            if not cap or not cap.isOpened():
                 print("DEMO: Cannot start loop, video source not opened.")
                 app_running = False
            else:
                while app_running:
                    frame_start_loop = time.time() # For current frame's latency and FPS
                    current_frame_timestamp_for_buffer = time.time()
                    
                    instant_detection_rate_main_thread = 0.0
                    current_model_outputs_for_metric_thread = None

                    try:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            print("DEMO INFO: End of stream or failed to grab frame.")
                            app_running = False; break
                    except Exception as e:
                        print(f"DEMO ERROR: Exception while reading frame: {e}")
                        app_running = False; break
                    
                    frame_count_orig += 1
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        orig_h, orig_w = rgb_frame.shape[:2]
                    except Exception as e:
                        print(f"DEMO ERROR: Failed to convert frame to RGB: {e}")
                        main_thread_prev_model_outputs["valid_for_comparison"] = False; continue

                    dets_prepared_list = []
                    dets = np.array([])
                    try:
                        dets_raw = mot.detect_frame(rgb_frame)
                        if dets_raw is not None and dets_raw.shape[0] > 0:
                            for d_val in dets_raw:
                                if len(d_val) >= 4:
                                    x1, y1, x2, y2 = d_val[:4]
                                    w_det, h_det = x2 - x1, y2 - y1
                                    if w_det > 0 and h_det > 0:
                                        c_x, c_y = x1 + w_det / 2, y1 + h_det / 2
                                        size = max(w_det, h_det) * 1.2
                                        dets_prepared_list.append(np.array([c_x, c_y, size, size]))
                        if dets_prepared_list: dets = np.array(dets_prepared_list)
                    except Exception as e:
                        print(f"DEMO ERROR: Exception during object detection: {e}")
                        main_thread_prev_model_outputs["valid_for_comparison"] = False; continue
                    
                    if dets.shape[0] == 0: # No detections
                        instant_detection_rate_main_thread = 0.0
                        main_thread_prev_model_outputs["valid_for_comparison"] = False
                        if sock: # Send T-Pose
                            poses_body_send_t = np.zeros((1, POSE_BODY_DIM), dtype=np.float32)
                            poses_root_send_t = np.zeros((1, 3), dtype=np.float32)
                            betas_send_t = np.zeros((1, NUM_BETAS), dtype=np.float32)
                            trans_send_t = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                            data_to_send_t = {"poses_body": poses_body_send_t, "poses_root": poses_root_send_t,
                                             "betas": betas_send_t, "trans": trans_send_t}
                            if not send_data(sock, data_to_send_t):
                                print("DEMO ERROR: Failed to send T-pose (no detections). Exiting."); app_running=False; break
                        if args.display:
                            display_frame_no_det = frame.copy()
                            fps_loop_no_det = 1.0 / (time.time() - frame_start_loop) if (time.time() - frame_start_loop) > 0 else 0
                            cv2.putText(display_frame_no_det, f"FPS: {fps_loop_no_det:.2f} (No Detections)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                            cv2.imshow("Webcam Demo - POCO", display_frame_no_det)
                            key_press = cv2.waitKey(1) & 0xFF
                            if key_press == ord('q'): app_running=False; break
                            if key_press == ord('r') and gui_thread_instance: record_metrics_action()
                        # Fall through to metrics packet sending for this frame
                    else: # Detections found
                        instant_detection_rate_main_thread = 100.0
                        inp_images, bbox_info_list, focal_lengths_list, scales_list, centers_list, orig_shapes_list = [],[],[],[],[],[]
                        try:
                            num_persons_to_process = dets.shape[0]
                            for i in range(num_persons_to_process):
                                det_person_i = dets[i]
                                norm_img, _, _ = get_single_image_crop_demo(rgb_frame, det_person_i, kp_2d=None, scale=1.0, crop_size=tester.model_cfg.DATASET.IMG_RES)
                                center = [det_person_i[0], det_person_i[1]]; scale_val = det_person_i[2] / 200.0
                                inp_images.append(norm_img.float()); orig_shape = [orig_h, orig_w]
                                centers_list.append(center); orig_shapes_list.append(orig_shape); scales_list.append(scale_val)
                                bbox_info_list.append(calculate_bbox_info(center, scale_val, orig_shape))
                                focal_lengths_list.append(calculate_focal_length(orig_h, orig_w))
                            if not inp_images:
                                print("DEMO WARNING: No valid detections after batch prep."); main_thread_prev_model_outputs["valid_for_comparison"] = False; continue
                            batch = {'img': torch.stack(inp_images).to(tester.device),
                                     'bbox_info': torch.FloatTensor(bbox_info_list).to(tester.device),
                                     'focal_length': torch.FloatTensor(focal_lengths_list).to(tester.device),
                                     'scale': torch.FloatTensor(scales_list).to(tester.device),
                                     'center': torch.FloatTensor(centers_list).to(tester.device),
                                     'orig_shape': torch.FloatTensor(orig_shapes_list).to(tester.device)}
                        except Exception as e:
                            print(f"DEMO ERROR: Batch preparation: {e}"); main_thread_prev_model_outputs["valid_for_comparison"] = False; continue
                        
                        output = None
                        try:
                            tester.model.eval()
                            with torch.no_grad(): output = tester.model(batch)
                            if output is None or not output:
                                print("DEMO WARNING: Model inference returned None/empty."); main_thread_prev_model_outputs["valid_for_comparison"] = False; continue
                        except Exception as e:
                            print(f"DEMO ERROR: Model inference: {e}"); traceback.print_exc(); main_thread_prev_model_outputs["valid_for_comparison"] = False; continue

                        # --- DATA EXTRACTION FOR SOCKET (Original Logic - First Person) ---
                        pred_pose_raw_socket = output.get('pred_pose')
                        pred_shape_socket = output.get('pred_shape')
                        pred_cam_socket = output.get('pred_cam')
                        
                        poses_body_send_socket = np.zeros((1, POSE_BODY_DIM), dtype=np.float32)
                        poses_root_send_socket = np.zeros((1, 3), dtype=np.float32)
                        betas_send_socket = np.zeros((1, NUM_BETAS), dtype=np.float32)
                        trans_send_socket = np.array([[0.0, 0.8, 0.0]], dtype=np.float32)

                        if pred_pose_raw_socket is not None and pred_shape_socket is not None and pred_cam_socket is not None and \
                           pred_pose_raw_socket.shape[0] > 0: # Check if there's at least one person's output
                            
                            pred_pose_person0_socket_tensor = pred_pose_raw_socket[0]
                            pred_shape_person0_socket_tensor = pred_shape_socket[0]
                            pred_cam_person0_socket_tensor = pred_cam_socket[0]
                            pred_pose_aa_person0_socket = None

                            if pred_pose_person0_socket_tensor.ndim == 3 and pred_pose_person0_socket_tensor.shape == (24,3,3):
                                pred_pose_aa_person0_socket = rotation_matrix_to_angle_axis(pred_pose_person0_socket_tensor.reshape(-1,3,3)).reshape(72)
                            elif pred_pose_person0_socket_tensor.ndim == 1 and pred_pose_person0_socket_tensor.shape == (72,):
                                pred_pose_aa_person0_socket = pred_pose_person0_socket_tensor
                            elif pred_pose_person0_socket_tensor.ndim == 2 and pred_pose_person0_socket_tensor.shape == (24,3):
                                pred_pose_aa_person0_socket = pred_pose_person0_socket_tensor.reshape(72)
                            else: print(f"DEMO WARNING: Unrecognized pose format for socket data.")

                            if pred_pose_aa_person0_socket is not None:
                                try:
                                    global_orient_aa_socket = pred_pose_aa_person0_socket[:3]
                                    body_pose_aa_socket_np = pred_pose_aa_person0_socket[3:].cpu().numpy()
                                    global_orient_rotmat_socket = batch_rodrigues(global_orient_aa_socket.unsqueeze(0))
                                    rot_180_x_socket = torch.tensor([[[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]], dtype=global_orient_rotmat_socket.dtype, device=global_orient_rotmat_socket.device)
                                    rotated_global_orient_rotmat_socket = torch.bmm(rot_180_x_socket, global_orient_rotmat_socket)
                                    rotated_global_orient_aa_socket_np = rotation_matrix_to_angle_axis(rotated_global_orient_rotmat_socket).squeeze(0).cpu().numpy()
                                    
                                    shape_person0_socket_np = pred_shape_person0_socket_tensor.cpu().numpy()
                                    cam_person0_socket_np = pred_cam_person0_socket_tensor.cpu().numpy()

                                    if rotated_global_orient_aa_socket_np.shape==(3,) and body_pose_aa_socket_np.shape==(POSE_BODY_DIM,) and shape_person0_socket_np.shape[0]>=NUM_BETAS:
                                        tx_socket = cam_person0_socket_np[1]; ty_socket = cam_person0_socket_np[2]
                                        trans_send_socket = np.array([tx_socket, ty_socket + vertical_offset, depth_z], dtype=np.float32).reshape(1,3)
                                        poses_body_send_socket = body_pose_aa_socket_np.reshape(1, POSE_BODY_DIM)
                                        poses_root_send_socket = rotated_global_orient_aa_socket_np.reshape(1, 3)
                                        betas_send_socket = shape_person0_socket_np[:NUM_BETAS].reshape(1, NUM_BETAS)
                                except Exception as e_sock_ext: print(f"DEMO ERROR: Socket data extraction: {e_sock_ext}.")
                        
                        data_to_send_socket = {"poses_body": poses_body_send_socket, "poses_root": poses_root_send_socket,
                                             "betas": betas_send_socket, "trans": trans_send_socket}
                        if sock:
                            if not send_data(sock, data_to_send_socket):
                                print("DEMO ERROR: Failed to send data. Exiting."); app_running=False; break
                        
                        # --- DATA EXTRACTION FOR METRICS (First Person) ---
                        current_pred_betas_np_metric = pred_shape_socket[0, :NUM_BETAS].cpu().numpy()
                        smpl_vertices_output = output.get('smpl_vertices')
                        current_pred_vertices_np_metric = smpl_vertices_output[0].cpu().numpy() if smpl_vertices_output is not None and smpl_vertices_output.shape[0] > 0 else None
                        
                        pred_pose_metric_tensor = output.get('pred_pose')[0]
                        body_pose_rotmats_metric_torch = None
                        if pred_pose_metric_tensor.ndim == 3 and pred_pose_metric_tensor.shape == (24,3,3):
                            body_pose_rotmats_metric_torch = pred_pose_metric_tensor[1:]
                        elif pred_pose_metric_tensor.ndim == 1 and pred_pose_metric_tensor.shape[0] == 72:
                            body_pose_rotmats_metric_torch = batch_rodrigues(pred_pose_metric_tensor[3:].reshape(23,3).unsqueeze(0)).squeeze(0)
                        elif pred_pose_metric_tensor.ndim == 2 and pred_pose_metric_tensor.shape == (24,3):
                            body_pose_rotmats_metric_torch = batch_rodrigues(pred_pose_metric_tensor[1:].unsqueeze(0)).squeeze(0)
                        else: print(f"DEMO WARNING: Unrecognized pred_pose for metrics.")
                        current_pred_rotmat_body_np_metric = body_pose_rotmats_metric_torch.cpu().numpy() if body_pose_rotmats_metric_torch is not None else None

                        pred_cam_metric_person0_tensor = output.get('pred_cam')[0:1]
                        det_metric_person0 = dets[0:1]
                        current_pred_cam_t_np_metric = None
                        if pred_cam_metric_person0_tensor is not None and det_metric_person0 is not None and pred_cam_metric_person0_tensor.shape[0] > 0:
                            orig_cam_for_metric = convert_crop_cam_to_orig_img(cam=pred_cam_metric_person0_tensor.cpu().numpy(), bbox=det_metric_person0, img_width=orig_w, img_height=orig_h)
                            current_pred_cam_t_np_metric = orig_cam_for_metric[0]
                        else: print(f"DEMO WARNING: Could not get cam translation for metrics.")

                        if all(v is not None for v in [current_pred_rotmat_body_np_metric, current_pred_betas_np_metric, current_pred_vertices_np_metric, current_pred_cam_t_np_metric]):
                            current_model_outputs_for_metric_thread = {
                                "pred_rotmat_body": current_pred_rotmat_body_np_metric, "pred_betas": current_pred_betas_np_metric,
                                "pred_vertices": current_pred_vertices_np_metric, "pred_cam_t": current_pred_cam_t_np_metric}
                        else: main_thread_prev_model_outputs["valid_for_comparison"] = False; print(f"DEMO WARNING: Metrics data incomplete.")

                        # --- Optional Display (Original Logic) ---
                        if args.display:
                            try:
                                render_frame_display = frame.copy()
                                smpl_vertices_render = output.get('smpl_vertices')
                                pred_cam_render = output.get('pred_cam')
                                if smpl_vertices_render is not None and pred_cam_render is not None and smpl_vertices_render.shape[0] > 0:
                                    num_to_render = min(dets.shape[0], smpl_vertices_render.shape[0], pred_cam_render.shape[0])
                                    if num_to_render > 0:
                                        valid_pred_cam_render = pred_cam_render[:num_to_render].cpu().numpy()
                                        valid_dets_render = dets[:num_to_render]
                                        orig_cam_render = convert_crop_cam_to_orig_img(cam=pred_cam.cpu().numpy(), bbox=dets, img_width=orig_w, img_height=orig_h)
                                        local_renderer = Renderer(resolution=(orig_w,orig_h), orig_img=True, wireframe=args.wireframe) # Re-init renderer
                                        render_frame_display_rgb = cv2.cvtColor(render_frame_display, cv2.COLOR_BGR2RGB)
                                        for i in range(num_to_render):
                                            verts_render = smpl_vertices_render[i].cpu().numpy()
                                            if np.isnan(verts_render).any() or np.isinf(verts_render).any(): continue
                                            render_frame_display_rgb = local_renderer.render(render_frame_display_rgb, verts_render, cam=orig_cam_render[i], color=[0.7,0.7,0.7])
                                        render_frame_display = cv2.cvtColor(render_frame_display_rgb, cv2.COLOR_RGB2BGR)
                                fps_loop_display = 1.0 / (time.time() - frame_start_loop) if (time.time() - frame_start_loop) > 0 else 0
                                cv2.putText(render_frame_display, f"FPS: {fps_loop_display:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                                cv2.putText(render_frame_display, f"Latency: {(time.time()-frame_start_loop)*1000:.1f}ms",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                                cv2.imshow("Webcam Demo - POCO", render_frame_display)
                                key_press = cv2.waitKey(1) & 0xFF
                                if key_press == ord('q'): app_running=False; break
                                if key_press == ord('r') and gui_thread_instance: record_metrics_action()
                            except Exception as e_disp: print(f"DEMO ERROR: Display update: {e_disp}"); traceback.print_exc()
                        else: time.sleep(0.2) # No display sleep

                    # --- Send data to metrics thread ---
                    data_packet_for_metric_thread = {
                        "timestamp": current_frame_timestamp_for_buffer,
                        "current_outputs": current_model_outputs_for_metric_thread,
                        "prev_outputs_for_comparison": main_thread_prev_model_outputs.copy(),
                        "detection_rate": instant_detection_rate_main_thread}
                    try: metric_data_queue.put_nowait(data_packet_for_metric_thread)
                    except queue.Full: print("DEMO WARNING: Metric queue full.")

                    if current_model_outputs_for_metric_thread:
                        main_thread_prev_model_outputs["pred_rotmat_body"] = current_model_outputs_for_metric_thread["pred_rotmat_body"]
                        main_thread_prev_model_outputs["pred_cam_t"] = current_model_outputs_for_metric_thread["pred_cam_t"]
                        main_thread_prev_model_outputs["pred_vertices"] = current_model_outputs_for_metric_thread["pred_vertices"]
                        main_thread_prev_model_outputs["valid_for_comparison"] = True
                    else: main_thread_prev_model_outputs["valid_for_comparison"] = False
                    
                    loop_end_time_main_thread = time.time()
                    processing_time_main_thread = loop_end_time_main_thread - frame_start_loop
                    instant_fps_main_thread = 1.0 / processing_time_main_thread if processing_time_main_thread > 1e-6 else 0.0
                    fps_buffer.append((current_frame_timestamp_for_buffer, instant_fps_main_thread))
                    latency_buffer.append((current_frame_timestamp_for_buffer, processing_time_main_thread * 1000))

                    current_time_log = time.time()
                    if current_time_log - last_log_time >= 10.0:
                        avg_fps_so_far = frame_count_orig / (current_time_log - start_time_proc_orig) if (current_time_log - start_time_proc_orig) > 0 else 0
                        print(f"DEMO DEBUG: Frame {frame_count_orig}, Current Avg FPS (Overall): {avg_fps_so_far:.2f}")
                        last_log_time = current_time_log
                    
                    if args.display:
                        try:
                            if cv2.getWindowProperty("Webcam Demo - POCO", cv2.WND_PROP_VISIBLE) < 1:
                                print("DEMO: OpenCV window closed."); app_running = False
                        except cv2.error: print("DEMO: OpenCV window destroyed."); app_running = False
        
        except KeyboardInterrupt: print("\nDEMO: Ctrl+C detected. Exiting..."); app_running = False
        except Exception as e: print(f"\nDEMO ERROR: Main loop: {e}"); traceback.print_exc(); app_running = False
        finally: # Cleanup for webcam mode
            print("\nDEMO: Cleaning up webcam resources...")
            app_running = False # Ensure flag is set for threads

            if cap is not None and cap.isOpened(): cap.release()
            if args.display: cv2.destroyAllWindows()
            if sock: sock.close()

            if metric_thread_instance and metric_thread_instance.is_alive():
                print("DEMO: Signaling metric thread to stop...")
                try: metric_data_queue.put_nowait(None)
                except queue.Full: print("DEMO WARNING: Metric queue full on shutdown.")
                metric_thread_instance.join(timeout=2.0)
                if metric_thread_instance.is_alive(): print("DEMO WARNING: Metric thread didn't join.")
            
            if gui_thread_instance and gui_thread_instance.is_alive():
                print("DEMO: Closing GUI thread...")
                if gui_root:
                    try: gui_root.destroy()
                    except Exception: pass
                gui_thread_instance.join(timeout=2.0)
                if gui_thread_instance.is_alive(): print("DEMO WARNING: GUI thread didn't join.")

            end_time_proc_orig = time.time()
            total_time_orig = end_time_proc_orig - start_time_proc_orig
            avg_fps_orig = frame_count_orig / total_time_orig if total_time_orig > 0 else 0
            print(f"DEMO FINAL STATS: Processed {frame_count_orig} frames in {total_time_orig:.2f}s. Avg FPS: {avg_fps_orig:.2f}")
            print("DEMO: Webcam demo finished.")
    else:
        print(f"DEMO ERROR: Invalid demo mode: {demo_mode}")
        app_running = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/demo_poco_cliff.yaml', help='config file')
    parser.add_argument('--ckpt', type=str, default='data/poco_cliff.pt', help='checkpoint path')
    parser.add_argument('--mode', default='webcam', choices=['video', 'folder', 'directory', 'webcam'], help='Demo type')
    parser.add_argument('--vid_file', type=str, default=None, help='input video path or youtube link')
    parser.add_argument('--image_folder', type=str, default=None, help='input image folder')
    parser.add_argument('--output_folder', type=str, default='out', help='output folder')
    parser.add_argument('--stream', type=str2bool, default=False, help='RTMP stream input')
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'], help='object detector')
    parser.add_argument('--yolo_img_size', type=int, default=256, help='yolo input size')
    parser.add_argument('--tracker_batch_size', type=int, default=1, help='tracker batch size')
    parser.add_argument('--batch_size', type=int, default=32, help='POCO batch size')
    parser.add_argument('--display', action='store_true', help='display intermediate results')
    parser.add_argument('--smooth', action='store_true', help='smooth results')
    parser.add_argument('--min_cutoff', type=float, default=0.004, help='one euro min cutoff')
    parser.add_argument('--beta', type=float, default=1.5, help='one euro beta')
    parser.add_argument('--no_kinematic_uncert', action='store_false', help='Do not use SMPL Kinematic for uncert')
    parser.add_argument('--wireframe', action='store_true', help='render wireframes')
    parser.add_argument('--exp', type=str, default='', help='experiment description')
    parser.add_argument('--inf_model', type=str, default='best', help='select model from checkpoint')
    parser.add_argument('--skip_frame', type=int, default=1, help='skip frames')
    parser.add_argument('--dir_chunk_size', type=int, default=1000, help='dir chunk size')
    parser.add_argument('--dir_chunk', type=int, default=0, help='dir chunk index')
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'], help='tracking method')
    parser.add_argument('--staf_dir', type=str, default='/path/to/pose-track-framework', help='STAF dir')
    parser.add_argument('--no_render', action='store_true', help='disable rendering video output')
    parser.add_argument('--render_crop', action='store_true', help='Render cropped image')
    parser.add_argument('--no_uncert_color', action='store_true', help='No uncertainty color')
    parser.add_argument('--sideview', action='store_true', help='render side viewpoint')
    parser.add_argument('--draw_keypoints', action='store_true', help='draw 2d keypoints')
    parser.add_argument('--save_obj', action='store_true', help='save obj files')
    args = parser.parse_args()

    if args.mode == 'webcam' and args.tracker_batch_size != 1:
        print(f"DEMO WARNING: Forcing tracker_batch_size to 1 for webcam mode (was {args.tracker_batch_size}).")
        args.tracker_batch_size = 1
        
    main(args)

