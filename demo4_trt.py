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

MIN_NUM_FRAMES = 0

# Define connection details for remote_relay.py
RELAY_HOST = 'localhost'
RELAY_PORT = 9999 # Port for demo -> relay communication

# Define expected pose body dimension and betas for SMPL data sending
POSE_BODY_DIM = 69 # For basic 'smpl' model used in relay/viewer
NUM_BETAS = 10 # Standard number of shape parameters

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
        # print("Error: Socket is not connected.") # Reduce noise
        return False
    try:
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        length  = len(payload)
        crc32   = zlib.crc32(payload) & 0xFFFFFFFF
        header  = MAGIC + struct.pack('>I I', length, crc32)
        sock.sendall(header + payload)                      # Send data
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


def main(args):

    # Determine device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"DEMO: Using device: {device}")

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
            # Load the TorchScript model containing the TensorRT engine
            model = torch.jit.load(args.trt_model).to(device).eval()
            print("DEMO: TensorRT model loaded successfully.")
            using_trt_model = True
            # Still initialize POCOTester to get configuration easily
            print("DEMO: Initializing POCOTester for configuration...")
            try:
                tester = POCOTester(args)
                model_cfg = tester.model_cfg
                print("DEMO: POCOTester initialized for config.")
            except Exception as e:
                print(f"DEMO ERROR: Failed to initialize POCOTester for config in TRT mode: {e}")
                traceback.print_exc()
                if sock: sock.close()
                sys.exit(1)

        except Exception as e:
            print(f"DEMO ERROR: Failed to load TensorRT model: {e}")
            print("DEMO INFO: Ensure 'import torch_tensorrt' is present and successful.")
            traceback.print_exc()
            if sock: sock.close()
            sys.exit(1) # Exit if TRT model loading fails
    else:
        if args.trt_model:
             print(f"DEMO WARNING: TensorRT model path provided but not found: {args.trt_model}. Falling back to standard model.")
        # Load standard model using POCOTester
        print("DEMO: Initializing POCOTester to load standard PyTorch model...")
        try:
            tester = POCOTester(args)
            model = tester.model # Use the model loaded by the tester
            model_cfg = tester.model_cfg
            model.eval() # Ensure model is in eval mode
            print("DEMO: POCO Tester Initialized with standard model.")
        except Exception as e:
            print(f"DEMO ERROR: Failed to initialize POCOTester with standard model: {e}")
            traceback.print_exc()
            if sock: sock.close()
            sys.exit(1)

    # --- Sanity Check ---
    if model is None or model_cfg is None:
        print("DEMO ERROR: Model or configuration failed to load.")
        if sock: sock.close()
        sys.exit(1)

    # --- Handle different demo modes ---
    demo_mode = args.mode
    stream_mode = args.stream

    # --- Variables for FPS calculation (initialized before potential loop) ---
    frame_count = 0
    start_time_proc = time.time() # Initialize start time here
    cap = None # Initialize cap to None

    if demo_mode == 'video':
        print("DEMO: Running in VIDEO mode. Graceful shutdown/FPS reporting not implemented for this mode here.")
        # TODO: Adapt video mode with try...finally if needed
        video_file = args.vid_file
        if not isfile(video_file): exit(f'Input video \"{video_file}\" does not exist!')
        output_path = join(args.output_folder, basename(video_file).replace('.mp4', '_' + args.exp))
        input_path = join(dirname(video_file), basename(video_file).replace('.mp4', '_' + args.exp))
        os.makedirs(input_path, exist_ok=True); os.makedirs(output_path, exist_ok=True)
        # ... (rest of original video logic - needs adaptation for TRT input/output and try/finally) ...

    elif demo_mode == 'folder':
        print("DEMO: Running in FOLDER mode. Graceful shutdown/FPS reporting not implemented for this mode here.")
         # TODO: Adapt folder mode with try...finally if needed
        # ... (Original folder logic - needs adaptation for TRT input/output and try/finally) ...

    elif demo_mode == 'directory':
        print("DEMO: Running in DIRECTORY mode. Graceful shutdown/FPS reporting not implemented for this mode here.")
        # TODO: Adapt directory mode with try...finally if needed
        # ... (Original directory logic - needs adaptation for TRT input/output and try/finally) ...

    elif demo_mode == 'webcam':
        # --- Webcam Mode Modifications ---
        if sock is None and not args.display:
             print("DEMO WARNING: Socket not connected. Cannot send data. Running display only.")
        elif sock is None and args.display:
             print("DEMO WARNING: Socket not connected. Cannot send data. Displaying locally.")


        print(f'DEMO: Webcam Demo options: \n {args}')

        print("DEMO: Initializing Multi-Person Tracker...")
        try:
            mot = MPT(
                device=device, batch_size=1, display=args.display, # Use device directly
                detector_type=args.detector, output_format='dict',
                yolo_img_size=args.yolo_img_size,
            )
            print("DEMO: Tracker Initialized.")
        except Exception as e:
            print(f"DEMO ERROR: Failed to initialize Multi-Person Tracker: {e}")
            if sock: sock.close()
            return # Changed from exit to return to allow finally block execution if needed

        # --- Video Capture ---
        print("DEMO: Opening video source...")
        if (stream_mode):
            rtmp_url = "rtmp://35.246.39.155:1935/live/webcam" # Example URL
            print(f"DEMO: Attempting to connect to RTMP stream: {rtmp_url}")
            cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
        else:
            webcam_idx = 0 # Default webcam index
            print(f"DEMO: Attempting to open webcam (index {webcam_idx})...")
            cap = cv2.VideoCapture(webcam_idx)


        if not cap or not cap.isOpened(): # Check if cap is None or not opened
            print("DEMO ERROR: Cannot open video source (webcam/stream)")
            if sock: sock.close()
            # No need to exit here, finally block will handle cleanup if cap exists
        else:
            print("DEMO: Video source opened successfully.")
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"DEMO: Video source resolution: {frame_width}x{frame_height}, FPS: {video_fps if video_fps > 0 else 'N/A'}")

        print("DEMO: Starting webcam stream. Press 'q' in OpenCV window (if displayed) or Ctrl+C in terminal to exit.")
        if sock: print("DEMO: Sending SMPL data to relay server...")

        # Reset frame count and start time specifically for the webcam loop
        frame_count = 0
        start_time_proc = time.time()
        last_log_time = time.time()

        # --- Fixed parameters for data sending (example values) ---
        # These might need adjustment based on your visualization setup
        vertical_offset = 0.8
        depth_z = 0.0
        # --- End fixed parameters ---

        # === Graceful Shutdown and FPS Reporting Setup ===
        try: # Start the main processing block that needs cleanup
            if not cap or not cap.isOpened():
                 print("DEMO: Cannot start loop, video source not opened.")
                 # Allow finally block to run for cleanup if needed

            else: # Only run the loop if the capture device is open
                while True: # The main processing loop
                    frame_start_loop = time.time()
                    try:
                        ret, frame = cap.read()
                        if not ret:
                            print("DEMO INFO: End of stream or failed to grab frame.")
                            break # Exit loop if no frame
                    except Exception as e:
                        print(f"DEMO ERROR: Exception while reading frame: {e}")
                        break

                    # Check if frame is None right after reading
                    if frame is None:
                        print("DEMO WARNING: Grabbed frame is None. Skipping frame.")
                        continue # Skip processing this frame

                    frame_count += 1 # Increment frame count only for successfully read frames

                    # --- Frame Preprocessing ---
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        orig_h, orig_w = rgb_frame.shape[:2]
                    except Exception as e:
                        print(f"DEMO ERROR: Failed to convert frame {frame_count} to RGB: {e}")
                        continue # Skip this frame

                    # --- Detection ---
                    dets_prepared_list = []
                    dets = np.array([]) # Initialize dets
                    try:
                        dets_raw = mot.detect_frame(rgb_frame)
                        if dets_raw is not None and dets_raw.shape[0] > 0:
                            for d in dets_raw:
                                if len(d) >= 4:
                                    x1, y1, x2, y2 = d[:4]
                                    w, h = x2 - x1, y2 - y1
                                    if w > 0 and h > 0:
                                        c_x, c_y = x1 + w / 2, y1 + h / 2
                                        size = max(w, h) * 1.2
                                        bbox_prepared = np.array([c_x, c_y, size, size])
                                        dets_prepared_list.append(bbox_prepared)

                        if dets_prepared_list:
                            dets = np.array(dets_prepared_list)
                        else:
                            dets = np.array([]) # Ensure it's an empty array

                    except Exception as e:
                        print(f"DEMO ERROR: Exception during object detection on frame {frame_count}: {e}")
                        continue # Skip this frame

                    # --- Handle No Detections ---
                    if dets.shape[0] == 0:
                        # Send T-Pose if no detections and socket is connected
                        if sock:
                            poses_body_send = np.zeros((1, POSE_BODY_DIM), dtype=np.float32)
                            poses_root_send = np.zeros((1, 3), dtype=np.float32)
                            betas_send = np.zeros((1, NUM_BETAS), dtype=np.float32)
                            trans_send = np.array([[0.0, 0.0, 0.0]], dtype=np.float32) # Send at origin
                            data_to_send = {
                                "poses_body": poses_body_send, "poses_root": poses_root_send,
                                "betas": betas_send, "trans": trans_send,
                            }
                            if not send_data(sock, data_to_send):
                                print("DEMO ERROR: Failed to send T-pose data (no detections). Exiting loop.")
                                break # Exit loop on send failure

                        # Update display if enabled
                        if args.display:
                            display_frame = frame.copy() # Work on a copy
                            fps_loop = 1.0 / (time.time() - frame_start_loop) if (time.time() - frame_start_loop) > 0 else 0
                            cv2.putText(display_frame, f"FPS: {fps_loop:.2f} (No Detections)", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imshow("Webcam Demo - POCO", display_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("DEMO: 'q' pressed, exiting loop.")
                                break # Exit loop if 'q' is pressed
                        continue # Skip rest of the loop for this frame

                    # --- Prepare Batch for Model ---
                    inp_images = []
                    bbox_info_list = []
                    focal_lengths_list = []
                    scales_list = []
                    centers_list = []
                    orig_shapes_list = []
                    try:
                        for det in dets:
                            if len(det) == 4:
                                # Use model_cfg from the loaded tester/config
                                norm_img, _, _ = get_single_image_crop_demo(
                                    rgb_frame, det, kp_2d=None, scale=1.0,
                                    crop_size=model_cfg.DATASET.IMG_RES # Use config value
                                )
                                center = [det[0], det[1]]
                                scale_val = det[2] / 200.0 # Example scaling
                                inp_images.append(norm_img.float())
                                orig_shape = [orig_h, orig_w]
                                centers_list.append(center)
                                orig_shapes_list.append(orig_shape)
                                scales_list.append(scale_val)
                                bbox_info = calculate_bbox_info(center, scale_val, orig_shape)
                                bbox_info_list.append(bbox_info)
                                focal_length = calculate_focal_length(orig_h, orig_w)
                                focal_lengths_list.append(focal_length)

                        if not inp_images:
                            print(f"DEMO WARNING: No valid detections to process after batch prep on frame {frame_count}.")
                            continue # Skip this frame

                        # --- Create Batch Dictionary (Used by both TRT and standard model paths) ---
                        batch = {
                            'img': torch.stack(inp_images).to(device),
                            'bbox_info': torch.FloatTensor(bbox_info_list).to(device),
                            'focal_length': torch.FloatTensor(focal_lengths_list).to(device),
                            'scale': torch.FloatTensor(scales_list).to(device),
                            'center': torch.FloatTensor(centers_list).to(device),
                            'orig_shape': torch.FloatTensor(orig_shapes_list).to(device),
                        }
                    except Exception as e:
                        print(f"DEMO ERROR: Exception during batch preparation on frame {frame_count}: {e}")
                        continue # Skip this frame


                    # --- Run Model Inference ---
                    output = None # Initialize output dictionary
                    try:
                        model.eval() # Ensure model is in eval mode
                        with torch.no_grad():
                            if using_trt_model:
                                # --- Prepare Input Tuple for TRT Model ---
                                try:
                                    # Ensure tensors exist and have the correct batch dim
                                    num_dets_in_batch = batch['img'].shape[0]
                                    trt_input_tuple = tuple(batch[key][:num_dets_in_batch] for key in TRT_INPUT_KEYS)
                                except KeyError as e:
                                    print(f"DEMO ERROR: Missing key '{e}' in batch dictionary needed for TRT input tuple on frame {frame_count}.")
                                    continue # Skip frame if input can't be prepared
                                except Exception as e_prep:
                                    print(f"DEMO ERROR: Failed preparing TRT input tuple on frame {frame_count}: {e_prep}")
                                    continue # Skip frame

                                # --- Call TRT Model ---
                                # Unpack the tuple into separate arguments using *
                                output_tuple = model(*trt_input_tuple)

                                # --- Reconstruct Output Dictionary from TRT Output Tuple ---
                                if len(output_tuple) == len(TRT_OUTPUT_KEYS):
                                    output = {key: tensor for key, tensor in zip(TRT_OUTPUT_KEYS, output_tuple)}
                                else:
                                    print(f"DEMO ERROR: TRT model output tuple length ({len(output_tuple)}) "
                                          f"does not match expected keys ({len(TRT_OUTPUT_KEYS)}) on frame {frame_count}.")
                                    output = {} # Set empty dict on error

                            else:
                                # --- Call Standard PyTorch Model ---
                                output = model(batch) # Call the original model loaded via POCOTester

                        if output is None or not output: # Check if output is None or empty
                            print(f"DEMO WARNING: Model inference returned None or empty output on frame {frame_count}.")
                            continue # Skip this frame

                    except Exception as e:
                        print(f"DEMO ERROR: Exception during model inference on frame {frame_count}: {e}")
                        traceback.print_exc() # Print full traceback for inference errors
                        continue # Skip this frame

                    # --- *** DATA EXTRACTION AND SENDING (WITH POSE CONVERSION) *** ---
                    pred_pose_raw = output.get('pred_pose')
                    pred_shape = output.get('pred_shape')
                    pred_cam = output.get('pred_cam')

                    # Default values (T-Pose)
                    poses_body_send = np.zeros((1, POSE_BODY_DIM), dtype=np.float32)
                    poses_root_send = np.zeros((1, 3), dtype=np.float32)
                    betas_send = np.zeros((1, NUM_BETAS), dtype=np.float32)
                    trans_send = np.array([[0.0, 0.8, 0.0]], dtype=np.float32) # Default translation

                    data_valid = False
                    pred_pose_aa = None

                    if pred_pose_raw is not None and pred_shape is not None and pred_cam is not None:
                        if pred_pose_raw.shape[0] > 0 and pred_shape.shape[0] > 0 and pred_cam.shape[0] > 0:
                            batch_size_out = pred_pose_raw.shape[0]
                            if not (batch_size_out == pred_shape.shape[0] == pred_cam.shape[0]):
                                print(f"DEMO WARNING: Mismatched batch sizes in output tensors on frame {frame_count}. Sending T-Pose.")
                            else:
                                # --- Check pose format and convert if necessary ---
                                try:
                                    current_pose_shape = tuple(pred_pose_raw.shape[1:])
                                    if current_pose_shape == (24, 3, 3): # RotMat
                                        pred_pose_rotmat_flat = pred_pose_raw.reshape(-1, 3, 3)
                                        pred_pose_aa_flat = rotation_matrix_to_angle_axis(pred_pose_rotmat_flat)
                                        pred_pose_aa = pred_pose_aa_flat.reshape(batch_size_out, 72)
                                    elif current_pose_shape == (72,): # AxisAngle (B, 72)
                                        pred_pose_aa = pred_pose_raw
                                    elif current_pose_shape == (24, 3): # AxisAngle (B, 24, 3)
                                        pred_pose_aa = pred_pose_raw.reshape(batch_size_out, 72)
                                    else:
                                        print(f"DEMO WARNING: Unrecognized pose format {current_pose_shape} on frame {frame_count}. Sending T-Pose.")
                                        pred_pose_aa = None
                                except Exception as e:
                                    print(f"DEMO ERROR: Exception during pose conversion on frame {frame_count}: {e}. Sending T-Pose.")
                                    pred_pose_aa = None

                                # --- Extract data for the first person ---
                                if pred_pose_aa is not None:
                                    try:
                                        pose_person0_aa = pred_pose_aa[0]
                                        shape_person0 = pred_shape[0].cpu().numpy()
                                        cam_person0 = pred_cam[0].cpu().numpy()

                                        # Apply 180-degree X-axis rotation fix
                                        global_orient_aa = pose_person0_aa[:3]
                                        body_pose_aa = pose_person0_aa[3:]
                                        global_orient_rotmat = batch_rodrigues(global_orient_aa.unsqueeze(0))
                                        rot_180_x = torch.tensor([[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]],
                                                                 dtype=global_orient_rotmat.dtype, device=global_orient_rotmat.device)
                                        rotated_global_orient_rotmat = torch.bmm(rot_180_x, global_orient_rotmat)
                                        rotated_global_orient_aa = rotation_matrix_to_angle_axis(rotated_global_orient_rotmat).squeeze(0)

                                        poses_root_extracted = rotated_global_orient_aa.cpu().numpy()
                                        poses_body_extracted = body_pose_aa.cpu().numpy()
                                        betas_extracted = shape_person0[:NUM_BETAS]

                                        if poses_root_extracted.shape == (3,) and poses_body_extracted.shape == (POSE_BODY_DIM,) and betas_extracted.shape[0] >= NUM_BETAS:
                                            tx = cam_person0[1]
                                            ty = cam_person0[2]
                                            trans_extracted = np.array([tx, ty + vertical_offset, depth_z], dtype=np.float32)

                                            poses_body_send = poses_body_extracted.reshape(1, POSE_BODY_DIM)
                                            poses_root_send = poses_root_extracted.reshape(1, 3)
                                            betas_send = betas_extracted.reshape(1, NUM_BETAS)
                                            trans_send = trans_extracted.reshape(1, 3)
                                            data_valid = True
                                        else:
                                            print(f"DEMO WARNING: Extracted pose/shape dimensions incorrect on frame {frame_count}. Sending T-Pose.")
                                            data_valid = False
                                    except Exception as e:
                                        print(f"DEMO ERROR: Exception during data extraction/fix on frame {frame_count}: {e}. Sending T-Pose.")
                                        data_valid = False
                        else:
                            print(f"DEMO WARNING: Model output tensors empty on frame {frame_count}. Sending T-Pose.")
                    else:
                        missing_keys = [k for k in ['pred_pose', 'pred_shape', 'pred_cam'] if output.get(k) is None]
                        print(f"DEMO WARNING: Keys {missing_keys} not found on frame {frame_count}. Sending T-Pose.")

                    # Package data
                    data_to_send = {
                        "poses_body": poses_body_send, "poses_root": poses_root_send,
                        "betas": betas_send, "trans": trans_send,
                    }

                    # Send data over socket
                    if sock:
                        try:
                            if not send_data(sock, data_to_send):
                                print("DEMO ERROR: Failed to send data to relay server. Exiting loop.")
                                break # Exit loop on send failure
                        except Exception as e:
                            print(f"DEMO ERROR: Exception during socket send on frame {frame_count}: {e}. Exiting loop.")
                            break # Exit loop on send failure

                    # --- Optional Display ---
                    if args.display:
                        try:
                            render_frame_display = frame.copy() # Work on a copy
                            smpl_vertices = output.get('smpl_vertices') # Get vertices from output dict

                            if smpl_vertices is not None and pred_cam is not None and smpl_vertices.shape[0] > 0 and pred_cam.shape[0] > 0:
                                num_to_render = min(len(dets), smpl_vertices.shape[0], pred_cam.shape[0])
                                if num_to_render > 0:
                                    # Ensure pred_cam used here matches the batch size of vertices
                                    valid_pred_cam = pred_cam[:num_to_render].cpu().numpy()
                                    valid_dets = dets[:num_to_render]

                                    orig_cam = convert_crop_cam_to_orig_img(
                                        cam=valid_pred_cam,
                                        bbox=valid_dets,
                                        img_width=orig_w, img_height=orig_h
                                    )
                                    renderer = Renderer(resolution=(orig_w, orig_h), orig_img=True, wireframe=args.wireframe)
                                    render_frame_display_rgb = cv2.cvtColor(render_frame_display, cv2.COLOR_BGR2RGB)

                                    for i in range(num_to_render):
                                        verts = smpl_vertices[i].cpu().numpy()
                                        if np.isnan(verts).any() or np.isinf(verts).any():
                                            print(f"DEMO WARNING: NaN/Inf detected in vertices for person {i} on frame {frame_count}, skipping render.")
                                            continue
                                        render_frame_display_rgb = renderer.render(
                                            render_frame_display_rgb, verts, cam=orig_cam[i], color=[0.7, 0.7, 0.7]
                                        )
                                    render_frame_display = cv2.cvtColor(render_frame_display_rgb, cv2.COLOR_RGB2BGR)

                            fps_loop = 1.0 / (time.time() - frame_start_loop) if (time.time() - frame_start_loop) > 0 else 0
                            cv2.putText(render_frame_display, f"FPS: {fps_loop:.2f}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.imshow("Webcam Demo - POCO", render_frame_display)

                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("DEMO: 'q' pressed, exiting loop.")
                                break # Exit loop if 'q' is pressed
                        except Exception as e:
                            print(f"DEMO ERROR: Exception during display update on frame {frame_count}: {e}")
                            traceback.print_exc()
                            # Decide if you want to break the loop on display error, or just continue
                            # break

                    else:
                        time.sleep(0.001) # Prevent busy-waiting if not displaying

                    # Log performance periodically
                    current_time = time.time()
                    if current_time - last_log_time >= 10.0: # Log every 10 seconds
                        loop_time_avg = (current_time - start_time_proc) / frame_count if frame_count > 0 else 0
                        fps_avg_so_far = frame_count / (current_time - start_time_proc) if (current_time - start_time_proc) > 0 else 0
                        print(f"DEMO DEBUG: Frame {frame_count}, Avg Loop Time: {loop_time_avg:.4f}s, Current Avg FPS: {fps_avg_so_far:.2f}")
                        last_log_time = current_time

        # === Handle KeyboardInterrupt (Ctrl+C) ===
        except KeyboardInterrupt:
            print("\nDEMO: Ctrl+C detected. Exiting gracefully...")
            # The 'finally' block will handle cleanup and FPS calculation.

        # === Handle other potential exceptions during the loop ===
        except Exception as e:
            print(f"\nDEMO ERROR: An unexpected error occurred in the main loop: {e}")
            traceback.print_exc()
            # The 'finally' block will still execute for cleanup.

        # === Cleanup and Final FPS Calculation (ALWAYS RUNS) ===
        finally:
            print("\nDEMO: Cleaning up resources...")
            end_time_proc = time.time() # Record end time

            if cap is not None and cap.isOpened():
                print("DEMO: Releasing video capture device.")
                cap.release()

            if args.display:
                print("DEMO: Closing OpenCV windows.")
                cv2.destroyAllWindows()

            if sock:
                print("DEMO: Closing socket connection.")
                sock.close()

            # --- Calculate and Print Final Average FPS ---
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
                print(f"\nDEMO FINAL STATS: Processed {frame_count} frames, but total time was too short to calculate reliable FPS.")
            else:
                print("\nDEMO FINAL STATS: No frames were processed.")

            print("DEMO: Webcam demo finished.")
        # End of try...finally block for webcam mode

    else: # Should not happen if mode validation is correct
        print(f"DEMO ERROR: Invalid demo mode selected: {demo_mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --- Essential Arguments ---
    parser.add_argument('--cfg', type=str, default='configs/demo_poco_cliff.yaml', help='config file')
    parser.add_argument('--ckpt', type=str, default='data/poco_cliff.pt', help='Standard PyTorch checkpoint path')
    parser.add_argument('--trt_model', type=str, default=None, help='Path to the optimized TensorRT TorchScript model (.pt)') # <<< NEW ARGUMENT
    parser.add_argument('--mode', default='webcam', choices=['video', 'folder', 'directory', 'webcam'], help='Demo type')

    # --- Arguments for different modes ---
    parser.add_argument('--vid_file', type=str, default=None, help='input video path or youtube link')
    parser.add_argument('--image_folder', type=str, default=None, help='input image folder')
    parser.add_argument('--output_folder', type=str, default='out', help='output folder')
    parser.add_argument('--stream', action='store_true', help='Use RTMP stream input instead of webcam') # Changed to action='store_true'

    # --- Detection/Tracking Arguments ---
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'], help='object detector')
    parser.add_argument('--yolo_img_size', type=int, default=256, help='yolo input size')
    parser.add_argument('--tracker_batch_size', type=int, default=1, help='tracker batch size (should be 1 for webcam)') # Use 1 for webcam

    # --- POCO/Model Arguments ---
    parser.add_argument('--batch_size', type=int, default=32, help='POCO batch size (used in non-webcam modes)')

    # --- Optional Arguments ---
    parser.add_argument('--display', action='store_true', help='display intermediate results (OpenCV window)')
    parser.add_argument('--smooth', action='store_true', help='smooth results (if implemented in POCO Tester)')
    parser.add_argument('--min_cutoff', type=float, default=0.004, help='one euro min cutoff')
    parser.add_argument('--beta', type=float, default=1.5, help='one euro beta')
    parser.add_argument('--no_kinematic_uncert', action='store_false',
                        help='Do not use SMPL Kinematic for uncert')
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

    # --- Ensure tracker batch size is 1 for webcam mode ---
    if args.mode == 'webcam':
        if args.tracker_batch_size != 1:
            print(f"DEMO WARNING: Forcing tracker_batch_size to 1 for webcam mode (was {args.tracker_batch_size}).")
            args.tracker_batch_size = 1

    main(args)