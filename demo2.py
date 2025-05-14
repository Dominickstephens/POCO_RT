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
# It's good practice to add error handling for these imports
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

# Define expected pose body dimension and betas
POSE_BODY_DIM = 69 # For basic 'smpl' model used in relay/viewer
NUM_BETAS = 10 # Standard number of shape parameters

MAGIC = b'SMPL'    # 4-byte magic

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
        print("Error: Socket is not connected.")
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

    # Initialize socket client (only in webcam mode for this example)
    sock = None
    if args.mode == 'webcam':
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Add a timeout for connection attempt
            sock.settimeout(5.0)
            sock.connect((RELAY_HOST, RELAY_PORT))
            # Reset timeout for sending data or set a different one if needed
            sock.settimeout(None)
            print(f"DEMO: Connected to relay server at {RELAY_HOST}:{RELAY_PORT}")
        except ConnectionRefusedError:
            print(f"DEMO ERROR: Connection refused. Is remote_relay.py running and listening on port {RELAY_PORT}?")
            sock = None # Ensure sock is None if connection failed
        except socket.timeout:
             print(f"DEMO ERROR: Connection timed out. Is remote_relay.py running and listening on port {RELAY_PORT}?")
             sock = None
        except Exception as e:
            print(f"DEMO ERROR: Error connecting to relay server: {e}")
            traceback.print_exc()
            sock = None

    demo_mode = args.mode
    stream_mode = args.stream

    # Initialize the POCO tester
    print("DEMO: Initializing POCO Tester...")
    try:
        tester = POCOTester(args)
        print("DEMO: POCO Tester Initialized.")
    except Exception as e:
         print(f"DEMO ERROR: initializing POCO Tester: {e}")
         traceback.print_exc()
         if sock: sock.close()
         sys.exit(1)


    # --- Handle different demo modes ---
    # Note: Socket sending logic is currently only added to 'webcam' mode.
    # You can adapt the data extraction and sending logic to other modes if needed.

    if demo_mode == 'video':
        print("DEMO: Running in VIDEO mode. Data sending to relay not implemented.")
        video_file = args.vid_file
        if not isfile(video_file): exit(f'Input video \"{video_file}\" does not exist!')
        output_path = join(args.output_folder, basename(video_file).replace('.mp4', '_' + args.exp))
        input_path = join(dirname(video_file), basename(video_file).replace('.mp4', '_' + args.exp))
        os.makedirs(input_path, exist_ok=True); os.makedirs(output_path, exist_ok=True)
        # ... (rest of original video logic) ...

    elif demo_mode == 'folder':
        print("DEMO: Running in FOLDER mode. Data sending to relay not implemented.")
        # ... (Original folder logic) ...

    elif demo_mode == 'directory':
        print("DEMO: Running in DIRECTORY mode. Data sending to relay not implemented.")
        # ... (Original directory logic) ...


    elif demo_mode == 'webcam':
        # --- Webcam Mode Modifications ---
        if sock is None:
            print("DEMO ERROR: Cannot proceed in webcam mode without connection to relay server.")
            return # Exit if connection failed earlier

        print(f'DEMO: Webcam Demo options: \n {args}')
        print("DEMO: Using device:", tester.device) # Log the device being used

        print("DEMO: Initializing Multi-Person Tracker...")
        try:
            mot = MPT(
                device=tester.device, batch_size=1, display=args.display,
                detector_type=args.detector, output_format='dict',
                yolo_img_size=args.yolo_img_size,
            )
            print("DEMO: Tracker Initialized.")
        except Exception as e:
            print(f"DEMO ERROR: Failed to initialize Multi-Person Tracker: {e}")
            if sock: sock.close()
            return

        # --- Video Capture ---
        print("DEMO: Opening video source...")
        if (stream_mode):
            rtmp_url = "rtmp://35.246.39.155:1935/live/webcam"
            print(f"DEMO: Attempting to connect to RTMP stream: {rtmp_url}")
            cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
        else:
            webcam_idx = 0 # Default webcam index
            print(f"DEMO: Attempting to open webcam (index {webcam_idx})...")
            cap = cv2.VideoCapture(webcam_idx)


        if not cap.isOpened():
            print("DEMO ERROR: Cannot open video source (webcam/stream)")
            if sock: sock.close()
            exit()
        print("DEMO: Video source opened successfully.")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"DEMO: Video source resolution: {frame_width}x{frame_height}, FPS: {video_fps if video_fps > 0 else 'N/A'}")


        print("DEMO: Starting webcam stream. Press 'q' in OpenCV window (if displayed) to exit.")
        print("DEMO: Sending SMPL data to relay server...")

        frame_count = 0
        start_time_proc = time.time()
        last_log_time = time.time()

        while True:
            frame_start_loop = time.time()
            try:
                ret, frame = cap.read()
                if not ret:
                    print("DEMO INFO: End of stream or failed to grab frame.")
                    break # Exit loop if no frame
            except Exception as e:
                print(f"DEMO ERROR: Exception while reading frame: {e}")
                break

            frame_count += 1
            if frame is None: # Should be caught by 'not ret' but double-check
                print("DEMO WARNING: Grabbed frame is None.")
                continue

            # --- Frame Preprocessing ---
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = rgb_frame.shape[:2]
            except Exception as e:
                print(f"DEMO ERROR: Failed to convert frame to RGB: {e}")
                continue # Skip frame if conversion fails

            # --- Detection and Cropping ---
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
                                size = max(w, h) * 1.2 # Add padding factor if needed
                                bbox_prepared = np.array([c_x, c_y, size, size])
                                dets_prepared_list.append(bbox_prepared)

                if dets_prepared_list:
                    dets = np.array(dets_prepared_list)
                else:
                    dets = np.array([])

            except Exception as e:
                print(f"DEMO ERROR: Exception during object detection: {e}")
                continue

            # --- Handle No Detections ---
            if dets.shape[0] == 0:
                # Send T-Pose if no detections are found
                poses_body_send = np.zeros((1, POSE_BODY_DIM), dtype=np.float32)
                poses_root_send = np.zeros((1, 3), dtype=np.float32)
                betas_send = np.zeros((1, NUM_BETAS), dtype=np.float32)
                trans_send = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                data_to_send = {
                    "poses_body": poses_body_send, "poses_root": poses_root_send,
                    "betas": betas_send, "trans": trans_send,
                }
                if not send_data(sock, data_to_send):
                    print("DEMO ERROR: Failed to send T-pose data (no detections). Exiting.")
                    break
                
                if args.display:

                    fps_loop = 1.0 / (time.time() - frame_start_loop) if (time.time() - frame_start_loop) > 0 else 0
                    cv2.putText(frame, f"FPS: {fps_loop:.2f} (No Detections)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Webcam Demo - POCO", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # --- Prepare Batch for POCO ---
            inp_images = []
            bbox_info_list = []
            focal_lengths_list = []
            scales_list = []
            centers_list = []
            orig_shapes_list = []
            try:
                for det in dets:
                    if len(det) == 4:
                        norm_img, _, _ = get_single_image_crop_demo(
                            rgb_frame, det, kp_2d=None, scale=1.0,
                            crop_size=tester.model_cfg.DATASET.IMG_RES
                        )
                        center = [det[0], det[1]]
                        scale_val = det[2] / 200.0
                        inp_images.append(norm_img.float())
                        orig_shape = [orig_h, orig_w]
                        centers_list.append(center)
                        orig_shapes_list.append(orig_shape)
                        scales_list.append(scale_val)
                        bbox_info = calculate_bbox_info(center, scale_val, orig_shape)
                        bbox_info_list.append(bbox_info)
                        focal_length = calculate_focal_length(orig_h, orig_w)
                        focal_lengths_list.append(focal_length)
                    # else: print(f"DEMO WARNING: Skipping invalid detection format in batch prep: {det}") # Optional Debug

                if not inp_images:
                    print("DEMO WARNING: No valid detections to process after batch prep.")
                    continue

                inp_images_tensor = torch.stack(inp_images).to(tester.device)
                batch = {
                    'img': inp_images_tensor,
                    'bbox_info': torch.FloatTensor(bbox_info_list).to(tester.device),
                    'focal_length': torch.FloatTensor(focal_lengths_list).to(tester.device),
                    'scale': torch.FloatTensor(scales_list).to(tester.device),
                    'center': torch.FloatTensor(centers_list).to(tester.device),
                    'orig_shape': torch.FloatTensor(orig_shapes_list).to(tester.device),
                }
            except Exception as e:
                print(f"DEMO ERROR: Exception during batch preparation: {e}")
                continue


            # --- Run POCO Inference ---
            output = None
            try:
                tester.model.eval()
                with torch.no_grad():
                    output = tester.model(batch)
                if output is None:
                    print("DEMO WARNING: Model inference returned None.")
                    continue
            except Exception as e:
                print(f"DEMO ERROR: Exception during model inference: {e}")
                continue

            # --- *** DATA EXTRACTION AND SENDING (WITH POSE CONVERSION) *** ---
            pred_pose_raw = output.get('pred_pose')   # Could be (B, 24, 3, 3) or (B, 72)
            pred_shape = output.get('pred_shape')     # Expecting (B, 10) betas
            pred_cam = output.get('pred_cam')         # Expecting (B, 3) [s, tx, ty]
            pred_uncert = output.get('var_pose')         # Expecting (B, 24, 3, 3) or (B, 72)
            # print(output.keys())
            
            #print(pred_uncert, pred_uncert.shape)

            # Default values (T-Pose)
            poses_body_send = np.zeros((1, POSE_BODY_DIM), dtype=np.float32)
            poses_root_send = np.zeros((1, 3), dtype=np.float32)
            betas_send = np.zeros((1, NUM_BETAS), dtype=np.float32)
            trans_send = np.array([[0.0, 0.8, 0.0]], dtype=np.float32)

            data_valid = False
            pred_pose_aa = None # Variable to hold axis-angle pose

            if pred_pose_raw is not None and pred_shape is not None and pred_cam is not None:
                if pred_pose_raw.shape[0] > 0 and pred_shape.shape[0] > 0 and pred_cam.shape[0] > 0:
                    batch_size_out = pred_pose_raw.shape[0]
                    if not (batch_size_out == pred_shape.shape[0] == pred_cam.shape[0]):
                        print(f"DEMO WARNING: Mismatched batch sizes in output tensors "
                            f"(pose: {pred_pose_raw.shape[0]}, shape: {pred_shape.shape[0]}, cam: {pred_cam.shape[0]}). Sending T-Pose.")
                    else:
                        # --- Check pose format and convert if necessary ---
                        try:
                            current_pose_shape = tuple(pred_pose_raw.shape[1:]) # Get shape excluding batch dim

                            if current_pose_shape == (24, 3, 3): # Rotation Matrix format
                                #print("DEMO DEBUG: Pose is rotation matrix (24, 3, 3). Converting to axis-angle.")
                                if rotation_matrix_to_angle_axis is None:
                                    print("DEMO ERROR: rotation_matrix_to_angle_axis function not available. Cannot convert pose. Sending T-Pose.")
                                else:
                                    # Reshape, convert, reshape back
                                    pred_pose_rotmat_flat = pred_pose_raw.reshape(-1, 3, 3)
                                    pred_pose_aa_flat = rotation_matrix_to_angle_axis(pred_pose_rotmat_flat)
                                    pred_pose_aa = pred_pose_aa_flat.reshape(batch_size_out, 72) # Reshape to (B, 72)
                                    # print(f"DEMO DEBUG: Conversion complete. New pose shape: {pred_pose_aa.shape}")

                            elif current_pose_shape == (72,): # Axis-angle format (B, 72)
                                print("DEMO DEBUG: Pose is already axis-angle (72,). Using directly.")
                                pred_pose_aa = pred_pose_raw

                            elif current_pose_shape == (24, 3): # Axis-angle format (B, 24, 3)
                                print("DEMO DEBUG: Pose is axis-angle (24, 3). Reshaping to (B, 72).")
                                pred_pose_aa = pred_pose_raw.reshape(batch_size_out, 72)

                            else:
                                print(f"DEMO WARNING: Unrecognized pose format shape {current_pose_shape}. Expected (24, 3, 3) or (72,) or (24, 3). Sending T-Pose.")

                        except Exception as e:
                            print(f"DEMO ERROR: Exception during pose format check/conversion: {e}. Sending T-Pose.")
                            pred_pose_aa = None # Ensure it's None on error

                        # --- Extract data for the first person using the converted pose (pred_pose_aa) ---
                        if pred_pose_aa is not None:
                            try:
                                # Extract data for person 0 from the potentially converted tensor
                                pose_person0_aa = pred_pose_aa[0] # Get the full (72,) tensor for person 0
                                shape_person0 = pred_shape[0].cpu().numpy()     # Shape (10,)
                                cam_person0 = pred_cam[0].cpu().numpy()         # Shape (3,)

                                # --- Apply 180-degree X-axis rotation to global orientation ---
                                if batch_rodrigues is not None and rotation_matrix_to_angle_axis is not None:
                                    global_orient_aa = pose_person0_aa[:3] # Original global orientation (axis-angle)
                                    body_pose_aa = pose_person0_aa[3:]   # Original body pose (axis-angle)

                                    # Convert global orientation to rotation matrix using batch_rodrigues
                                    global_orient_rotmat = batch_rodrigues(global_orient_aa.unsqueeze(0)) # Add batch dim -> (1, 3, 3)

                                    # Create 180-degree rotation matrix around X-axis
                                    rot_180_x = torch.tensor([[[1.0, 0.0, 0.0],
                                                            [0.0, -1.0, 0.0],
                                                            [0.0, 0.0, -1.0]]], dtype=global_orient_rotmat.dtype, device=global_orient_rotmat.device)

                                    # Apply the rotation (pre-multiply)
                                    rotated_global_orient_rotmat = torch.bmm(rot_180_x, global_orient_rotmat)

                                    # Convert back to axis-angle
                                    rotated_global_orient_aa = rotation_matrix_to_angle_axis(rotated_global_orient_rotmat).squeeze(0) # Remove batch dim -> (3,)

                                    # Use the rotated global orientation
                                    poses_root_extracted = rotated_global_orient_aa.cpu().numpy()
                                    poses_body_extracted = body_pose_aa.cpu().numpy() # Body pose remains the same
                                    #print("DEMO DEBUG: Applied 180-deg X rotation to global orient.")
                                else:
                                    print("DEMO WARNING: Geometry functions unavailable, cannot apply orientation fix. Using original pose.")
                                    # Fallback to original extraction if functions are missing
                                    poses_root_extracted = pose_person0_aa[:3].cpu().numpy()
                                    poses_body_extracted = pose_person0_aa[3:].cpu().numpy()
                                # --- End Rotation Fix ---

                                betas_extracted = shape_person0[:NUM_BETAS]

                                # Final check on extracted shapes (should still be correct)
                                if poses_root_extracted.shape == (3,) and poses_body_extracted.shape == (POSE_BODY_DIM,) and betas_extracted.shape[0] >= NUM_BETAS:

                                    # --- Calculate Translation (Revert the flip) ---
                                    tx = cam_person0[1]
                                    ty = cam_person0[2]
                                    vertical_offset = 0.8 # Example fixed vertical offset (Keep this or your original logic)
                                    depth_z = 0.0 # Example fixed depth (Keep this or your original logic)
                                    trans_extracted = np.array([tx, ty + vertical_offset, depth_z], dtype=np.float32) # Use original ty and depth_z
                                    # --- End Translation Calculation ---


                                    # Update send variables
                                    poses_body_send = poses_body_extracted.reshape(1, POSE_BODY_DIM)
                                    poses_root_send = poses_root_extracted.reshape(1, 3)
                                    betas_send = betas_extracted.reshape(1, NUM_BETAS)
                                    trans_send = trans_extracted.reshape(1, 3)
                                    data_valid = True
                                    # print(f"DEMO DEBUG: Extracted valid SMPL params after orientation fix.")

                                else:
                                    print(f"DEMO WARNING: Extracted pose/shape dimensions incorrect after orientation fix. "
                                        f"Root: {poses_root_extracted.shape}, Body: {poses_body_extracted.shape}, Betas: {betas_extracted.shape}. Sending T-Pose.")
                                    data_valid = False # Ensure T-pose is sent if shapes mismatch

                            except Exception as e:
                                print(f"DEMO ERROR: Exception during data extraction/orientation fix for person 0: {e}. Sending T-Pose.")
                                data_valid = False # Ensure T-pose on error
                        # else: # pred_pose_aa is None due to conversion error or unrecognized format
                            # print("DEMO DEBUG: Skipping extraction because pred_pose_aa is None.") # Already handled by data_valid=False

                else:
                    print("DEMO WARNING: Model output tensors are empty (batch size 0). Sending T-Pose.")
            else:
                missing_keys = [k for k in ['pred_pose', 'pred_shape', 'pred_cam'] if output.get(k) is None]
                print(f"DEMO WARNING: Keys {missing_keys} not found in model output dictionary. Sending T-Pose.")

            # Package data (will be T-pose if data_valid is False)
            data_to_send = {
                "poses_body": poses_body_send,
                "poses_root": poses_root_send,
                "betas": betas_send,
                "trans": trans_send,
            }
            
            # fps_loop = 1.0 / (time.time() - frame_start_loop) if (time.time() - frame_start_loop) > 0 else 0
            # frame_show = frame.copy()
            # cv2.putText(frame_show, f"FPS: {fps_loop:.2f}", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imshow("Webcam Demo - POCO", frame_show)
            
            
            # if not args.display:
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         print("DEMO: 'q' pressed, exiting.")
            #         break
            
            

            # Send data over socket
            try:
                if not send_data(sock, data_to_send):
                    print("DEMO ERROR: Failed to send data to relay server. Exiting.")
                    break
            except Exception as e:
                print(f"DEMO ERROR: Exception during socket send: {e}. Exiting.")
                break


            # --- Optional Display ---
            # (Display logic remains the same as before, using the 'output' dict which might contain smpl_vertices)
            if args.display:
                try:
                    render_frame_display = frame.copy()
                    smpl_vertices = output.get('smpl_vertices') # Vertices computed based on the RAW model output pose format

                    # If display needs vertices based on the *converted* pose, you'd need to run SMPL again here
                    # with pred_pose_aa, pred_shape, pred_cam if data_valid is True.
                    # This adds computational overhead. The current code displays vertices based on the model's internal pose.

                    if smpl_vertices is not None and smpl_vertices.shape[0] > 0 and pred_cam is not None:
                        orig_cam = convert_crop_cam_to_orig_img(
                            cam=pred_cam.cpu().numpy(), # Use the original pred_cam
                            bbox=dets,
                            img_width=orig_w, img_height=orig_h
                        )
                        renderer = Renderer(resolution=(orig_w, orig_h), orig_img=True, wireframe=args.wireframe)
                        render_frame_display_rgb = cv2.cvtColor(render_frame_display, cv2.COLOR_BGR2RGB)
                        num_to_render = min(len(dets), smpl_vertices.shape[0], len(orig_cam))
                        for i in range(num_to_render):
                            verts = smpl_vertices[i].cpu().numpy()
                            if np.isnan(verts).any() or np.isinf(verts).any():
                                # print(f"DEMO WARNING: NaN/Inf detected in vertices for person {i}, skipping render.") # Optional Debug
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
                        print("DEMO: 'q' pressed, exiting.")
                        break
                except Exception as e:
                    print(f"DEMO ERROR: Exception during display update: {e}")
                
            else:
                # pass
                time.sleep(0.2)  # Small delay to avoid high CPU usage when not displaying


            # Log performance periodically
            current_time = time.time()
            if current_time - last_log_time >= 10.0:
                loop_time = current_time - frame_start_loop
                print(f"DEMO DEBUG: Frame {frame_count}, Loop Time: {loop_time:.4f}s")
                last_log_time = current_time


        # --- Cleanup ---
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        if sock:
            print("DEMO: Closing socket connection.")
            sock.close()
        print("DEMO: Webcam demo finished.")
        end_time_proc = time.time()
        total_time = end_time_proc - start_time_proc
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"DEMO: Processed {frame_count} frames in {total_time:.2f} seconds. Average FPS: {avg_fps:.2f}")

    # --- End of webcam mode section ---

    else:
        print(f"DEMO ERROR: Invalid demo mode selected: {demo_mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --- Essential Arguments ---
    parser.add_argument('--cfg', type=str, default='configs/demo_poco_cliff.yaml', help='config file')
    parser.add_argument('--ckpt', type=str, default='data/poco_cliff.pt', help='checkpoint path')
    parser.add_argument('--mode', default='webcam', choices=['video', 'folder', 'directory', 'webcam'], help='Demo type')

    # --- Arguments for different modes ---
    parser.add_argument('--vid_file', type=str, default=None, help='input video path or youtube link')
    parser.add_argument('--image_folder', type=str, default=None, help='input image folder')
    parser.add_argument('--output_folder', type=str, default='out', help='output folder')
    parser.add_argument('--stream', type=str2bool, default=False, help='RTMP stream input')

    # --- Detection/Tracking Arguments ---
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'], help='object detector')
    parser.add_argument('--yolo_img_size', type=int, default=256, help='yolo input size')
    parser.add_argument('--tracker_batch_size', type=int, default=1, help='tracker batch size') # Use 1 for webcam

    # --- POCO/Model Arguments ---
    parser.add_argument('--batch_size', type=int, default=32, help='POCO batch size')

    # --- Optional Arguments ---
    parser.add_argument('--display', action='store_true', help='display intermediate results (OpenCV window)')
    parser.add_argument('--smooth', action='store_true', help='smooth results (if implemented in POCO Tester)')
    parser.add_argument('--min_cutoff', type=float, default=0.004, help='one euro min cutoff')
    parser.add_argument('--beta', type=float, default=1.5, help='one euro beta')
    parser.add_argument('--no_kinematic_uncert', action='store_false',
                        help='Do not use SMPL Kinematic for uncert')
    parser.add_argument('--wireframe', action='store_true', help='render wireframes in demo display')
    # Add any other relevant args from original demo.py if needed by your config/model
    parser.add_argument('--exp', type=str, default='', help='experiment description')
    parser.add_argument('--inf_model', type=str, default='best', help='select model from checkpoint')
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
    main(args)